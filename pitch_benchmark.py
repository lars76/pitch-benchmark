import argparse
import gc
import importlib.util
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from algorithms import (
    build_algorithm,
    get_algorithm,
    list_algorithms,
    resolve_requested_algorithms,
)
from datasets import (
    REGISTRY,
    Augment,
    Truncate,
    build_pipeline,
    get_pitch_dataset,
    list_pitch_datasets,
    subset,
)
from metrics import (
    voicing_boundary_latency,
    clip_and_group,
    DEFAULT_THRESHOLDS,
    MetricAccumulator,
    is_voiced,
    summarize_threshold_sweep,
    to_json_safe,
)


def _failure_dict(skipped_samples: int) -> dict:
    """NaN-filled result for an algorithm whose run was invalidated (a clip raised, or no data)."""
    return {
        "voicing_detection": {"f1": np.nan, "precision": np.nan, "recall": np.nan},
        "pitch_accuracy": {
            "rmse": np.nan, "cents_error": np.nan, "rpa": np.nan, "rca": np.nan,
            "octave_error_rate": np.nan, "gross_error_rate": np.nan, "valid_frames": 0,
        },
        "smoothness_metrics": {"relative_smoothness": np.nan, "continuity_breaks": np.nan},
        "combined_score": np.nan,
        "optimal_threshold": np.nan,
        "threshold_sweep": [],
        "coverage": {"clips_evaluated": 0, "clips_skipped": skipped_samples},
    }


def run_single_evaluation(
    dataset: Dataset,
    algorithm_class: object,
    thresholds: np.ndarray,
    device: str = "auto",
) -> tuple[dict, bool]:
    """Evaluate one algorithm on one dataset with a streaming, O(1)-memory threshold sweep.

    Metrics fold in one clip at a time (metrics.MetricAccumulator, one per threshold), so runner
    memory is independent of dataset size and clip length, and identical to evaluating on the fully
    concatenated arrays.

    Returns ``(metrics, crashed)``. ``crashed`` is True only when a clip raised (e.g. OOM); the caller
    stamps it into ``metadata.crashed`` so the report counts it as 0 (not dropped) and the failed cell
    is cached like any other. A deterministically empty run (all clips unvoiced, or no finite
    threshold) returns crashed=False -- it is a normal cached result.

    Memory: device cache is released once per algorithm at teardown, NOT per clip. PyTorch's
    caching allocator reuses freed blocks within a run, so emptying per clip only adds malloc churn;
    the per-call peak is already bounded by each algorithm's internal windowing. A light periodic
    gc() guards against reference cycles left by some DSP libraries.
    """
    algo_name = algorithm_class.get_name()
    # Construction can fail (e.g. a missing backend/model like BasicPitch's ONNX runtime). Record it as
    # a crash -- same convention as a per-clip failure -- so one tracker's bad env can't abort the run.
    try:
        algo = build_algorithm(
            algorithm_class, dataset.sample_rate, dataset.hop_size, dataset.fmin, dataset.fmax,
            device=device,
        )
    except Exception as e:
        tqdm.write(f"FATAL: {algo_name} failed to build ({e}). Recording as crashed.")
        return to_json_safe(_failure_dict(0)), True

    accumulators = [MetricAccumulator() for _ in thresholds]
    per_clip_rows = [[] for _ in thresholds]   # see per_clip.schema below
    latencies = [([], []) for _ in thresholds]  # pooled (onset_ms, offset_ms) region latencies
    skipped_samples = 0
    clips_evaluated = 0
    did_fail = False

    # Iterate the dataset directly (one sample dict at a time, in order), single-process on purpose:
    # the per-dataset in-memory decode cache (built on the first algorithm's pass) is reused by every
    # later algorithm only when the dataset lives in this process; DataLoader workers would re-decode
    # per algorithm and discard that cache each epoch.
    sample_pbar = tqdm(
        range(len(dataset)), desc=f"{algo_name}", leave=False, unit=" samples"
    )

    for idx in sample_pbar:
        try:
            sample = dataset[idx]
            audio = sample["audio"].numpy()
            true_pitch = sample["pitch"].numpy()
            true_voicing = sample["periodicity"].numpy()   # voicing confidence in [0,1]
            # Optional GT pitch-confidence (laryngograph speech datasets); None elsewhere -> RPA not gated.
            pc = sample.get("pitch_conf")
            pc = pc.numpy() if pc is not None else None

            if not is_voiced(true_voicing).any():
                skipped_samples += 1
                continue

            results = algo.extract_pitch(
                audio, thresholds=list(thresholds), compute_notes=False
            )
            if len(results) != len(thresholds):
                raise ValueError(
                    f"Expected {len(thresholds)} results, got {len(results)}"
                )

            clip_id, group = clip_and_group(dataset, sample.get("wav_path"), idx)
            for ti, (acc, (pred_pitch, pred_voicing, _)) in enumerate(zip(accumulators, results)):
                acc.update(pred_pitch, pred_voicing, true_pitch, true_voicing, pitch_conf=pc)
                # per-clip metrics: a fresh single-clip accumulator (same code path as the aggregate)
                one = MetricAccumulator()
                one.update(pred_pitch, pred_voicing, true_pitch, true_voicing, pitch_conf=pc)
                v, pm = one.voicing_metrics(), one.pitch_metrics()
                frame_period = dataset.hop_size / dataset.sample_rate
                on_ms, off_ms = voicing_boundary_latency(
                    (np.asarray(pred_voicing).astype(bool) & (np.asarray(pred_pitch) > 0)),
                    is_voiced(true_voicing), frame_period)
                latencies[ti][0].extend(on_ms); latencies[ti][1].extend(off_ms)
                ss = one.suff_stats()   # additive sufficient stats -> frame-weighted cluster-CIs in the report
                per_clip_rows[ti].append([
                    clip_id, group, int(len(true_pitch)),
                    round(float(pm.get("rpa", float("nan"))), 4),
                    round(float(one.pitch_coverage()), 4),
                    round(float(v.get("f1", float("nan"))), 4),
                    round(float(np.median(on_ms)), 1) if on_ms else None,
                    round(float(np.median(off_ms)), 1) if off_ms else None,
                    round(float(pm.get("cents_error", float("nan"))), 2),
                    round(float(one.combined_score()), 4),
                    ss["valid"], ss["n_rpa"], round(ss["sum_cents"], 2), ss["n_octave"], ss["n_gross"],
                    ss["tp"], ss["fp"], ss["fn"],
                ])
            clips_evaluated += 1

        except Exception as e:
            tqdm.write(
                f"FATAL: {algo_name} failed on sample {idx}. "
                f"Aborting this algorithm. Error: {e}"
            )
            did_fail = True
            break

        finally:
            # Cycle insurance: some DSP libs leave reference cycles holding arrays. Python's GC
            # collects these anyway; an occasional explicit pass bounds the lag cheaply. No per-clip
            # empty_cache (see docstring).
            if (idx + 1) % 200 == 0:
                gc.collect()

    sample_pbar.close()

    # Release the model + this algorithm's cached device memory before the next algorithm builds.
    # The right place for empty_cache: it returns the now-unused reserved pool to other processes.
    # (TensorFlow keeps its own pool until the process exits, by design.)
    del algo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if did_fail or clips_evaluated == 0:
        return to_json_safe(_failure_dict(skipped_samples)), did_fail

    if skipped_samples > 0:
        tqdm.write(f"  ({algo_name} skipped {skipped_samples} unvoiced samples)")

    best_idx, best_metrics = summarize_threshold_sweep(accumulators, thresholds)
    if best_idx < 0:
        return to_json_safe(_failure_dict(skipped_samples)), False
    best_metrics["coverage"] = {
        "clips_evaluated": clips_evaluated, "clips_skipped": skipped_samples,
    }
    on_ms, off_ms = latencies[best_idx]
    best_metrics["voicing_latency"] = {
        "onset_median_ms": float(np.median(on_ms)) if on_ms else None,
        "onset_p90_ms": float(np.percentile(on_ms, 90)) if on_ms else None,
        "offset_median_ms": float(np.median(off_ms)) if off_ms else None,
        "offset_p90_ms": float(np.percentile(off_ms, 90)) if off_ms else None,
        "n_regions": len(on_ms),
    }
    best_metrics["per_clip"] = {
        "threshold": float(thresholds[best_idx]),
        # derived per-clip values (rpa..combined) + the additive SUFFICIENT STATS (valid..fn) that let
        # the report recompute frame-weighted aggregates per cluster-bootstrap resample (incl. combined).
        "schema": ["clip_id", "group", "n_frames", "rpa", "pitch_coverage", "voicing_f1",
                   "onset_lat_ms", "offset_lat_ms", "cents_mae", "combined",
                   "valid", "n_rpa", "sum_cents", "n_octave", "n_gross", "tp", "fp", "fn"],
        "rows": per_clip_rows[best_idx],
    }
    return to_json_safe(best_metrics), False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single pitch benchmark task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--dataset", type=str, required=True, choices=list_pitch_datasets()
    )
    required.add_argument("--data-dir", type=str, required=True)
    required.add_argument("--output-dir", type=str, default="results")
    # --chime-dir / --demand-dir are only needed for the "chime" / "demand" degradations.
    parser.add_argument("--chime-dir", type=str, default=None)
    parser.add_argument("--demand-dir", type=str, default=None)
    parser.add_argument(
        "--degradation",
        type=str,
        default="clean",
        choices=list(REGISTRY),
    )
    # Robustness probe: cap clips and/or truncate duration (clean leaderboard uses neither).
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,  # None => run every installed algorithm (resolved after parsing)
        choices=list_algorithms(),  # all known names; an uninstalled one is reported by get_algorithm
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for the device-aware trackers (auto picks cuda->mps->cpu). Pin cpu or "
        "cuda for reproducible leaderboard numbers; mps is a local speed option (numerics differ).",
    )
    args = parser.parse_args()

    # Resolve the algorithm set. No --algorithms flag => run every algorithm whose backend is
    # installed, so a partial install (uv sync --extra ...) just works. An explicit list is honored
    # exactly; a named-but-uninstalled backend raises a precise error in the loop below.
    args.algorithms = resolve_requested_algorithms(
        args.algorithms, report_skipped=True, on_empty=parser.error
    )

    # Validate every degradation prerequisite up front: a missing dir or backend would otherwise
    # surface only after expensive setup (dataset scan / model load), deep in the eval loop.
    if args.degradation == "chime" and not args.chime_dir:
        parser.error("--chime-dir is required when --degradation chime")
    if args.degradation == "demand" and not args.demand_dir:
        parser.error("--demand-dir is required when --degradation demand")
    if args.degradation == "room" and importlib.util.find_spec("pyroomacoustics") is None:
        parser.error("--degradation room requires pyroomacoustics (core dependency; run `uv sync`)")

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    thresholds = DEFAULT_THRESHOLDS

    print(
        f"--- Starting benchmark for dataset '{args.dataset}' with seed {args.seed} ---"
    )
    base_dataset = get_pitch_dataset(args.dataset)(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        hop_size=args.hop_size,
    )

    # Robustness probe: cap (built-in Subset) + truncate. The clean leaderboard sets neither
    # (full datasets); robustness runs share one probe across conditions.
    is_probe = bool(args.max_samples or args.max_seconds)
    if args.max_samples and args.max_samples < len(base_dataset):
        idxs = sorted(
            {round(i) for i in np.linspace(0, len(base_dataset) - 1, args.max_samples)}
        )
        base_dataset = subset(base_dataset, idxs)
    if args.max_seconds:
        base_dataset = Truncate(base_dataset, args.max_seconds)

    # One augmentation layer applies the chosen pipeline (clean = empty = pass-through).
    pipeline = build_pipeline(
        args.degradation, chime_dir=args.chime_dir, demand_dir=args.demand_dir,
        sample_rate=args.sample_rate,
    )
    eval_dataset = Augment(base_dataset, pipeline, seed=args.seed)

    for algo_name in args.algorithms:
        algo_class = get_algorithm(algo_name)
        # Device only changes results for device-aware trackers (DSP/own-runtime are cpu). It is part
        # of the cache key AND the filename, so cpu/gpu results coexist rather than overwrite -- a cpu
        # re-run no longer clobbers a gpu file, and the leaderboard cannot silently mix devices.
        effective_device = algo_class.resolve_effective_device(args.device)
        param_str = (
            f"{args.degradation}_"  # condition, e.g. clean / chime / pink
            # probe runs encode their size so a different probe doesn't reuse stale results
            + (f"probe-n{args.max_samples}-t{args.max_seconds}_" if is_probe else "")
            + f"sr{int(args.sample_rate / 1000)}k_"  # e.g., sr16k
            + f"hop{args.hop_size}_"  # e.g., hop256
            + effective_device  # cpu / cuda / mps
        )
        result_path = os.path.join(
            args.output_dir,
            f"{args.dataset}_{algo_name}_{param_str}_seed{args.seed}.json",
        )
        # Cache-as-done: any existing result is skipped, a recorded crash included (it is a finished
        # failed cell -- delete the file to force a re-run).
        if os.path.exists(result_path):
            tqdm.write(f"Skipping: {os.path.basename(result_path)} already exists.")
            continue

        start_time = time.time()
        metrics, crashed = run_single_evaluation(
            dataset=eval_dataset,
            algorithm_class=algo_class,
            thresholds=thresholds,
            device=args.device,
        )
        execution_time = time.time() - start_time

        if crashed:
            tqdm.write(f"  ({algo_name} crashed; recorded as a failed cell -- delete the file to redo)")

        score = metrics.get("combined_score")
        threshold = metrics.get("optimal_threshold")

        tqdm.write(
            f"Finished {algo_name} in {execution_time:.2f}s. "
            f"Score: {f'{score:.4f}' if score is not None else 'N/A'} @ "
            f"Threshold: {f'{threshold:.2f}' if threshold is not None else 'N/A'}"
        )

        full_result = {
            "metadata": {
                "algorithm_name": algo_name,
                "dataset_name": args.dataset,
                "condition": args.degradation,
                "probe": is_probe,
                "seed": args.seed,
                "device": effective_device,
                "crashed": crashed,   # recorded failure -> report counts it as 0 (shared with ood_benchmark)
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "execution_time_seconds": round(execution_time, 2),
            },
            "parameters": {
                "sample_rate": args.sample_rate,
                "hop_size": args.hop_size,
                "max_samples": args.max_samples,
                "max_seconds": args.max_seconds,
                "fmin": eval_dataset.fmin,
                "fmax": eval_dataset.fmax,
            },
            "results": metrics,
        }

        with open(result_path, "w") as f:
            json.dump(to_json_safe(full_result), f, indent=4)
        print(f"Success: Saved result to {os.path.basename(result_path)}")

    print(f"\n--- Benchmark run for seed {args.seed} finished. ---")
