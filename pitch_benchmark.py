"""The frame track's per-cell measurement (pure library -- evaluate.py orchestrates).

run_single_evaluation scores ONE algorithm on ONE built dataset with a streaming threshold
sweep; it writes nothing, caches nothing, spawns nothing.
"""
import gc

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from algorithms import build_algorithm
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
