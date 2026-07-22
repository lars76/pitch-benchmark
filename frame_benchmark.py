"""The frame track's per-cell measurement (a pure library; evaluate.py orchestrates).

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
    clip_and_group,
    FRAME_STAT_COLS,
    MetricAccumulator,
    is_voiced,
    sweep_summary,
    to_json_safe,
)


def _failure_dict(skipped_samples: int) -> dict:
    """Empty result for an algorithm whose run was invalidated (a clip raised, or no data)."""
    return {
        "thresholds": [],
        "sweep": [],
        "per_clip": {"stats_schema": list(FRAME_STAT_COLS), "clips": [], "stats": []},
    }


def run_single_evaluation(
    dataset: Dataset,
    algorithm_class: object,
    thresholds: np.ndarray,
    device: str = "auto",
) -> tuple[dict, bool, "str | None"]:
    """Evaluate one algorithm on one dataset with a streaming, O(1)-memory threshold sweep.

    Metrics fold in one clip at a time (metrics.MetricAccumulator, one per threshold), so runner
    memory is independent of dataset size and clip length, and identical to evaluating on the fully
    concatenated arrays.

    Returns ``(metrics, crashed, crash_kind)``. ``crashed`` is True only when a clip raised (e.g.
    OOM); ``crash_kind`` is the exception type name then (else None), so the caller can stamp
    ``metadata.crash_kind`` -- the SAME field a spawned segfault records -- and the report groups
    an in-process failure by its kind instead of showing "unknown". A deterministically empty run
    (all clips unvoiced, or no finite threshold) returns crashed=False; it is a normal cached result.

    Memory: device cache is released once per algorithm at teardown, NOT per clip. PyTorch's
    caching allocator reuses freed blocks within a run, so emptying per clip only adds malloc churn;
    the per-call peak is already bounded by each algorithm's internal windowing. A light periodic
    gc() guards against reference cycles left by some DSP libraries.
    """
    algo_name = algorithm_class.get_name()
    # Construction can fail (e.g. a missing model file or inference backend). Record it as
    # a crash, same convention as a per-clip failure, so one tracker's bad env can't abort the run.
    try:
        algo = build_algorithm(
            algorithm_class, dataset.sample_rate, dataset.hop_size, dataset.fmin, dataset.fmax,
            device=device,
        )
    except Exception as e:
        tqdm.write(f"FATAL: {algo_name} failed to build ({e}). Recording as crashed.")
        return to_json_safe(_failure_dict(0)), True, type(e).__name__

    accumulators = [MetricAccumulator() for _ in thresholds]
    clips_meta = []                             # [clip_id, group, n_frames] once per clip
    per_clip_stats = []                         # per clip: one suff-stat row per threshold
    skipped_samples = 0
    clips_evaluated = 0
    did_fail = False
    crash_kind = None                           # the exception type name if a clip raises

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
            clips_meta.append([clip_id, group, int(len(true_pitch))])
            clip_stats = []
            for ti, (acc, (pred_pitch, pred_voicing, _)) in enumerate(zip(accumulators, results)):
                acc.update(pred_pitch, pred_voicing, true_pitch, true_voicing, pitch_conf=pc)
                # per-clip suff stats: a fresh single-clip accumulator (same code path as the
                # aggregate); summing rows across clips reproduces the aggregate EXACTLY
                one = MetricAccumulator()
                one.update(pred_pitch, pred_voicing, true_pitch, true_voicing, pitch_conf=pc)
                ss = one.suff_stats()
                clip_stats.append([
                    ss[c] if not isinstance(ss[c], float) else round(ss[c], 2)
                    for c in FRAME_STAT_COLS
                ])
            per_clip_stats.append(clip_stats)
            clips_evaluated += 1

        except Exception as e:
            tqdm.write(
                f"FATAL: {algo_name} failed on sample {idx}. "
                f"Aborting this algorithm. Error: {e}"
            )
            did_fail = True
            crash_kind = type(e).__name__
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
        return to_json_safe(_failure_dict(skipped_samples)), did_fail, crash_kind

    if skipped_samples > 0:
        tqdm.write(f"  ({algo_name} skipped {skipped_samples} unvoiced samples)")

    sweep = sweep_summary(accumulators, thresholds)
    results = {
        "thresholds": [float(t) for t in thresholds],
        "sweep": sweep,
        # suff-stat rows per clip PER THRESHOLD: threshold choice is a report-time decision
        "per_clip": {"stats_schema": list(FRAME_STAT_COLS),
                     "clips": clips_meta, "stats": per_clip_stats},
    }
    return to_json_safe(results), False, None
