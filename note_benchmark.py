"""The note track's per-cell measurement (a pure library; evaluate.py orchestrates).

run_note_evaluation scores ONE algorithm on ONE built dataset; it writes nothing, caches
nothing, spawns nothing. Datasets must return ground-truth "notes" (list of {start, end,
midi_pitch}), currently Vocadito and URMP.

Fairness design: every contour tracker is segmented by the benchmark's own note layer
(algorithms.base.notes_from_pitch_contour: exact changepoint DP + audio-derived boundary
gates). The voicing threshold AND the split penalty lam are swept per algorithm, so each
tracker is ranked at its own best note operating point, the same per-algorithm
optimal-threshold policy as the pitch track. The gates read the audio (identical input
for every tracker), so they cannot favor any one algorithm.

Scoring: mir_eval transcription F1s on CONTINUOUS-Hz pitch (no semitone rounding on
either side): COnP (onset 50 ms + pitch 50 cents) and COnPOff (+ offset within
max(20% duration, 50 ms)). Results carry per-clip rows for CI rendering.
"""
import gc

import mir_eval
import numpy as np
import torch
from tqdm import tqdm

from algorithms import build_algorithm

LAM_GRID = [250.0, 375.0, 500.0, 750.0]


def _notes_to_arrays(notes):
    iv = np.array([[n["start"], n["end"]] for n in notes], dtype=float).reshape(-1, 2)
    hz = np.array([440.0 * 2 ** ((n["midi_pitch"] - 69) / 12.0) for n in notes])
    return iv, hz


def score_notes(est_notes, ref_notes):
    """Returns (conp, conpoff) for one clip."""
    if not est_notes or not ref_notes:
        return 0.0, 0.0
    ei, eh = _notes_to_arrays(est_notes)
    ri, rh = _notes_to_arrays(ref_notes)
    # mir_eval requires positive-length intervals
    ok = ei[:, 1] > ei[:, 0]
    ei, eh = ei[ok], eh[ok]
    if len(eh) == 0:
        return 0.0, 0.0
    conp = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rh, ei, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
    )[2]
    conpoff = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rh, ei, eh, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05,
    )[2]
    return float(conp), float(conpoff)


def run_note_evaluation(dataset, algorithm_class, thresholds, device="auto"):
    """(metrics, crashed): sweep threshold x lam, rank at the best mean COnP pair."""
    algo_name = algorithm_class.get_name()
    try:
        algo = build_algorithm(
            algorithm_class, dataset.sample_rate, dataset.hop_size,
            dataset.fmin, dataset.fmax, device=device,
        )
    except Exception as e:
        tqdm.write(f"FATAL: {algo_name} failed to build ({e}). Recording as crashed.")
        return {"conp": np.nan, "conpoff": np.nan}, True

    # scores[(ti, li)] -> list of (conp, conpoff)
    scores = {(ti, li): [] for ti in range(len(thresholds)) for li in range(len(LAM_GRID))}
    crashed = False
    pbar = tqdm(range(len(dataset)), desc=algo_name, leave=False, unit=" clips")
    for idx in pbar:
        try:
            sample = dataset[idx]
            ref_notes = sample.get("notes")
            if not ref_notes:
                continue
            audio = sample["audio"].numpy()
            contours = algo.extract_pitch(
                audio, thresholds=list(thresholds), compute_notes=False
            )
            for ti, (pitch, voicing, _) in enumerate(contours):
                for li, lam in enumerate(LAM_GRID):
                    est = algo.notes_from_pitch_contour(
                        pitch, voicing, audio=audio, lam_per_s=lam
                    )
                    conp, conpoff = score_notes(est, ref_notes)
                    scores[(ti, li)].append((conp, conpoff))
        except Exception as e:
            tqdm.write(f"FATAL: {algo_name} failed on clip {idx}: {e}")
            crashed = True
            break
        finally:
            if (idx + 1) % 200 == 0:
                gc.collect()
    pbar.close()

    # Same teardown as the frame track: release the model + cached device memory before the
    # next algorithm builds in this process.
    del algo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if crashed or not any(scores.values()):
        return {"conp": np.nan, "conpoff": np.nan}, crashed

    means = {k: float(np.mean([r[0] for r in v])) for k, v in scores.items() if v}
    (ti, li) = max(means, key=means.get)
    rows = scores[(ti, li)]
    return {
        "conp": means[(ti, li)],
        "conpoff": float(np.mean([r[1] for r in rows])),
        "optimal_threshold": float(thresholds[ti]),
        "optimal_lam_per_s": LAM_GRID[li],
        "clips_evaluated": len(rows),
    }, False
