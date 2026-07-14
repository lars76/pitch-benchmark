"""Note-transcription benchmark runner (the note track).

Mirrors pitch_benchmark.py's conventions: one result JSON per (algorithm, dataset,
condition, seed); crashes recorded (not dropped); degradations via datasets.augment,
deterministic per (seed, idx). Datasets must return ground-truth "notes"
(list of {start, end, midi_pitch}) -- currently Vocadito and URMP.

Fairness design: every contour tracker is segmented by the benchmark's own note layer
(algorithms.base.notes_from_pitch_contour: exact changepoint DP + audio-derived boundary
gates). The voicing threshold AND the split penalty lam are swept per algorithm, so each
tracker is ranked at its own best note operating point -- the same per-algorithm
optimal-threshold policy as the pitch track. The gates read the audio (identical input
for every tracker), so they cannot favor any one algorithm.

Scoring: mir_eval transcription F1s on CONTINUOUS-Hz pitch (no semitone rounding on
either side): COnP (onset 50 ms + pitch 50 cents) and COnPOff (+ offset within
max(20% duration, 50 ms)). Results carry per-clip rows for CI rendering.

Usage:
    uv run python note_benchmark.py --dataset Vocadito --data-dir ../datasets/vocadito \
        --algorithms Praat SwiftF0 --output-dir results_notes
"""
import argparse
import subprocess
import sys
import gc
import json
import os
import time
from datetime import datetime, timezone

import mir_eval
import numpy as np
from tqdm import tqdm

from algorithms import build_algorithm, get_algorithm
from datasets import get_pitch_dataset
from datasets.augment import Augment, Truncate, build_pipeline
from metrics import clip_and_group
from pitch_benchmark import to_json_safe

LAM_GRID = [250.0, 375.0, 500.0, 750.0]


def _notes_to_arrays(notes):
    iv = np.array([[n["start"], n["end"]] for n in notes], dtype=float).reshape(-1, 2)
    hz = np.array([440.0 * 2 ** ((n["midi_pitch"] - 69) / 12.0) for n in notes])
    return iv, hz


def score_notes(est_notes, ref_notes):
    """Returns (conp, conpoff, octave_errors) for one clip."""
    if not est_notes or not ref_notes:
        return 0.0, 0.0, 0
    ei, eh = _notes_to_arrays(est_notes)
    ri, rh = _notes_to_arrays(ref_notes)
    # mir_eval requires positive-length intervals
    ok = ei[:, 1] > ei[:, 0]
    ei, eh = ei[ok], eh[ok]
    if len(eh) == 0:
        return 0.0, 0.0, 0
    conp = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rh, ei, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
    )[2]
    conpoff = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rh, ei, eh, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05,
    )[2]
    # octave errors: onset-matched (<=50 ms) est notes whose pitch is ~an octave off
    oct_err = 0
    for s, hz in zip(ei[:, 0], eh):
        j = int(np.argmin(np.abs(ri[:, 0] - s)))
        if abs(ri[j, 0] - s) <= 0.05:
            cents = abs(1200.0 * np.log2(hz / rh[j]))
            if 1150.0 <= cents <= 1250.0:
                oct_err += 1
    return float(conp), float(conpoff), int(oct_err)


def run_note_evaluation(dataset, algorithm_class, thresholds, device="auto", indices=None):
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

    # scores[(ti, li)] -> list of (clip_id, group, n_ref, conp, conpoff, oct)
    scores = {(ti, li): [] for ti in range(len(thresholds)) for li in range(len(LAM_GRID))}
    crashed = False
    pbar = tqdm(indices if indices is not None else range(len(dataset)),
                desc=algo_name, leave=False, unit=" clips")
    for idx in pbar:
        try:
            sample = dataset[idx]
            ref_notes = sample.get("notes")
            if not ref_notes:
                continue
            audio = sample["audio"].numpy()
            # group = the true speaker/singer/piece (get_group), NOT dirname -- so the note-track
            # cluster-bootstrap CIs cluster correctly (shared helper with the frame track).
            clip_id, group = clip_and_group(dataset, sample.get("wav_path"), idx)
            contours = algo.extract_pitch(
                audio, thresholds=list(thresholds), compute_notes=False
            )
            for ti, (pitch, voicing, _) in enumerate(contours):
                for li, lam in enumerate(LAM_GRID):
                    est = algo.notes_from_pitch_contour(
                        pitch, voicing, audio=audio, lam_per_s=lam
                    )
                    conp, conpoff, oct_err = score_notes(est, ref_notes)
                    scores[(ti, li)].append(
                        (clip_id, group, len(ref_notes), conp, conpoff, oct_err)
                    )
        except Exception as e:
            tqdm.write(f"FATAL: {algo_name} failed on clip {idx}: {e}")
            crashed = True
            break
        finally:
            if (idx + 1) % 100 == 0:
                gc.collect()
    pbar.close()

    if crashed or not any(scores.values()):
        return {"conp": np.nan, "conpoff": np.nan}, crashed

    means = {k: float(np.mean([r[3] for r in v])) for k, v in scores.items() if v}
    (ti, li) = max(means, key=means.get)
    rows = scores[(ti, li)]
    return {
        "conp": means[(ti, li)],
        "conpoff": float(np.mean([r[4] for r in rows])),
        "octave_errors_total": int(sum(r[5] for r in rows)),
        "optimal_threshold": float(thresholds[ti]),
        "optimal_lam_per_s": LAM_GRID[li],
        "clips_evaluated": len(rows),
        "per_clip": {
            "threshold": float(thresholds[ti]),
            "lam_per_s": LAM_GRID[li],
            "schema": ["clip_id", "group", "n_ref_notes", "conp", "conpoff", "octave_errors"],
            "rows": [list(r) for r in rows],
        },
    }, False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Dataset with note GT (Vocadito, URMP)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--algorithms", nargs="+", required=True)
    ap.add_argument("--output-dir", default="results_notes")
    ap.add_argument("--degradation", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--hop-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chime-dir", default=None)
    ap.add_argument("--demand-dir", default=None)
    ap.add_argument("--no-isolate", action="store_true",
                    help="Run algorithms in-process (internal; used by the isolation re-invoke)")
    ap.add_argument("--cell-timeout", type=int, default=1800,
                    help="Seconds per (algorithm, dataset) subprocess when isolated")
    args = ap.parse_args()

    # Per-algorithm subprocess isolation (same rationale as ood_benchmark.py): some C-extension
    # trackers (pyreaper, pysptk) ABORT (SIGSEGV/SIGABRT) on some inputs; an abort must cost only
    # its own cell, recorded as crashed -- not kill the batch.
    if not args.no_isolate and len(args.algorithms) >= 1:
        for algo_name in args.algorithms:
            cond = args.degradation or "clean"
            fname = f"notes_{algo_name}_{args.dataset}_{cond}_seed{args.seed}.json"
            result_path = os.path.join(args.output_dir, fname)
            if os.path.exists(result_path):
                print(f"skip {fname} (exists; delete to redo)")
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--no-isolate",
                   "--dataset", args.dataset, "--data-dir", args.data_dir,
                   "--algorithms", algo_name, "--output-dir", args.output_dir,
                   "--seed", str(args.seed), "--sample-rate", str(args.sample_rate),
                   "--hop-size", str(args.hop_size), "--device", args.device]
            if args.degradation: cmd += ["--degradation", args.degradation]
            if args.max_samples: cmd += ["--max-samples", str(args.max_samples)]
            if args.max_seconds: cmd += ["--max-seconds", str(args.max_seconds)]
            if args.chime_dir: cmd += ["--chime-dir", args.chime_dir]
            if args.demand_dir: cmd += ["--demand-dir", args.demand_dir]
            kind = None
            try:
                out = subprocess.run(cmd, timeout=args.cell_timeout)
                if out.returncode != 0:
                    kind = f"hard (exit {out.returncode})"
            except subprocess.TimeoutExpired:
                kind = f"timeout (> {args.cell_timeout}s)"
            if not os.path.exists(result_path):
                # Temp file + atomic rename, same constraint as the main result write below.
                tmp_path = result_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump({"metadata": {"track": "notes", "algorithm_name": algo_name,
                        "dataset_name": args.dataset, "condition": cond, "seed": args.seed,
                        "crashed": True, "crash_kind": kind or "no output",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                        "parameters": {}, "results": {"conp": None, "conpoff": None}}, f, indent=4)
                os.replace(tmp_path, result_path)
                print(f"CRASHED cell written for {algo_name} ({kind or 'no output'})")
            elif kind:
                print(f"note: {algo_name} exited abnormally ({kind}) but wrote its result")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    thresholds = np.round(np.arange(0.0, 1.01, 0.1), 2)

    base = get_pitch_dataset(args.dataset)(
        root_dir=args.data_dir, sample_rate=args.sample_rate, hop_size=args.hop_size
    )
    if args.max_seconds:
        base = Truncate(base, args.max_seconds)
    indices = None
    if args.max_samples and args.max_samples < len(base):
        indices = sorted({round(i) for i in np.linspace(0, len(base) - 1, args.max_samples)})
    pipeline = build_pipeline(
        args.degradation, chime_dir=args.chime_dir, demand_dir=args.demand_dir
    ) if args.degradation else None
    dataset = Augment(base, pipeline, seed=args.seed) if pipeline else base

    for algo_name in args.algorithms:
        cond = args.degradation or "clean"
        fname = f"notes_{algo_name}_{args.dataset}_{cond}_seed{args.seed}.json"
        result_path = os.path.join(args.output_dir, fname)
        if os.path.exists(result_path):
            print(f"skip {fname} (exists; delete to redo)")
            continue
        t0 = time.time()
        try:
            algo_cls = get_algorithm(algo_name)
        except Exception as e:
            print(f"FATAL: cannot resolve {algo_name} ({e}) -- recording as crashed")
            algo_cls = None
        metrics, crashed = (
            run_note_evaluation(dataset, algo_cls, thresholds, device=args.device,
                                indices=indices)
            if algo_cls is not None else ({"conp": float("nan"), "conpoff": float("nan")}, True)
        )
        full = {
            "metadata": {
                "track": "notes",
                "algorithm_name": algo_name,
                "dataset_name": args.dataset,
                "condition": cond,
                "seed": args.seed,
                "crashed": crashed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "execution_time_seconds": round(time.time() - t0, 2),
            },
            "parameters": {
                "sample_rate": args.sample_rate,
                "hop_size": args.hop_size,
                "thresholds": [float(t) for t in thresholds],
                "lam_grid": LAM_GRID,
                "onset_tolerance_s": 0.05,
                "pitch_tolerance_cents": 50.0,
                "offset_ratio": 0.2,
            },
            "results": metrics,
        }
        # Temp file + atomic rename: a kill mid-write must not leave a truncated JSON that the
        # skip-if-done resume above then treats as finished.
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(to_json_safe(full), f, indent=4)
        os.replace(tmp_path, result_path)
        print(f"saved {fname}: COnP={metrics.get('conp')} COnPOff={metrics.get('conpoff')}")


if __name__ == "__main__":
    main()
