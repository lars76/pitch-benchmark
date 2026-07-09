#!/usr/bin/env python3
"""Synthetic out-of-distribution (OOD) generalization test for pitch trackers.

Generates synthetic signal families with EXACT, non-circular labels (no pitch detector in the label
loop) and measures per-family F0 accuracy, so we can say "signal type X breaks tracker Y". Complements
pitch_benchmark.py (accuracy on real data + label-preserving degradations) and speed_benchmark.py
(timing); generate_report.py merges all three. Mirrors speed_benchmark.py's structure.

Each (tracker, family) cell runs in its OWN subprocess: some C-extension trackers (pyreaper, pysptk)
ABORT (SIGSEGV/SIGABRT) on degenerate inputs such as a pure sine or an extreme spectral tilt, and
per-cell isolation keeps one abort from killing the whole run (it is recorded as a crashed cell).

Voiced families are scored by RPA (reusing the 11-threshold sweep + metrics.MetricAccumulator from the
main runner); unvoiced controls, where the correct answer is "no pitch", are scored by false-positive
rate. Ground truth is exact by construction, which is why the first cut is synthetic-only (see the
plan/investigation: OOD is exactly where detector-derived labels are least trustworthy).
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import (
    build_algorithm,
    get_algorithm,
    list_algorithms,
    resolve_requested_algorithms,
)
from datasets.augment import item_rng
from metrics import (
    DEFAULT_THRESHOLDS as THRESHOLDS,
)
from metrics import (
    MetricAccumulator,
    summarize_threshold_sweep,
    to_json_safe,
)

AVAILABLE_ALGORITHMS = list_algorithms()

SR, HOP = 16000, 256
FMIN, FMAX = 50.0, 2000.0                     # wide common range so the pitch-range axis is not clamped
NYQ = SR / 2.0
N = 2 * SR                                   # 2 s per clip
CELL_TIMEOUT = 300                           # seconds per (tracker, family) subprocess

MECH_F0 = [90.0, 160.0, 250.0, 370.0]        # low-mid grid for the spectral-mechanism families
BANDS_F0 = {                                 # pitch-range axis (2 f0s/band; reaches ~2 kHz)
    "bass": [60.0, 72.0], "low": [110.0, 210.0], "mid": [340.0, 520.0],
    "high": [780.0, 950.0], "vhigh": [1400.0, 1900.0],
}

# name -> (kind, f0_list). `kind` selects the generator; `f0_list` the pitches pooled into that cell.
# Two reported axes: spectral MECHANISMS (isolated at low-mid) and a pitch-RANGE sweep
# (sine/harmonic/tilt per band). `harm_low` doubles as the normal-tone positive control.
CONTROL_KINDS = ("noise", "whisper")
FAMILIES = {
    "missing_f0":   ("missing_f0", MECH_F0),
    "unresolved":   ("unresolved", MECH_F0),
    "irn":          ("irn", MECH_F0),
    "vibrato_fast": ("vibrato", None),
    "sine_bass":    ("sine", BANDS_F0["bass"]),
    "sine_low":     ("sine", BANDS_F0["low"]),
    "sine_mid":     ("sine", BANDS_F0["mid"]),
    "sine_high":    ("sine", BANDS_F0["high"]),
    "sine_vhigh":   ("sine", BANDS_F0["vhigh"]),
    "harm_low":     ("harmonic", BANDS_F0["low"]),
    "harm_mid":     ("harmonic", BANDS_F0["mid"]),
    "harm_high":    ("harmonic", BANDS_F0["high"]),
    "harm_vhigh":   ("harmonic", BANDS_F0["vhigh"]),
    "tilt_low":     ("tilt", BANDS_F0["low"]),
    "tilt_mid":     ("tilt", BANDS_F0["mid"]),
    "tilt_high":    ("tilt", BANDS_F0["high"]),
    "tilt_vhigh":   ("tilt", BANDS_F0["vhigh"]),
    "noise":        ("noise", None),
    "whisper":      ("whisper", None),
}
VOICED_FAMILIES = [n for n, (k, _) in FAMILIES.items() if k not in CONTROL_KINDS]
CONTROL_FAMILIES = [n for n, (k, _) in FAMILIES.items() if k in CONTROL_KINDS]
ALL_FAMILIES = list(FAMILIES)


# --------------------------------------------------------------------------- #
# Synthetic signal generators (exact labels; no pitch detector in the loop)
# --------------------------------------------------------------------------- #
RAMP_MS = 20.0        # raised-cosine on/off gate; avoids onset spectral splatter / clicks at the edges
TARGET_RMS = 0.1      # RMS-normalize so families are equal-loudness (removes a cross-family confound)


def _ramp(x):
    """Apply a raised-cosine on/off ramp so hard edges do not inject a click / spectral splatter."""
    r = int(RAMP_MS / 1000.0 * SR)
    if 2 * r >= len(x):
        return x.astype(float)
    w = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
    y = x.astype(float).copy()
    y[:r] *= w
    y[-r:] *= w[::-1]
    return y


def _finalize(x):
    """Standard stimulus conditioning: raised-cosine ramp, then RMS-normalize with a peak guard."""
    x = _ramp(x)
    x = x * (TARGET_RMS / (np.sqrt(np.mean(x ** 2)) + 1e-9))
    peak = np.max(np.abs(x))
    if peak > 0.95:
        x = x * (0.95 / peak)                 # guard against clipping for high-crest-factor signals
    return x.astype(np.float32)


def _synth(parts, n):
    """parts: list of (per-sample-freq array, amplitude). Sum of sines, ramped + RMS-normalized."""
    x = np.zeros(n)
    for fr, a in parts:
        x += a * np.sin(2 * np.pi * np.cumsum(fr) / SR)
    return _finalize(x)


def _harm_parts(f0_track, amp_fn, missing_f0=False, inharm_B=0.0):
    """Harmonic (or stretched-inharmonic) partials below Nyquist; drop zero-amplitude harmonics."""
    parts, k = [], 1
    while k <= 60:
        fk = k * f0_track * (np.sqrt(1 + inharm_B * k * k) if inharm_B else 1.0)
        if np.max(fk) >= NYQ - 100:
            break
        if not (missing_f0 and k == 1):
            a = float(amp_fn(k))
            if a != 0.0:
                parts.append((fk, a))
        k += 1
    return parts


def _tilt_amp(f0, db):
    """Harmonic-amplitude function for a spectral slope of `db` dB/octave referenced to 1 kHz."""
    return lambda k: 10 ** (db * np.log2(k * f0 / 1000.0) / 20.0)


def _frame_f0(ft, n):
    """Per-sample f0 track -> per-frame f0 at the eval-grid frame centers. Frame m is centered at
    sample m*HOP (the shared contract predictions are placed on by resample_to_grid), NOT m*HOP+HOP/2;
    the half-hop offset made every label 8 ms late, taxing the time-varying vibrato family ~19 cents."""
    nf = n // HOP
    return ft[(np.arange(nf) * HOP).clip(0, n - 1)].astype(np.float32)


def _voiced_clip(ft, parts):
    return _synth(parts, N), _frame_f0(ft, N), np.ones(N // HOP, dtype=np.float32)


def _irn(f0, rng, n_iter=8, gain=1.0):
    """Iterated rippled noise: repeated delay-and-add of broadband noise; pitch at 1/delay."""
    d = round(SR / f0)
    x = rng.standard_normal(N)
    for _ in range(n_iter):
        x = x + gain * np.concatenate([np.zeros(d), x[:-d]])
    return _finalize(x), SR / d               # actual f0 for the integer-rounded delay


def _control_signal(family, rng):
    """Broadband noise (`noise`) or formant-shaped noise (`whisper`): no periodicity, no f0."""
    x = rng.standard_normal(N)
    if family == "whisper":
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(N, 1.0 / SR)
        env = np.zeros_like(f)
        for fc, bw in ((500.0, 120.0), (1500.0, 180.0), (2500.0, 250.0)):
            env += np.exp(-0.5 * ((f - fc) / bw) ** 2)
        x = np.fft.irfft(spec * env, n=N)
    return _finalize(x)


def _family_id(family):
    """Stable integer id for a family (alphabetical rank), for per-clip seeding. Sorting decouples
    the id from FAMILIES' insertion order, so reordering the table cannot re-roll the clips."""
    return sorted(FAMILIES).index(family)


def make_clips(family):
    """Return a list of (audio, f0_per_frame, voiced_per_frame) clips with exact labels.

    Stochastic clips (noise/whisper/IRN) use the shared per-item seeding idiom
    (datasets.augment.item_rng, a pure function of the ids), replacing the earlier ad-hoc
    `default_rng(100+i)`/`200+i` scheme -- deterministic like before, but collision-free by
    construction and consistent with the rest of the codebase. The clips this generates differ
    from the pre-A5-fix era either way (the label-grid fix already invalidated old OOD results)."""
    kind, f0_list = FAMILIES[family]

    if kind in CONTROL_KINDS:
        zeros = np.zeros(N // HOP, dtype=np.float32)
        return [(_control_signal(kind, item_rng(0, _family_id(family), i)), zeros, zeros)
                for i in range(2)]

    if kind == "vibrato":
        t = np.arange(N) / SR
        ft = 220.0 * 2 ** (1.0 / 12 * np.sin(2 * np.pi * 6 * t))     # +-1 semitone at 6 Hz
        return [_voiced_clip(ft, _harm_parts(ft, lambda k: 1.0 / k))]

    clips = []
    for i, f0 in enumerate(f0_list):
        if kind == "irn":
            x, f0_true = _irn(f0, item_rng(0, _family_id(family), i))
            ft = np.full(N, f0_true)
            clips.append((x, _frame_f0(ft, N), np.ones(N // HOP, dtype=np.float32)))
            continue
        amp = {
            "sine": lambda k: 1.0 if k == 1 else 0.0,
            "harmonic": lambda k: 1.0 / k,
            "tilt": _tilt_amp(f0, 6.0),
            "missing_f0": lambda k: 1.0 / k,
            "unresolved": lambda k: 1.0 if 10 <= k <= 17 else 0.0,
        }[kind]
        ft = np.full(N, float(f0))
        parts = _harm_parts(ft, amp, missing_f0=(kind == "missing_f0"))
        clips.append(_voiced_clip(ft, parts))
    return clips


# --------------------------------------------------------------------------- #
# Scoring (reuses metrics.MetricAccumulator + the 11-threshold sweep)
# --------------------------------------------------------------------------- #
def _score_voiced(algo, clips):
    """Sweep 11 voicing thresholds, pick the best combined score; emit pitch_benchmark-shaped metrics."""
    accs = [MetricAccumulator() for _ in THRESHOLDS]
    for x, f0f, vf in clips:
        results = algo.extract_pitch(x, thresholds=list(THRESHOLDS), compute_notes=False)
        for acc, (pp, pv, _) in zip(accs, results):
            pp = np.asarray(pp, dtype=float)
            pv = np.asarray(pv, dtype=bool)
            L = min(len(pp), len(f0f))                # trackers differ by a frame; align by min length
            acc.update(pp[:L], pv[:L], f0f[:L], vf[:L])

    best_idx, best_metrics = summarize_threshold_sweep(accs, THRESHOLDS)
    if best_idx < 0:
        return {"ood_accuracy": float("nan")}
    # Coverage-aware accuracy: correct frames / ALL ground-truth-voiced frames (tp + fn). Unlike the
    # conditional pitch_accuracy.rpa (only over frames the tracker chose to voice), this DROPS when a
    # tracker copes with a hard signal by refusing to voice it -- the OOD failure we care about. GT is
    # all-voiced here, so it is "fraction of frames voiced AND within 50 cents of f0".
    best = accs[best_idx]
    p = best.pitch_metrics()
    gt_voiced = best.tp + best.fn
    rpa = p["rpa"]
    n_correct = 0.0 if (rpa is None or np.isnan(rpa)) else rpa * p["valid_frames"]
    best_metrics["ood_accuracy"] = float(n_correct / gt_voiced) if gt_voiced > 0 else float("nan")
    return best_metrics


def _score_control(algo, clips):
    """False-positive rate: fraction of frames the tracker voices on a signal that has no pitch."""
    fps = []
    for x, _, _ in clips:
        _, v, _ = algo.extract_pitch(x, compute_notes=False)[0]     # default operating threshold
        v = np.asarray(v, dtype=bool)
        fps.append(float(np.mean(v)) if len(v) else float("nan"))
    return {"false_positive_rate": float(np.nanmean(fps)) if fps else float("nan")}


# --------------------------------------------------------------------------- #
# JSON cells (resumable, one per (tracker, family); mirrors pitch_benchmark.py)
# --------------------------------------------------------------------------- #
def _cell_path(output_dir, algo, family, device):
    # Device is in the filename so cpu/gpu cells coexist rather than overwrite (mirrors pitch_benchmark).
    return os.path.join(output_dir, f"ood_{algo}_{family}_{device}_sr{SR // 1000}k_hop{HOP}.json")


def _write_cell(output_dir, algo, family, ftype, device, results, crashed=False, error=None):
    meta = {
        "benchmark_type": "ood",
        "algorithm_name": algo,
        "family": family,
        "family_type": ftype,
        "device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if crashed:
        meta["crashed"] = True
        meta["error"] = error
    obj = {
        "metadata": meta,
        "parameters": {
            "sample_rate": SR, "hop_size": HOP, "fmin": FMIN, "fmax": FMAX,
            "n_seconds": N / SR, "f0s": FAMILIES.get(family, (None, None))[1],
        },
        "results": results,
    }
    with open(_cell_path(output_dir, algo, family, device), "w") as f:
        json.dump(to_json_safe(obj), f, indent=2)


# --------------------------------------------------------------------------- #
# Worker (one tracker x one family, isolated) and parent (spawns + records)
# --------------------------------------------------------------------------- #
def run_worker(algo, family, output_dir, device):
    ftype = "control" if family in CONTROL_FAMILIES else "voiced"
    cls = get_algorithm(algo)
    algo_obj = build_algorithm(cls, SR, HOP, FMIN, FMAX, device=device)
    eff_device = cls.resolve_effective_device(device)
    clips = make_clips(family)
    results = _score_control(algo_obj, clips) if ftype == "control" else _score_voiced(algo_obj, clips)
    _write_cell(output_dir, algo, family, ftype, eff_device, results)


def run_parent(algos, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    # Effective device is part of the filename, so cpu/gpu cells coexist. fail_silently: an uninstalled
    # backend is left to the worker (which fails and gets recorded as a crashed cell), keeping per-cell
    # isolation instead of crashing the whole parent. Resolve once per algo, reuse across families.
    eff = {}
    for algo in algos:
        cls = get_algorithm(algo, fail_silently=True)
        eff[algo] = cls.resolve_effective_device(device) if cls is not None else device

    cells = [(a, f) for a in algos for f in ALL_FAMILIES]
    counts = {"done": 0, "skip": 0, "crash": 0}
    # One bar over every (tracker, family) cell; ETA/rate come free. Per-cell crashes/timeouts go to
    # bar.write so they log above the bar; the running done/skip/crash tally rides in the postfix.
    bar = tqdm(cells, desc="OOD cells", unit="cell")
    for algo, family in bar:
        effective = eff[algo]
        path = _cell_path(output_dir, algo, family, effective)
        # Cache-as-done: any existing cell is skipped, a recorded crash included (delete the file to redo).
        if os.path.exists(path):
            counts["skip"] += 1
            bar.set_postfix(**counts)
            continue
        ftype = "control" if family in CONTROL_FAMILIES else "voiced"
        cmd = [
            sys.executable, os.path.abspath(__file__), "--_worker",
            "--algo", algo, "--family", family, "--output-dir", output_dir, "--device", device,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=CELL_TIMEOUT)
        except subprocess.TimeoutExpired:
            _write_cell(output_dir, algo, family, ftype, effective, {}, crashed=True,
                        error=f"timeout > {CELL_TIMEOUT}s")
            counts["crash"] += 1
            bar.write(f"TIMEOUT {algo}/{family}")
        else:
            if not os.path.exists(path):        # worker died (e.g. C-extension abort) before writing
                tail = (out.stderr.strip().splitlines() or ["(no stderr)"])[-3:]
                _write_cell(output_dir, algo, family, ftype, effective, {}, crashed=True,
                            error=f"exit {out.returncode}: " + " | ".join(tail))
                counts["crash"] += 1
                bar.write(f"CRASH {algo}/{family} (exit {out.returncode})")
            else:
                counts["done"] += 1
        bar.set_postfix(**counts)


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic OOD generalization test for pitch trackers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algorithms", type=str, nargs="+", default=None, choices=AVAILABLE_ALGORITHMS,
                        help="Algorithms to test (default: every installed algorithm).")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for result JSONs.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for device-aware trackers (auto, matching pitch_benchmark).")
    # Hidden worker plumbing (parent re-invokes itself per cell for crash isolation).
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algo", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--family", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        run_worker(args.algo, args.family, args.output_dir, args.device)
        return

    algos = resolve_requested_algorithms(args.algorithms, on_empty=parser.error)
    print(f"OOD probe: {len(algos)} algorithms x {len(ALL_FAMILIES)} families -> {args.output_dir}")
    run_parent(algos, args.output_dir, args.device)


if __name__ == "__main__":
    main()
