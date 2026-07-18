#!/usr/bin/env python3
"""Synthetic out-of-distribution (OOD) generalization test for pitch trackers.

Generates synthetic signal families with EXACT, non-circular labels (no pitch detector in the label
loop) and measures per-family F0 accuracy, so we can say "signal type X breaks tracker Y". Complements
pitch_benchmark.py (accuracy on real data + label-preserving degradations) and speed_benchmark.py
(timing); generate_report.py merges all three.

A pure per-cell measurement library; evaluate.py orchestrates. NOTE: some C-extension trackers
ABORT (SIGSEGV/SIGABRT) on these degenerate stimuli (a pure sine, an extreme
spectral tilt); that is why evaluate runs every ood cell in a child process by default.

Voiced families are scored by RPA (reusing the 11-threshold sweep + metrics.MetricAccumulator from the
main runner); unvoiced controls, where the correct answer is "no pitch", are scored by false-positive
rate. Ground truth is exact by construction, which is why the first cut is synthetic-only: OOD is
exactly where detector-derived labels are least trustworthy.
"""
import numpy as np

import gc

from algorithms import build_algorithm
from datasets.augment import item_rng
from metrics import (
    DEFAULT_THRESHOLDS as THRESHOLDS,
)
from metrics import (
    MetricAccumulator,
    summarize_threshold_sweep,
)

SR, HOP = 16000, 256
FMIN, FMAX = 50.0, 2000.0                     # wide common range so the pitch-range axis is not clamped
NYQ = SR / 2.0
N = 2 * SR                                   # 2 s per clip

MECH_F0 = [90.0, 160.0, 250.0, 370.0]        # low-mid grid for the spectral-mechanism families
BANDS_F0 = {                                 # pitch-range axis (2 f0s/band; reaches ~2 kHz)
    "bass": [60.0, 72.0], "low": [110.0, 210.0], "mid": [340.0, 520.0],
    "high": [780.0, 950.0], "vhigh": [1400.0, 1900.0],
}

# name -> (kind, f0_list). `kind` selects the generator; `f0_list` the pitches pooled into that cell.
# Two reported axes: spectral MECHANISMS (isolated at low-mid) and a pitch-RANGE sweep
# (sine/harmonic/tilt per band). `harm_low` doubles as the normal-tone positive control.
LEVELS_DB = (-6.0, -16.0, -26.0, -36.0, -46.0, -56.0, -66.0)  # absolute-level sweep, 10 dB steps
LEVEL_F0 = 160.0     # a MECH_F0 member: every tracker is competent here at nominal level, so any
                     # drop under attenuation isolates LEVEL, not pitch range
INTERF_DB = (-20.0, -10.0, -5.0)   # interfering-source level RELATIVE to the dominant source
INTERF_F0 = 50.0                   # the interfering source's f0 (a 50 Hz tone + 2nd harmonic)
CONTROL_KINDS = ("noise", "whisper", "silence")
FAMILIES = {
    "missing_f0":   ("missing_f0", MECH_F0),
    "unresolved":   ("unresolved", MECH_F0),
    "irn":          ("irn", MECH_F0),
    "vibrato_fast": ("vibrato", None),
    "glide":        ("glide", None),
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
    "harm_bass":    ("harmonic", BANDS_F0["bass"]),   # bass complexes (sine_bass is a pure sine)
    "sine_level":   ("level_sine", None),             # absolute-level sweep (pure tone)
    "harm_level":   ("level_harm", None),             # absolute-level sweep (harmonic tone)
    "interference": ("interference", None),           # two periodic sources; follow the dominant
    "silence":      ("silence", None),                # near-silence FP control
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


def _finalize_at(x, rms):
    """Stimulus conditioning at an explicit rms, for the level families. _finalize normalizes
    every stimulus to TARGET_RMS, which would erase the level axis. No peak guard: every swept
    level is at or below TARGET_RMS."""
    x = _ramp(x)
    x = x * (rms / (np.sqrt(np.mean(x ** 2)) + 1e-9))
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
    """Broadband noise (`noise`), formant-shaped noise (`whisper`), or near-silence
    (`silence`): no periodicity, no f0."""
    if family == "silence":
        # NOT _finalize'd: RMS-normalizing silence up to TARGET_RMS would defeat the probe.
        return (1e-6 * rng.standard_normal(N)).astype(np.float32)
    x = rng.standard_normal(N)
    if family == "whisper":
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(N, 1.0 / SR)
        env = np.zeros_like(f)
        for fc, bw in ((500.0, 120.0), (1500.0, 180.0), (2500.0, 250.0)):
            env += np.exp(-0.5 * ((f - fc) / bw) ** 2)
        x = np.fft.irfft(spec * env, n=N)
    return _finalize(x)


# FROZEN per-family ids for stochastic-clip seeding (noise/whisper/irn). The values are
# arbitrary; what matters is that they NEVER change: a sorted-rank lookup would silently
# renumber families whenever one is added, re-rolling every stochastic clip and invalidating
# existing result comparisons. Give a new family the next unused id, at the end; never
# renumber. test_ood.py locks the table against FAMILIES drift.
_FAMILY_IDS = {
    "harm_high": 0, "harm_low": 1, "harm_mid": 2, "harm_vhigh": 3, "irn": 4, "missing_f0": 5,
    "noise": 6, "sine_bass": 7, "sine_high": 8, "sine_low": 9, "sine_mid": 10, "sine_vhigh": 11,
    "tilt_high": 12, "tilt_low": 13, "tilt_mid": 14, "tilt_vhigh": 15, "unresolved": 16,
    "vibrato_fast": 17, "whisper": 18,
    "glide": 19,
    "harm_bass": 20, "sine_level": 21, "harm_level": 22, "interference": 23, "silence": 24,
}


def family_type(name):
    """The one routing rule: control (FP-rate) or voiced (coverage-aware RPA)."""
    return "control" if FAMILIES[name][0] in CONTROL_KINDS else "voiced"


def _family_id(family):
    """Stable integer id for a family, for per-clip seeding (see _FAMILY_IDS)."""
    return _FAMILY_IDS[family]


def make_clips(family):
    """Return a list of (audio, f0_per_frame, voiced_per_frame) clips with exact labels.

    Stochastic clips (noise/whisper/IRN) are seeded by item_rng over the frozen family ids."""
    kind, f0_list = FAMILIES[family]

    if kind in CONTROL_KINDS:
        zeros = np.zeros(N // HOP, dtype=np.float32)
        return [(_control_signal(kind, item_rng(0, _family_id(family), i)), zeros, zeros)
                for i in range(2)]

    if kind == "vibrato":
        t = np.arange(N) / SR
        ft = 220.0 * 2 ** (1.0 / 12 * np.sin(2 * np.pi * 6 * t))     # +-1 semitone at 6 Hz
        return [_voiced_clip(ft, _harm_parts(ft, lambda k: 1.0 / k))]

    if kind == "glide":
        # Monotonic one-octave glides (600 cents/s, the calibration-probe slope, but scored as
        # RPA against exact per-frame labels): up + down, low + mid register. Deterministic.
        t = np.arange(N) / SR
        clips = []
        for f_start, octaves in ((110.0, +1.0), (220.0, -1.0), (330.0, +1.0), (660.0, -1.0)):
            ft = f_start * 2.0 ** (octaves * t * SR / N)
            clips.append(_voiced_clip(ft, _harm_parts(ft, lambda k: 1.0 / k)))
        return clips

    if kind in ("level_sine", "level_harm"):
        amp = (lambda k: 1.0 if k == 1 else 0.0) if kind == "level_sine" else (lambda k: 1.0 / k)
        ft = np.full(N, LEVEL_F0)
        parts = _harm_parts(ft, amp)
        x = np.zeros(N)
        for fr, a in parts:
            x += a * np.sin(2 * np.pi * np.cumsum(fr) / SR)
        return [(_finalize_at(x, TARGET_RMS * 10 ** (db / 20.0)),
                 _frame_f0(ft, N), np.ones(N // HOP, dtype=np.float32))
                for db in LEVELS_DB]

    if kind == "interference":
        # Two simultaneous periodic sources: a dominant one (220 Hz with vibrato, so following
        # is tested rather than coincidence) and a quieter one (a 50 Hz tone, swept relative
        # level). GT is the dominant source's contour; a tracker that switches to the quieter
        # source scores 0 on those frames. The quieter source stays below equal level on
        # purpose: "the f0" of equal-level polyphony is undefined for a monophonic tracker.
        t = np.arange(N) / SR
        ft = 220.0 * 2 ** (1.0 / 12 * np.sin(2 * np.pi * 3 * t))
        target = np.zeros(N)
        for fr, a in _harm_parts(ft, lambda k: 1.0 / k):
            target += a * np.sin(2 * np.pi * np.cumsum(fr) / SR)
        target /= np.sqrt(np.mean(target ** 2)) + 1e-9
        interferer = (np.sin(2 * np.pi * INTERF_F0 * t)
                      + 0.5 * np.sin(2 * np.pi * 2 * INTERF_F0 * t))
        interferer /= np.sqrt(np.mean(interferer ** 2)) + 1e-9
        return [(_finalize(target + 10 ** (db / 20.0) * interferer),
                 _frame_f0(ft, N), np.ones(N // HOP, dtype=np.float32))
                for db in INTERF_DB]

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
    # tracker copes with a hard signal by refusing to voice it, the OOD failure we care about. GT is
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
# The per-cell measurement (pure: no writing, no processes; evaluate.py owns those)
# --------------------------------------------------------------------------- #
def run_ood_cell(algorithm_class, family, device="auto"):
    """Score one (tracker, family) cell. Same shape as the other tracks: takes the algorithm
    CLASS, returns the results dict, touches nothing else."""
    algo_obj = build_algorithm(algorithm_class, SR, HOP, FMIN, FMAX, device=device)
    clips = make_clips(family)
    if family_type(family) == "control":
        results = _score_control(algo_obj, clips)
    else:
        results = _score_voiced(algo_obj, clips)
    del algo_obj
    gc.collect()
    return results
