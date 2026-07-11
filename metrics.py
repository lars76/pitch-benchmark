"""Pure-numpy evaluation core: pitch bands + frame-level metric functions.

Torch-free so it imports (and tests) without torch/torchaudio. Used by the runner
(pitch_benchmark.py) and the report (generate_report.py).
"""
import math

import numpy as np
from scipy.ndimage import find_objects, label

# --------------------------------------------------------------------------- #
# Voicing contract (single source of truth)
# --------------------------------------------------------------------------- #
# `periodicity` is a per-frame voicing CONFIDENCE in [0, 1] (binary {0,1} is the certain case);
# a frame is voiced iff its confidence is at least this threshold. Defined here (the torch-free
# core) and imported by the datasets so "voiced" has ONE definition. Works on numpy or torch.
VOICED_THRESHOLD = 0.5


def is_voiced(periodicity):
    """voiced iff periodicity >= VOICED_THRESHOLD; accepts numpy arrays or torch tensors."""
    return periodicity >= VOICED_THRESHOLD


def cents(a, b):
    """Pitch interval a/b in cents (1200 * log2(a/b)). Single definition shared by the scoring core
    and the offline consensus generator (scripts/build_consensus_labels.py). NaN/inf where a or b is
    <= 0; callers mask voiced frames before comparing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(a / b)


# --------------------------------------------------------------------------- #
# Pitch bands (single source of truth)
# --------------------------------------------------------------------------- #
# (name, lo_hz, hi_hz): half-open [lo, hi). Boundaries match the research eval (INVESTIGATION
# §3/§4): the high register splits at 650/1050 so the >1 kHz octave cliff is visible (hidden if
# everything >520 Hz is one band); <80 Hz is the intrinsic STFT-resolution floor.
PITCH_BANDS = [
    ("bass", 0.0, 80.0),
    ("low", 80.0, 260.0),
    ("mid", 260.0, 650.0),
    ("high", 650.0, 1050.0),
    ("vhigh", 1050.0, math.inf),
]


def band_label(lo, hi):
    """Human-readable Hz range derived from the bounds."""
    if lo <= 0:
        return f"<{hi:g} Hz"
    if hi == math.inf:
        return f">={lo:g} Hz"
    return f"{lo:g}-{hi:g} Hz"


# --------------------------------------------------------------------------- #
# Frame-level metrics
# --------------------------------------------------------------------------- #
def evaluate_pitch_smoothness(
    pitch_pred: np.ndarray, pred_voicing: np.ndarray, true_voicing: np.ndarray
) -> dict[str, float]:
    relative_smoothness, continuity_breaks = np.nan, np.nan
    voiced_idx = np.where(pred_voicing)[0]
    if len(voiced_idx) >= 2:
        consecutive_mask = np.diff(voiced_idx) == 1
        starts_idx = voiced_idx[:-1][consecutive_mask]
        ends_idx = voiced_idx[1:][consecutive_mask]
        if starts_idx.size > 0:
            pitch_starts = pitch_pred[starts_idx]
            pitch_ends = pitch_pred[ends_idx]
            valid_pairs_mask = (pitch_starts > 0) & (pitch_ends > 0)
            if np.any(valid_pairs_mask):
                pitch_starts = pitch_starts[valid_pairs_mask]
                pitch_ends = pitch_ends[valid_pairs_mask]
                rel_changes = np.abs(pitch_ends - pitch_starts) / (pitch_starts + 1e-8)
                mean_chg, std_chg = np.mean(rel_changes), np.std(rel_changes)
                relative_smoothness = std_chg / mean_chg if mean_chg > 1e-09 else 0.0 if std_chg < 1e-08 else np.nan

    labeled_segments, num_segments = label(true_voicing)
    if num_segments > 0:
        gt_segments = find_objects(labeled_segments)
        break_count = 0
        total_relevant_segments = 0
        for seg_slice_tuple in gt_segments:
            seg_slice = seg_slice_tuple[0]
            if seg_slice.stop - seg_slice.start > 1:
                total_relevant_segments += 1
                if not np.all(pred_voicing[seg_slice]):
                    break_count += 1
        if total_relevant_segments > 0:
            continuity_breaks = break_count / total_relevant_segments

    return {
        "relative_smoothness": float(relative_smoothness),
        "continuity_breaks": float(continuity_breaks),
    }


def calculate_combined_score(voicing_metrics: dict, pitch_metrics: dict) -> float:
    """Harmonic mean of six [0,1] components; zero/NaN floored to ~0 so a failed axis sinks it."""
    cents_err = pitch_metrics.get("cents_error")
    octave = pitch_metrics.get("octave_error_rate")
    gross = pitch_metrics.get("gross_error_rate")
    components = [
        pitch_metrics.get("rpa", 0.0),
        np.exp(-(500.0 if cents_err is None else cents_err) / 500.0),
        voicing_metrics.get("recall", 0.0),
        voicing_metrics.get("precision", 0.0),
        np.exp(-(1.0 if octave is None else octave) * 10.0),
        np.exp(-(1.0 if gross is None else gross) * 5.0),
    ]
    floored = [c if (c is not None and not np.isnan(c) and c > 1e-6) else 1e-6 for c in components]
    return len(floored) / sum(1.0 / c for c in floored)


# --------------------------------------------------------------------------- #
# Online (streaming) accumulation
# --------------------------------------------------------------------------- #
# Every frame-level metric above is a count or mean over evaluated frames, hence additive: clips
# fold in one at a time into running sufficient statistics, never holding the whole dataset's
# contours in memory. Results are identical to running evaluate_*() on the fully-concatenated
# arrays (same finite-frame rule, same octave/gross definitions). Memory is O(#bands).
RPA_TOLERANCE_CENTS = 50.0
GROSS_ERROR_CENTS = 200.0


class _PitchBin:
    """Running pitch-accuracy sufficient statistics over a subset of frames (overall or a band)."""

    __slots__ = ("n_gross", "n_octave", "n_rca", "n_rpa", "sum_cents", "sum_sq", "valid")

    def __init__(self):
        self.valid = 0
        self.n_rpa = self.n_rca = self.n_gross = self.n_octave = 0
        self.sum_cents = self.sum_sq = 0.0

    def add(self, pitch_pred: np.ndarray, pitch_true: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        pred, true = pitch_pred[mask], pitch_true[mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            abs_cents = np.abs(1200 * np.log2(pred / true))
        finite = np.isfinite(abs_cents)  # drop degenerate frames (pred/true <= 0 -> non-finite cents)
        abs_cents, pred, true = abs_cents[finite], pred[finite], true[finite]
        if abs_cents.size == 0:
            return
        self.valid += int(abs_cents.size)
        self.sum_cents += float(abs_cents.sum())
        self.sum_sq += float(np.sum((pred - true) ** 2))
        self.n_rpa += int(np.sum(abs_cents < RPA_TOLERANCE_CENTS))
        wrapped = abs_cents % 1200
        chroma = np.minimum(wrapped, 1200 - wrapped)
        self.n_rca += int(np.sum(chroma < RPA_TOLERANCE_CENTS))
        self.n_gross += int(np.sum(abs_cents > GROSS_ERROR_CENTS))
        nearest_octave = np.round(abs_cents / 1200.0)
        self.n_octave += int(
            np.sum((nearest_octave >= 1) & (np.abs(abs_cents - nearest_octave * 1200.0) < 100.0))
        )

    def result(self) -> dict:
        if self.valid == 0:
            return {
                "rmse": np.nan, "cents_error": np.nan, "rpa": np.nan, "rca": np.nan,
                "octave_error_rate": np.nan, "gross_error_rate": np.nan, "valid_frames": 0,
            }
        v = self.valid
        return {
            "rmse": float(np.sqrt(self.sum_sq / v)),
            "cents_error": self.sum_cents / v,
            "rpa": self.n_rpa / v,
            "rca": self.n_rca / v,
            "octave_error_rate": self.n_octave / v,
            "gross_error_rate": self.n_gross / v,
            "valid_frames": v,
        }


class MetricAccumulator:
    """Streaming metrics for ONE voicing threshold; fold clips in with update(), read at the end.

    Equivalent to concatenating every clip and scoring voicing P/R/F1, pitch accuracy, and
    evaluate_pitch_smoothness on the full arrays, but with O(#bands) memory.
    """

    def __init__(self, bands=PITCH_BANDS, tau=0.5):
        self.bands = bands
        self.tau = tau                      # pitch_conf threshold for scoring RPA (when provided)
        self.tp = self.fp = self.fn = 0
        self._overall = _PitchBin()
        self._band = {name: _PitchBin() for name, _, _ in bands}
        self._sm_sum = {"relative_smoothness": 0.0, "continuity_breaks": 0.0}
        self._sm_cnt = {"relative_smoothness": 0, "continuity_breaks": 0}

    def update(self, pred_pitch, pred_voicing, true_pitch, true_voicing, pitch_conf=None) -> None:
        # A frame the tracker calls voiced but assigns no pitch (pred_pitch <= 0) is a VOICING miss,
        # not a pitch error: fold the pitch-present test into the voicing decision so the frame is
        # charged exactly once (as a recall miss) and excluded from RPA -- instead of escaping BOTH
        # axes at low voicing thresholds, where pred_voicing is forced True yet the cents are
        # non-finite and _PitchBin.add drops the frame silently.
        pred_voicing = np.asarray(pred_voicing).astype(bool) & (np.asarray(pred_pitch) > 0)
        # `true_voicing` is a [0,1] confidence; threshold to voiced GT (single def: is_voiced).
        tv = is_voiced(np.asarray(true_voicing))
        self.tp += int(np.sum(pred_voicing & tv))
        self.fp += int(np.sum(pred_voicing & ~tv))
        self.fn += int(np.sum(~pred_voicing & tv))
        # Pitch RPA mask: mutually voiced, AND (when provided) the GT pitch is trustworthy.
        mask = pred_voicing & tv
        if pitch_conf is not None:
            mask = mask & (pitch_conf >= self.tau)
        self._overall.add(pred_pitch, true_pitch, mask)
        for name, lo, hi in self.bands:
            self._band[name].add(
                pred_pitch, true_pitch, mask & (true_pitch >= lo) & (true_pitch < hi)
            )
        sm = evaluate_pitch_smoothness(pred_pitch, pred_voicing, tv)
        for key, val in sm.items():
            if val is not None and np.isfinite(val):
                self._sm_sum[key] += val
                self._sm_cnt[key] += 1

    def voicing_metrics(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    def pitch_metrics(self) -> dict:
        return self._overall.result()

    def per_band(self) -> dict:
        out = {}
        for name, _, _ in self.bands:
            r = self._band[name].result()
            out[name] = {
                "rpa": r["rpa"],
                "octave": r["octave_error_rate"],
                "gross": r["gross_error_rate"],
                "valid_frames": r["valid_frames"],
            }
        return out

    def smoothness_metrics(self) -> dict:
        return {
            k: (self._sm_sum[k] / self._sm_cnt[k] if self._sm_cnt[k] else np.nan)
            for k in ("relative_smoothness", "continuity_breaks")
        }

    def combined_score(self) -> float:
        return calculate_combined_score(self.voicing_metrics(), self.pitch_metrics())


# --------------------------------------------------------------------------- #
# Shared evaluation helpers (used by both pitch_benchmark.py and ood_benchmark.py)
# --------------------------------------------------------------------------- #
DEFAULT_THRESHOLDS = np.linspace(0.0, 1.0, 11)  # voicing operating-point sweep, one source of truth


def to_json_safe(obj):
    """Recursively convert numpy types (and NaN/inf) to JSON-serializable values (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)  # before np.integer/int: a bool must stay a JSON bool, not become 0/1
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())  # recurse so NaN/inf inside arrays -> None, not bare NaN
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)  # last-resort for exotic types (set, Path, datetime, ...) -> string


def summarize_threshold_sweep(accumulators, thresholds):
    """Pick the best-combined-score threshold and assemble the shared metrics dict.

    Returns ``(best_idx, best_metrics)``; ``best_idx`` is -1 (and metrics None) only if no threshold
    scored finite. ``best_metrics`` omits the caller-specific ``coverage`` / ``ood_accuracy`` fields,
    which each runner adds. The ``threshold_sweep`` logs every threshold's scalars.
    """
    threshold_sweep, best_idx, best_score = [], -1, -1.0
    for i, (threshold, acc) in enumerate(zip(thresholds, accumulators)):
        v, p = acc.voicing_metrics(), acc.pitch_metrics()
        score = calculate_combined_score(v, p)
        threshold_sweep.append({
            "threshold": float(threshold), "combined_score": score,
            "rpa": p["rpa"], "rca": p["rca"], "cents_error": p["cents_error"],
            "octave_error_rate": p["octave_error_rate"], "gross_error_rate": p["gross_error_rate"],
            "voicing_precision": v["precision"], "voicing_recall": v["recall"], "voicing_f1": v["f1"],
        })
        if not np.isnan(score) and score > best_score:
            best_idx, best_score = i, score
    if best_idx < 0:
        return -1, None
    best = accumulators[best_idx]
    best_metrics = {
        "voicing_detection": best.voicing_metrics(),
        "pitch_accuracy": best.pitch_metrics(),
        "per_band": best.per_band(),
        "smoothness_metrics": best.smoothness_metrics(),
        "combined_score": best_score,
        "optimal_threshold": float(thresholds[best_idx]),
        "threshold_sweep": threshold_sweep,
    }
    return best_idx, best_metrics
