"""Pure-numpy evaluation core: pitch bands + frame-level metric functions.

Torch-free so it imports (and tests) without torch/torchaudio. Used by the runner
(pitch_benchmark.py) and the report (generate_report.py).
"""
import math
import os

import numpy as np
from scipy.ndimage import find_objects, label


def clip_and_group(dataset, wav_path, idx):
    """(clip_id, group) for a per-clip row: the ONE place both runners (pitch_benchmark.py,
    note_benchmark.py) derive them, so their cluster-bootstrap CIs share the same grouping.

    group = the dataset's leakage-safe get_group (speaker / singer / piece), forwarded through the
    Augment/Truncate/Subset wrappers; NOT basename(dirname(wav)) (a logging path that collapses many
    speakers into one bogus cluster). Falls back to dirname/idx only if the dataset has no get_group.
    """
    clip_id = os.path.basename(str(wav_path)) if wav_path is not None else str(idx)
    try:
        group = str(dataset.get_group(idx))
    except (AttributeError, IndexError, KeyError):
        group = os.path.basename(os.path.dirname(str(wav_path))) if wav_path is not None else str(idx)
    return clip_id, group

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
# (name, lo_hz, hi_hz): half-open [lo, hi). The high register splits at 650/1050 so the >1 kHz octave
# cliff is visible (it is hidden if everything >520 Hz is one band); <80 Hz is the intrinsic
# STFT-resolution floor.
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
        # charged exactly once (as a recall miss) and excluded from RPA, instead of escaping BOTH
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

    def pitch_coverage(self) -> float:
        """Fraction of ground-truth-voiced frames that entered pitch scoring (valid_frames /
        (TP+FN)). Pitch metrics are computed only where BOTH sides are voiced, so a
        conservative tracker scores RPA on an easier, self-selected subset; coverage is the
        column that makes RPA comparable across algorithms (low coverage = survivor-biased RPA)."""
        gt_voiced = self.tp + self.fn
        return self._overall.result()["valid_frames"] / gt_voiced if gt_voiced > 0 else 0.0

    def suff_stats(self) -> dict:
        """The additive sufficient statistics behind every scalar metric (overall pitch bin + voicing
        counts). Summing these across clips reproduces the full-dataset accumulator EXACTLY, so a
        cluster bootstrap can resample clips, sum, and RECOMPUTE any aggregate (RPA/coverage/F1/cents/
        combined) frame-weighted: the honest way to CI a nonlinear aggregate like combined_score."""
        o = self._overall
        return {"valid": int(o.valid), "n_rpa": int(o.n_rpa), "sum_cents": float(o.sum_cents),
                "n_octave": int(o.n_octave), "n_gross": int(o.n_gross),
                "tp": int(self.tp), "fp": int(self.fp), "fn": int(self.fn)}

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


def voicing_boundary_latency(pred_voiced, true_voiced, frame_period, min_frames=8):
    """Onset/offset truncation of ground-truth voiced regions, in ms.

    For each GT voiced region of >= min_frames, onset latency = time from the region's first
    frame to the tracker's first voiced frame inside it (region length if never voiced);
    offset latency symmetric. Aggregate voicing F1 is position-blind: a tracker that
    systematically eats the first 30 ms of every region (e.g. the low start of a rising tone)
    can share an F1 with one that scatters its misses; this is the metric that separates them.
    Returns (onset_ms_list, offset_ms_list), one entry per qualifying region."""
    pred_voiced = np.asarray(pred_voiced).astype(bool)
    true_voiced = np.asarray(true_voiced).astype(bool)
    on, off = [], []
    i, n = 0, min(len(pred_voiced), len(true_voiced))
    while i < n:
        if true_voiced[i]:
            j = i
            while j < n and true_voiced[j]:
                j += 1
            if j - i >= min_frames:
                hit = np.where(pred_voiced[i:j])[0]
                if len(hit) == 0:
                    on.append((j - i) * frame_period * 1000.0)
                    off.append((j - i) * frame_period * 1000.0)
                else:
                    on.append(hit[0] * frame_period * 1000.0)
                    off.append((j - i - 1 - hit[-1]) * frame_period * 1000.0)
            i = j
        else:
            i += 1
    return on, off


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
        "pitch_accuracy": {**best.pitch_metrics(), "coverage": best.pitch_coverage()},
        "per_band": best.per_band(),
        "smoothness_metrics": best.smoothness_metrics(),
        "combined_score": best_score,
        "optimal_threshold": float(thresholds[best_idx]),
        "threshold_sweep": threshold_sweep,
    }
    return best_idx, best_metrics

# --------------------------------------------------------------------------- #
# Cluster-bootstrap statistics over per-clip sufficient stats (the ONE bootstrap, shared by the
# report tables and evaluate.compare). Always cluster by the per-clip `group` (speaker/singer/
# piece via get_group), never by clip; correlated clips would give false precision. We pre-SUM
# each cluster's columns once (metrics are additive), resample cluster INDICES vectorized, and a
# reducer turns the summed picks into the frame-weighted aggregate (RPA/coverage/F1/cents/
# combined) or a clip-mean (note COnP). Because every reducer consumes only column SUMS,
# evaluating it on one pre-summed dict is exactly equivalent to handing it the picked dicts --
# the scalar reducers stay the single, verified formula implementation.
# --------------------------------------------------------------------------- #
def keyed_group_sums(rows, cols, group_col=1):
    """Per cluster {name: Σcolumn, ..., '_n': #clips} KEYED by the cluster id (so two algorithms
    scored on the same clips can be paired cluster-by-cluster). `cols` = [(name, col_index)].
    Deliberately NO clip-level fallback for a single-group result: between-source variance is
    inestimable from one source (clips of one source are correlated, so resampling them would
    understate the real uncertainty). Callers render `[n/a]` and make no tie claims. Every
    registered dataset exposes real source clusters via get_group, so this never triggers in
    practice; it exists for degenerate inputs (e.g. a custom dataset that is one source)."""
    groups = {}
    for r in rows:
        groups.setdefault(r[group_col], []).append(r)
    out = {}
    for g, rs in groups.items():
        d = {"_n": len(rs)}
        for name, idx in cols:
            d[name] = float(sum(float(r[idx]) for r in rs))
        out[g] = d
    return out


def boot_vals(per_group, reduce_fn, n_boot=2000, seed=0, per_group_b=None):
    """The n_boot bootstrap draws of reduce_fn over resampled clusters. If `per_group_b` is given
    (PAIRED: the same clusters scored by a second algorithm/condition, same order), each draw picks
    ONE set of cluster indices, applies it to BOTH sides, and returns reduce(A) - reduce(B) --
    shared per-cluster difficulty cancels, which is the honest A-vs-B comparison on shared clips."""
    n = len(per_group)
    if n == 0:
        return np.full(n_boot, np.nan)
    names = sorted(per_group[0])
    arr = np.array([[d[k] for k in names] for d in per_group], dtype=float)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    sums = arr[idx].sum(axis=1)                     # (n_boot, n_cols): the vectorized heavy part
    if per_group_b is None:
        return np.array([reduce_fn([dict(zip(names, row))]) for row in sums])
    arr_b = np.array([[d[k] for k in names] for d in per_group_b], dtype=float)
    sums_b = arr_b[idx].sum(axis=1)                 # SAME picks on both sides = paired
    return np.array([reduce_fn([dict(zip(names, ra))]) - reduce_fn([dict(zip(names, rb))])
                     for ra, rb in zip(sums, sums_b)])


def cluster_bootstrap(per_group, reduce_fn, n_boot=2000, seed=0, per_group_b=None):
    """95% percentile CI of reduce_fn over the clusters (or, with per_group_b, of the PAIRED
    difference reduce(A)-reduce(B); see boot_vals). Returns (nan, nan) with < 2 clusters:
    between-cluster variance cannot be estimated from one cluster, so no interval is produced
    (rendered as `[n/a]`)."""
    if len(per_group) < 2:
        return float("nan"), float("nan")
    vals = boot_vals(per_group, reduce_fn, n_boot, seed, per_group_b)
    if not np.isfinite(vals).all():
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_tied(keyed_a, keyed_b, reduce_fn, n_boot=2000, seed=0):
    """True if the paired 95% CI of reduce(A)-reduce(B) over the COMMON clusters includes 0 --
    i.e. A and B are statistically tied on the clips both scored. A tie is a POSITIVE statistical
    claim: with < 2 common clusters it cannot be assessed, so no tie is claimed (False)."""
    common = sorted(set(keyed_a) & set(keyed_b))
    if len(common) < 2:
        return False
    lo, hi = cluster_bootstrap([keyed_a[g] for g in common], reduce_fn, n_boot, seed,
                               per_group_b=[keyed_b[g] for g in common])
    return bool(lo <= 0.0 <= hi)


def paired_delta_ci(pairs, n_boot=2000):
    """95% CI of the mean-over-datasets PAIRED combined-score delta (clean - degraded), in
    percentage points. `pairs` = per dataset (keyed_clean, keyed_degraded) cluster sums over the
    SAME probe clips; each bootstrap draw resamples clusters within each dataset (stratified) and
    applies the same picks to both sides, so shared clip difficulty cancels. Datasets with < 2
    common clusters cannot be resampled and are skipped. Returns (lo, hi, n_datasets_used)."""
    per_ds = []
    for k, (kc, kd) in enumerate(pairs):
        common = sorted(set(kc) & set(kd))
        if len(common) < 2:
            continue
        per_ds.append(boot_vals([kc[g] for g in common], FRAME_REDUCERS["combined"],
                                n_boot, seed=k, per_group_b=[kd[g] for g in common]))
    if not per_ds:
        return float("nan"), float("nan"), 0
    vals = np.mean(per_ds, axis=0) * 100.0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(per_ds)


def sum_col(dicts, k):
    return sum(d[k] for d in dicts)


# Frame reducers: recompute the FRAME-weighted aggregate from summed sufficient stats (so CIs match
# the reported pitch_accuracy.* / combined_score exactly).
def agg_from_sums(dicts):
    v = sum_col(dicts, "valid")
    tp, fp, fn = sum_col(dicts, "tp"), sum_col(dicts, "fp"), sum_col(dicts, "fn")
    pm = {"rpa": sum_col(dicts, "n_rpa") / v if v else 0.0,
          "cents_error": sum_col(dicts, "sum_cents") / v if v else 0.0,
          "octave_error_rate": sum_col(dicts, "n_octave") / v if v else 0.0,
          "gross_error_rate": sum_col(dicts, "n_gross") / v if v else 0.0}
    vm = {"precision": tp / (tp + fp) if (tp + fp) else 0.0,
          "recall": tp / (tp + fn) if (tp + fn) else 0.0}
    vm["f1"] = 0.0 if (vm["precision"] + vm["recall"]) == 0 else \
        2 * vm["precision"] * vm["recall"] / (vm["precision"] + vm["recall"])
    coverage = v / (tp + fn) if (tp + fn) else 0.0
    return pm, vm, coverage


def reduce_combined(ds):
    pm, vm, _ = agg_from_sums(ds)
    return calculate_combined_score(vm, pm)


FRAME_STAT_COLS = ("valid", "n_rpa", "sum_cents", "n_octave", "n_gross", "tp", "fp", "fn")
FRAME_REDUCERS = {
    "combined": reduce_combined,
    "rpa": lambda ds: agg_from_sums(ds)[0]["rpa"],
    "voicing_f1": lambda ds: agg_from_sums(ds)[1]["f1"],
    "coverage": lambda ds: agg_from_sums(ds)[2],
    "cents_mae": lambda ds: agg_from_sums(ds)[0]["cents_error"],
}


def frame_keyed(pc):
    """Keyed cluster sums (+ #clips) for one per_clip block's suff-stat columns."""
    idx = [(c, pc["schema"].index(c)) for c in FRAME_STAT_COLS]
    return keyed_group_sums(pc["rows"], idx), len(pc["rows"])


def ci_cell(value, lo, hi, fmt="{:.3f}"):
    """A 'value [lo, hi]' markdown cell (shared by the frame + note CI tables). No interval
    (single cluster, see cluster_bootstrap) renders as '[n/a]'."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{fmt.format(value)} [n/a]"
    return f"{fmt.format(value)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def compare_keyed(keyed_a, keyed_b, metric="voicing_f1", n_boot=2000, seed=0):
    """Paired comparison of two keyed-cluster-sum dicts on their COMMON clusters.
    Returns (delta, ci_lo, ci_hi): point delta = reduce(A) - reduce(B) over all common clusters,
    CI from the paired cluster bootstrap. (delta, nan, nan) with < 2 common clusters."""
    reduce_fn = FRAME_REDUCERS[metric]
    common = sorted(set(keyed_a) & set(keyed_b))
    if not common:
        return float("nan"), float("nan"), float("nan")
    pa, pb = [keyed_a[g] for g in common], [keyed_b[g] for g in common]
    delta = float(reduce_fn(pa) - reduce_fn(pb))
    lo, hi = cluster_bootstrap(pa, reduce_fn, n_boot, seed, per_group_b=pb)
    return delta, lo, hi


def compare(per_clip_a, per_clip_b, metric="voicing_f1", n_boot=2000, seed=0):
    """Paired comparison of two algorithms from their per_clip blocks (same cell, both scored the
    same clips). Returns (delta, ci_lo, ci_hi) of A - B on the chosen frame metric."""
    return compare_keyed(frame_keyed(per_clip_a)[0], frame_keyed(per_clip_b)[0],
                         metric, n_boot, seed)
