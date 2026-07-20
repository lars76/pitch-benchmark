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


def pitch_prf(n_ok, tp, fp, fn) -> tuple[float, float, float]:
    """The pitch F-score family: precision/recall/F of the event "truly voiced frame,
    voiced with pitch output, within tolerance". n_ok = count of tp frames within tolerance; the
    denominators are the voicing sets, so every voicing mistake is charged and abstention
    can never inflate the score. Harmonic mean of P and R collapses to the Dice ratio
    2*n_ok/((tp+fp)+(tp+fn)) because both share the numerator. T->inf (n_ok == tp)
    recovers classic voicing F1."""
    r = n_ok / (tp + fn) if (tp + fn) > 0 else 0.0
    p = n_ok / (tp + fp) if (tp + fp) > 0 else 0.0
    denom = (tp + fp) + (tp + fn)
    f = 2.0 * n_ok / denom if denom > 0 else 0.0
    return p, r, f


# --------------------------------------------------------------------------- #
# Online (streaming) accumulation
# --------------------------------------------------------------------------- #
# Every frame-level metric above is a count or mean over evaluated frames, hence additive: clips
# fold in one at a time into running sufficient statistics, never holding the whole dataset's
# contours in memory. Results are identical to running evaluate_*() on the fully-concatenated
# arrays (same finite-frame rule, same octave/gross definitions). Memory is O(#bands).
PITCH_TOLERANCES = (10.0, 25.0, 50.0)  # cents tolerances for the pitch F-score family
T_MAX = 100.0                          # tolerance-AUC truncation: AUC = 1 - truncMAE/T_MAX
GROSS_ERROR_CENTS = 200.0


class _PitchBin:
    """Running pitch-accuracy sufficient statistics over a subset of frames (overall or a band)."""

    __slots__ = ("n_gross", "n_octave", "n_ok10", "n_ok25", "n_ok50", "n_rca",
                 "sum_cents", "sum_sq", "sum_trunc", "valid")

    def __init__(self):
        self.valid = 0
        self.n_ok10 = self.n_ok25 = self.n_ok50 = 0
        self.n_rca = self.n_gross = self.n_octave = 0
        self.sum_cents = self.sum_sq = self.sum_trunc = 0.0

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
        self.sum_trunc += float(np.minimum(abs_cents, T_MAX).sum())
        self.n_ok10 += int(np.sum(abs_cents < 10.0))
        self.n_ok25 += int(np.sum(abs_cents < 25.0))
        self.n_ok50 += int(np.sum(abs_cents < 50.0))
        wrapped = abs_cents % 1200
        chroma = np.minimum(wrapped, 1200 - wrapped)
        self.n_rca += int(np.sum(chroma < 50.0))
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
            "rpa": self.n_ok50 / v,
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
        cluster bootstrap can resample clips, sum, and RECOMPUTE any aggregate (pitch F / voicing F1 /
        coverage / cents / AUC) frame-weighted: the honest way to CI a nonlinear aggregate."""
        o = self._overall
        return {"valid": int(o.valid), "n_ok10": int(o.n_ok10), "n_ok25": int(o.n_ok25),
                "n_ok50": int(o.n_ok50), "sum_trunc_cents": float(o.sum_trunc),
                "sum_cents": float(o.sum_cents), "n_octave": int(o.n_octave),
                "n_gross": int(o.n_gross),
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

    def pitch_f_metrics(self) -> dict:
        """The pitch F-score family at every tolerance + the tolerance AUC."""
        o = self._overall
        out = {}
        for tol, n_ok in (("10", o.n_ok10), ("25", o.n_ok25), ("50", o.n_ok50)):
            p, r, f = pitch_prf(n_ok, self.tp, self.fp, self.fn)
            out[f"f{tol}"] = f
            if tol == "50":
                out["r50"], out["p50"] = r, p
        denom = (self.tp + self.fp) + (self.tp + self.fn)
        out["auc"] = (2.0 * (o.valid * T_MAX - o.sum_trunc) / (denom * T_MAX)
                      if denom > 0 else 0.0)
        return out


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


def sweep_summary(accumulators, thresholds):
    """The per-threshold aggregate block: one entry per threshold, NO selection. Threshold
    choice is a report-time decision (a global per-algorithm theta*), never a per-cell one."""
    out = []
    for threshold, acc in zip(thresholds, accumulators):
        v, p = acc.voicing_metrics(), acc.pitch_metrics()
        out.append({
            "threshold": float(threshold),
            "voicing": v,
            "pitch_f": acc.pitch_f_metrics(),
            "pitch": {**p, "coverage": acc.pitch_coverage()},
            "per_band": acc.per_band(),
            "smoothness": acc.smoothness_metrics(),
        })
    return out

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


def paired_delta_ci(pairs, reduce_fn=None, n_boot=2000):
    """95% CI of the mean-over-datasets PAIRED metric delta (clean - degraded), in
    percentage points. `pairs` = per dataset (keyed_clean, keyed_degraded) cluster sums over the
    SAME probe clips; each bootstrap draw resamples clusters within each dataset (stratified) and
    applies the same picks to both sides, so shared clip difficulty cancels. Datasets with < 2
    common clusters cannot be resampled and are skipped. Returns (lo, hi, n_datasets_used)."""
    reduce_fn = reduce_fn or FRAME_REDUCERS["pitch_f"]
    per_ds = []
    for k, (kc, kd) in enumerate(pairs):
        common = sorted(set(kc) & set(kd))
        if len(common) < 2:
            continue
        per_ds.append(boot_vals([kc[g] for g in common], reduce_fn,
                                n_boot, seed=k, per_group_b=[kd[g] for g in common]))
    if not per_ds:
        return float("nan"), float("nan"), 0
    vals = np.mean(per_ds, axis=0) * 100.0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(per_ds)


def sum_col(dicts, k):
    return sum(d[k] for d in dicts)


# Frame reducers: recompute the FRAME-weighted aggregate from summed sufficient stats (so CIs match
# the reported per-threshold sweep values exactly).
def agg_from_sums(dicts):
    v = sum_col(dicts, "valid")
    tp, fp, fn = sum_col(dicts, "tp"), sum_col(dicts, "fp"), sum_col(dicts, "fn")
    pm = {"rpa": sum_col(dicts, "n_ok50") / v if v else 0.0,
          "cents_error": sum_col(dicts, "sum_cents") / v if v else 0.0,
          "octave_error_rate": sum_col(dicts, "n_octave") / v if v else 0.0,
          "gross_error_rate": sum_col(dicts, "n_gross") / v if v else 0.0}
    vm = {"precision": tp / (tp + fp) if (tp + fp) else 0.0,
          "recall": tp / (tp + fn) if (tp + fn) else 0.0}
    vm["f1"] = 0.0 if (vm["precision"] + vm["recall"]) == 0 else \
        2 * vm["precision"] * vm["recall"] / (vm["precision"] + vm["recall"])
    coverage = v / (tp + fn) if (tp + fn) else 0.0
    return pm, vm, coverage


def _prf_from_sums(ds, tol):
    tp, fp, fn = sum_col(ds, "tp"), sum_col(ds, "fp"), sum_col(ds, "fn")
    return pitch_prf(sum_col(ds, f"n_ok{tol}"), tp, fp, fn)


def _auc_from_sums(ds):
    v, trunc = sum_col(ds, "valid"), sum_col(ds, "sum_trunc_cents")
    denom = (sum_col(ds, "tp") + sum_col(ds, "fp")) + (sum_col(ds, "tp") + sum_col(ds, "fn"))
    return 2.0 * (v * T_MAX - trunc) / (denom * T_MAX) if denom > 0 else 0.0


FRAME_STAT_COLS = ("valid", "n_ok10", "n_ok25", "n_ok50", "sum_trunc_cents",
                   "sum_cents", "n_octave", "n_gross", "tp", "fp", "fn")
FRAME_REDUCERS = {
    "pitch_f": lambda ds: _prf_from_sums(ds, 50)[2],
    "pitch_f25": lambda ds: _prf_from_sums(ds, 25)[2],
    "pitch_f10": lambda ds: _prf_from_sums(ds, 10)[2],
    "pitch_recall": lambda ds: _prf_from_sums(ds, 50)[1],
    "pitch_precision": lambda ds: _prf_from_sums(ds, 50)[0],
    "tol_auc": _auc_from_sums,
    "rpa": lambda ds: agg_from_sums(ds)[0]["rpa"],       # accuracy on detected frames
    "voicing_f1": lambda ds: agg_from_sums(ds)[1]["f1"],
    "coverage": lambda ds: agg_from_sums(ds)[2],
    "cents_mae": lambda ds: agg_from_sums(ds)[0]["cents_error"],
}


def frame_keyed(pc, theta_idx):
    """Keyed cluster sums (+ #clips) for one per_clip block at ONE threshold index. The v2
    per_clip block stores, per clip, the suff-stat row for every threshold; theta_idx picks
    the operating point (a report-time decision, typically the algorithm's theta*)."""
    rows = [list(meta[:2]) + list(stats[theta_idx])
            for meta, stats in zip(pc["clips"], pc["stats"])]
    idx = [(c, 2 + i) for i, c in enumerate(FRAME_STAT_COLS)]
    return keyed_group_sums(rows, idx), len(rows)


def ci_cell(value, lo, hi, fmt="{:.3f}"):
    """A 'value [lo, hi]' markdown cell (shared by the frame + note CI tables). No interval
    (single cluster, see cluster_bootstrap) renders as '[n/a]'."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{fmt.format(value)} [n/a]"
    return f"{fmt.format(value)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def compare_keyed(keyed_a, keyed_b, metric="pitch_f", n_boot=2000, seed=0):
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


def compare(per_clip_a, per_clip_b, theta_idx_a, theta_idx_b, metric="pitch_f",
            n_boot=2000, seed=0):
    """Paired comparison of two algorithms from their per_clip blocks (same cell, both scored
    the same clips), each read at its own operating point. Returns (delta, ci_lo, ci_hi) of
    A - B on the chosen frame metric."""
    return compare_keyed(frame_keyed(per_clip_a, theta_idx_a)[0],
                         frame_keyed(per_clip_b, theta_idx_b)[0], metric, n_boot, seed)


# --------------------------------------------------------------------------- #
# Track scoring (the v2 question layer): pure functions over a load_cells dict.
# Tracks are SITUATIONS of use, never decomposition axes, so nothing is scored twice;
# every decomposition (voicing P/R, accuracy-on-voiced, octave, cents...) is a
# diagnostic rendered beneath its track. Threshold selection happens HERE (one global
# theta* per algorithm), never per cell.
# --------------------------------------------------------------------------- #
SIGMA0 = 10.0        # cents; steady-jitter normalizer s = SIGMA0/(SIGMA0+jitter)
TRACKS_SCORED = ("accuracy", "noise", "signals", "stability", "dynamics", "notes", "speed")


def _frame_cells(cells, algo, condition, datasets=None):
    for (track, ds, cond, a), cell in cells.items():
        if track != "frame" or a != algo or cond != condition:
            continue
        if datasets and ds not in datasets:
            continue
        if cell.get("metadata", {}).get("crashed"):
            continue
        if cell.get("results", {}).get("sweep"):
            yield ds, cell


def _sweep_f50(cell, idx):
    return cell["results"]["sweep"][idx]["pitch_f"]["f50"]


def theta_star(cells, algo, *, datasets=None):
    """The algorithm's ONE frozen operating point: argmax over the threshold grid of the
    equal-per-dataset mean pitch F@50 on clean frame cells. Full (uncapped) clean cells
    are preferred; if only probe-sized clean baselines exist the fallback is used and
    stamped in the provenance. Ties resolve to the lowest threshold.
    Returns {"theta", "idx", "provenance", "n_datasets"}."""
    full, probe = {}, {}
    for ds, cell in _frame_cells(cells, algo, "clean", datasets):
        (probe if cell.get("metadata", {}).get("probe") else full)[ds] = cell
    pool, provenance = (full, "clean-full") if full else (probe, "clean-probe")
    if not pool:
        return {"theta": None, "idx": None, "provenance": "no-clean-cells", "n_datasets": 0}
    n_thr = len(next(iter(pool.values()))["results"]["thresholds"])
    means = [float(np.mean([_sweep_f50(c, i) for c in pool.values()])) for i in range(n_thr)]
    idx = int(np.argmax(means))                       # argmax returns the FIRST max: lowest theta
    theta = next(iter(pool.values()))["results"]["thresholds"][idx]
    return {"theta": float(theta), "idx": idx, "provenance": provenance,
            "n_datasets": len(pool)}


def oracle_theta(cell):
    """Per-cell F@50 argmax index (the oracle the stability track measures against)."""
    sweep = cell["results"]["sweep"]
    return int(np.argmax([e["pitch_f"]["f50"] for e in sweep]))


def _pooled_sums(cells, algo, *, theta_idx, datasets=None, conditions=("clean",)):
    """Summed suff stats over the selected frame cells at one threshold index."""
    total = None
    for cond in conditions:
        for _ds, cell in _frame_cells(cells, algo, cond, datasets):
            pc = cell["results"].get("per_clip")
            if not pc:
                continue
            for stats in pc["stats"]:
                row = dict(zip(FRAME_STAT_COLS, stats[theta_idx]))
                total = row if total is None else {k: total[k] + row[k] for k in total}
    return total


def pitch_prf_from_cells(cells, algo, *, theta_idx, tolerance=50, datasets=None,
                         conditions=("clean",)):
    """Frame-pooled pitch P/R/F over the selected cells at one operating point."""
    s = _pooled_sums(cells, algo, theta_idx=theta_idx, datasets=datasets,
                     conditions=conditions)
    if not s:
        return None
    p, r, f = pitch_prf(s[f"n_ok{int(tolerance)}"], s["tp"], s["fp"], s["fn"])
    return {"precision": p, "recall": r, "f": f}


def factorization(cells, algo, *, theta_idx, datasets=None, conditions=("clean",)):
    """The identity pitch_recall = voicing_recall x accuracy_on_voiced, from pooled
    counts (both classic RPA definitions as the factors of one product)."""
    s = _pooled_sums(cells, algo, theta_idx=theta_idx, datasets=datasets,
                     conditions=conditions)
    if not s:
        return None
    p, r, f = pitch_prf(s["n_ok50"], s["tp"], s["fp"], s["fn"])
    vrec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
    return {"pitch_f": f, "pitch_recall": r, "pitch_precision": p,
            "voicing_recall": vrec,
            "accuracy_on_voiced": s["n_ok50"] / s["tp"] if s["tp"] else 0.0}


def track_accuracy(cells, algo, *, datasets=None):
    """How correct is the output curve on clean real recordings?"""
    star = theta_star(cells, algo, datasets=datasets)
    if star["idx"] is None:
        return {"score": None, **star}
    per_ds = {}
    for ds, cell in _frame_cells(cells, algo, "clean", datasets):
        e = cell["results"]["sweep"][star["idx"]]
        per_ds[ds] = {"f50": e["pitch_f"]["f50"], "auc": e["pitch_f"]["auc"],
                      "probe": bool(cell.get("metadata", {}).get("probe"))}
    if not per_ds:
        return {"score": None, **star}
    return {"score": float(np.mean([d["f50"] for d in per_ds.values()])),
            "auc": float(np.mean([d["auc"] for d in per_ds.values()])),
            "per_dataset": per_ds, **star}


def track_noise(cells, algo, *, datasets=None):
    """How much of the clean score survives real-world corruption? Ratio of pitch F@50
    degraded/clean on the SAME probe clips, equal-per-dataset then mean over conditions."""
    star = theta_star(cells, algo, datasets=datasets)
    if star["idx"] is None:
        return {"score": None, **star}
    clean_probe = {}
    for ds, cell in _frame_cells(cells, algo, "clean_probe", datasets):
        clean_probe[ds] = cell
    if not clean_probe:                       # store holds only one clean variant per ds
        for ds, cell in _frame_cells(cells, algo, "clean", datasets):
            if cell.get("metadata", {}).get("probe"):
                clean_probe[ds] = cell
    conds = sorted({k[2] for k in cells
                    if k[0] == "frame" and k[3] == algo
                    and k[2] not in ("clean", "clean_probe")})
    per_cond = {}
    for cond in conds:
        ratios = []
        for ds, cell in _frame_cells(cells, algo, cond, datasets):
            if ds not in clean_probe:
                continue
            fc = _sweep_f50(clean_probe[ds], star["idx"])
            if fc > 0:
                ratios.append(min(_sweep_f50(cell, star["idx"]) / fc, 1.5))
        if ratios:
            per_cond[cond] = float(np.mean(ratios))
    if not per_cond:
        return {"score": None, **star}
    return {"score": float(np.clip(np.mean(list(per_cond.values())), 0.0, 1.0)),
            "per_condition": per_cond, **star}


def track_signals(cells, algo):
    """How much of the synthetic signal-class space does it handle? Mean over the
    stationary families of pitch recall@50 (coverage-aware accuracy) at theta*; each
    family is one probe question, equally weighted. The worst family is carried as the
    named diagnostic (a min SCORE was measured to zero out most real trackers -- 6 of 7
    surveyed have at least one dead family -- destroying all downstream ranking)."""
    star = theta_star(cells, algo)
    if star["idx"] is None:
        return {"score": None, **star}
    fams = {}
    for (track, fam, _c, a), cell in cells.items():
        if track != "synthetic" or a != algo or cell.get("metadata", {}).get("crashed"):
            continue
        res = cell.get("results", {})
        if res.get("kind") == "stationary" and res.get("sweep"):
            fams[fam] = res["sweep"][star["idx"]]["pitch_f"]["r50"]
    if not fams:
        return {"score": None, **star}
    worst = min(fams, key=fams.get)
    return {"score": float(np.mean(list(fams.values()))), "worst_family": worst,
            "worst": float(fams[worst]), "per_family": fams, **star}


def track_stability(cells, algo, *, datasets=None):
    """Does one global threshold work everywhere? Mean over clean cells of
    F(theta*)/F(per-cell oracle). Binary-confidence trackers are trivially 1.0."""
    star = theta_star(cells, algo, datasets=datasets)
    if star["idx"] is None:
        return {"score": None, **star}
    ratios = {}
    for ds, cell in _frame_cells(cells, algo, "clean", datasets):
        oracle = _sweep_f50(cell, oracle_theta(cell))
        if oracle > 0:
            ratios[ds] = _sweep_f50(cell, star["idx"]) / oracle
    if not ratios:
        return {"score": None, **star}
    return {"score": float(np.clip(np.mean(list(ratios.values())), 0.0, 1.0)),
            "per_dataset": ratios, **star}


def track_dynamics(cells, algo):
    """Is a moving pitch followed faithfully? Steady tones (jitter, bias) + vibrato
    (depth retention x coverage); the two families the pilot showed discriminate."""
    star = theta_star(cells, algo)
    if star["idx"] is None:
        return {"score": None, **star}
    steady, vib = [], []
    detail = {}
    for (track, fam, _c, a), cell in cells.items():
        if track != "synthetic" or a != algo or cell.get("metadata", {}).get("crashed"):
            continue
        res = cell.get("results", {})
        if res.get("kind") != "trajectory" or not res.get("sweep"):
            continue
        e = res["sweep"][star["idx"]]
        detail[fam] = e
        if "jitter_cents" in e:
            steady.append(SIGMA0 / (SIGMA0 + e["jitter_cents"]))
        if "depth_retention" in e:
            vib.append(np.clip(e["depth_retention"], 0.0, 1.0) * e.get("coverage", 1.0))
    parts = [float(np.mean(x)) for x in (steady, vib) if x]
    if not parts:
        return {"score": None, **star}
    return {"score": float(np.mean(parts)), "per_family": detail, **star}


def track_notes(cells, algo):
    """Is musical note structure recoverable? Mean COnP over note datasets (this track
    selects its own threshold x segmentation-cost internally, documented)."""
    vals = {}
    for (track, ds, _c, a), cell in cells.items():
        if track == "note" and a == algo and not cell.get("metadata", {}).get("crashed"):
            v = cell.get("results", {}).get("conp")
            if v is not None and np.isfinite(v):
                vals[ds] = v
    if not vals:
        return {"score": None}
    return {"score": float(np.mean(list(vals.values()))), "per_dataset": vals}


def track_speed(cells, algo):
    """Is it deployable? score = 1/(1+RTF_cpu)."""
    for (track, _ds, _c, a), cell in cells.items():
        if track == "speed" and a == algo:
            dev = cell.get("results", {}).get("device_results", {}).get("cpu", {})
            ms = dev.get("absolute_time_ms")
            if ms is None:
                return {"score": None}
            sec = cell.get("parameters", {}).get("signal_length_sec", 1.0)
            rtf = (ms / 1000.0) / sec
            return {"score": 1.0 / (1.0 + rtf), "rtf_cpu": rtf}
    return {"score": None}


def track_scores(cells, algo, *, datasets=None):
    """The seven track scores (None where a track has no cells)."""
    return {
        "accuracy": track_accuracy(cells, algo, datasets=datasets)["score"],
        "noise": track_noise(cells, algo, datasets=datasets)["score"],
        "signals": track_signals(cells, algo)["score"],
        "stability": track_stability(cells, algo, datasets=datasets)["score"],
        "dynamics": track_dynamics(cells, algo)["score"],
        "notes": track_notes(cells, algo)["score"],
        "speed": track_speed(cells, algo)["score"],
    }


def overall(cells, algo, *, datasets=None):
    """Harmonic mean of the seven track scores, equal weights: the same mean family as
    the F-scores inside the tracks, and the one that punishes a weak axis hardest (HM
    is dominated by the smallest score). None if ANY track is missing (never a silent
    partial mean); a zero track makes it 0 (a failed axis sinks it)."""
    scores = track_scores(cells, algo, datasets=datasets)
    vals = list(scores.values())
    if any(v is None for v in vals):
        return None
    if any(v <= 0 for v in vals):
        return 0.0
    return float(len(vals) / np.sum(1.0 / np.asarray(vals)))
