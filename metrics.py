"""Pure-numpy evaluation core: pitch bands + frame-level metric functions.

Torch-free so it imports (and tests) without torch/torchaudio. Used by the measurement
libraries (frame/note/synthetic_benchmark) and the report (generate_report.py).
"""
import hashlib
import math
import os
from dataclasses import dataclass
from enum import Enum

import numpy as np


def clip_and_group(dataset, wav_path, idx):
    """(clip_id, group) for a per-clip row: the ONE place the frame runner derives them, so
    its cluster-bootstrap CIs cluster correctly.

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
def pitch_prf(n_correct, tp, fp, fn) -> tuple[float, float, float]:
    """THE metric family -- one definition, parameterised only by the cents tolerance.

    `n_correct` counts frames that are truly voiced AND given a pitch by the tracker AND
    within tolerance. The denominators are the voicing sets, so every voicing mistake is
    charged and abstention can never inflate the score:

        pitch precision = n_correct / (tp + fp)     of what it output, how much was right
        pitch recall    = n_correct / (tp + fn)     of the pitch that exists, how much it got
        pitch F1        = 2 n_correct / ((tp+fp) + (tp+fn))

    P and R share a numerator, so their harmonic mean collapses to that Dice ratio --
    which is why F1 here is DERIVED, not chosen.

    Voicing precision / recall / F1 are the SAME formulas with the pitch test removed
    (a tolerance wide enough to accept any pitch, which is finite: the largest possible
    error is 1200*log2(fmax/fmin), about 6600 cents over a 46-2094 Hz range). So the
    benchmark has one metric family, not two.
    """
    r = n_correct / (tp + fn) if (tp + fn) > 0 else 0.0
    p = n_correct / (tp + fp) if (tp + fp) > 0 else 0.0
    denom = (tp + fp) + (tp + fn)
    f = 2.0 * n_correct / denom if denom > 0 else 0.0
    return p, r, f


# --------------------------------------------------------------------------- #
# Online (streaming) accumulation
# --------------------------------------------------------------------------- #
# Every frame-level metric above is a count or mean over evaluated frames, hence additive: clips
# fold in one at a time into running sufficient statistics, never holding the whole dataset's
# contours in memory. Folding clips in one at a time is identical to scoring the fully
# concatenated arrays (same finite-frame rule, same error-class definitions). O(#bands) memory.
# The two scored tolerances. An intermediate rung was dropped as redundant with its
# neighbours; EXACT and CORRECT are what the error classes need.
EXACT_CENTS = 10.0     # below the pitch JND: an inaudible error
CORRECT_CENTS = 50.0   # half a semitone: the melody-extraction convention
WRONG_CENTS = 200.0    # two semitones: a different pitch entirely


def error_rates(counts) -> dict:
    """The conditional diagnostics over the frames that were scored for pitch, from the
    sufficient statistics. ONE implementation, so the streaming accumulator and a
    bootstrap resample of summed per-clip rows cannot drift apart.

    `correct_rate` is exact+close, the literature's RPA. `cents_bias` is the mean SIGNED
    error over the IN-TOLERANCE frames only: 0 means symmetric scatter, non-zero means the
    tracker sits systematically sharp (+) or flat (-) and can be corrected by subtracting
    it. Its denominator is n_correct, NOT frames, so it is not comparable to cents_mae."""
    v = counts["frames"]
    if not v:
        return {"cents_mae": np.nan, "cents_bias": np.nan, "correct_rate": np.nan,
                "octave_rate": np.nan, "wrong_rate": np.nan, "frames": 0}
    ok = counts["n_correct"]
    return {"cents_mae": counts["sum_cents"] / v,
            "cents_bias": (counts.get("sum_signed_cents_correct", 0.0) / ok
                           if ok else np.nan),
            "correct_rate": counts["n_correct"] / v,
            "octave_rate": counts["n_octave"] / v,
            "wrong_rate": counts["n_wrong"] / v,
            "frames": v}


class _PitchBin:
    """Running pitch sufficient statistics over a subset of frames (overall or a band).

    Counts only; every reported rate is derived from these by division, so summing bins
    across clips reproduces the aggregate exactly (the property the bootstrap relies on).
    """

    __slots__ = ("n_correct", "n_exact", "n_octave", "n_wrong",
                 "sum_cents", "sum_signed", "valid")

    def __init__(self):
        self.valid = 0
        self.n_exact = self.n_correct = self.n_wrong = self.n_octave = 0
        self.sum_cents = self.sum_signed = 0.0

    def add(self, pitch_pred: np.ndarray, pitch_true: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        pred, true = pitch_pred[mask], pitch_true[mask]
        signed = cents(pred, true)
        finite = np.isfinite(signed)  # drop degenerate frames (pred/true <= 0 -> non-finite cents)
        signed = signed[finite]
        abs_cents = np.abs(signed)
        if abs_cents.size == 0:
            return
        self.valid += int(abs_cents.size)
        self.sum_cents += float(abs_cents.sum())
        self.n_exact += int(np.sum(abs_cents < EXACT_CENTS))
        within = abs_cents < CORRECT_CENTS
        self.n_correct += int(np.sum(within))
        # SIGNED sum, over the IN-TOLERANCE frames only. |error| alone cannot distinguish a
        # tracker running consistently sharp from one scattering symmetrically, and those
        # are different defects. Restricting to in-tolerance frames is what makes it mean
        # that: over ALL scored frames the mean signed error is dominated by the octave
        # tail, so it reports a large spurious "offset" for a tracker whose typical error is
        # small. A sum over a counted subset stays additive; a median, the other
        # obvious fix, does not.
        self.sum_signed += float(signed[within].sum())
        self.n_wrong += int(np.sum(abs_cents > WRONG_CENTS))
        # octave = within a semitone of a whole-octave multiple; a strict SUBSET of wrong
        # (the smallest octave error is 1100c), so wrong - octave is the non-harmonic mass
        nearest_octave = np.round(abs_cents / 1200.0)
        self.n_octave += int(
            np.sum((nearest_octave >= 1) & (np.abs(abs_cents - nearest_octave * 1200.0) < 100.0))
        )

    def counts(self) -> dict:
        """This bin's additive sufficient statistics, under the canonical column names."""
        return {"frames": int(self.valid), "n_exact": int(self.n_exact),
                "n_correct": int(self.n_correct), "n_wrong": int(self.n_wrong),
                "n_octave": int(self.n_octave), "sum_cents": float(self.sum_cents),
                "sum_signed_cents_correct": float(self.sum_signed)}

    def result(self) -> dict:
        return error_rates(self.counts())


class MetricAccumulator:
    """Streaming metrics for ONE voicing threshold; fold clips in with update(), read at the end.

    Equivalent to concatenating every clip and scoring the whole array at once, but with
    O(#bands) memory.
    """

    def __init__(self):
        self.bands = PITCH_BANDS
        self.tau = VOICED_THRESHOLD         # pitch_conf threshold for scoring, when provided
        self.tp = self.fp = self.fn = 0
        self._overall = _PitchBin()
        self._band = {name: _PitchBin() for name, _, _ in PITCH_BANDS}

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

    def voicing_metrics(self) -> dict:
        """The same family with the pitch test removed: every voiced frame counts as
        correct, which is pitch_prf's n_correct = tp."""
        p, r, f1 = pitch_prf(self.tp, self.tp, self.fp, self.fn)
        return {"precision": p, "recall": r, "f1": f1}

    def error_metrics(self) -> dict:
        """CONDITIONAL diagnostics over the frames scored for pitch (both sides voiced)."""
        return self._overall.result()

    def suff_stats(self) -> dict:
        """The additive sufficient statistics behind every scalar metric (overall pitch bin +
        voicing counts). Summing these across clips reproduces the full-dataset accumulator
        EXACTLY, so a cluster bootstrap can resample clips, sum, and RECOMPUTE any aggregate
        frame-weighted: the honest way to CI a nonlinear aggregate."""
        return {**self._overall.counts(),
                "tp": int(self.tp), "fp": int(self.fp), "fn": int(self.fn)}

    def per_band(self) -> dict:
        """Same conditional diagnostics, split by ground-truth pitch band."""
        return {name: self._band[name].result() for name, _, _ in self.bands}

    def pitch_metrics(self) -> dict:
        """The UNCONDITIONAL pitch family at this threshold: precision / recall / F1 of the
        event "truly voiced AND given a pitch AND within CORRECT_CENTS"."""
        p, r, f1 = pitch_prf(self._overall.n_correct, self.tp, self.fp, self.fn)
        return {"f1": f1, "recall": r, "precision": p}


# --------------------------------------------------------------------------- #
# Shared evaluation helpers (used by the frame and synthetic runners)
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


def sweep_summary(accumulators, thresholds):
    """The per-threshold aggregate block: one entry per threshold, NO selection. Threshold
    choice is a report-time decision (a global per-algorithm theta*), never a per-cell one."""
    out = []
    for threshold, acc in zip(thresholds, accumulators):
        out.append({
            "threshold": float(threshold),
            "pitch": acc.pitch_metrics(),        # unconditional family: f1 / recall / precision
            "voicing": acc.voicing_metrics(),    # the same family with the pitch test removed
            "errors": acc.error_metrics(),       # conditional diagnostics on scored frames
            "per_band": acc.per_band(),
        })
    return out

# --------------------------------------------------------------------------- #
# Cluster-bootstrap statistics over per-clip sufficient stats (the ONE bootstrap, shared by the
# report tables and evaluate.compare). Always cluster by the per-clip `group` (speaker/singer/
# piece via get_group), never by clip; correlated clips would give false precision. We pre-SUM
# each cluster's columns once (metrics are additive), resample cluster INDICES vectorized, and a
# reducer turns the summed picks into the frame-weighted aggregate (pitch F1 / voicing F1 /
# frame-weighted aggregate. Because every reducer consumes only column SUMS,
# evaluating it on one pre-summed dict is exactly equivalent to handing it the picked dicts --
# the scalar reducers stay the single, verified formula implementation.
# --------------------------------------------------------------------------- #
def keyed_group_sums(rows, cols):
    """Per cluster {name: Σcolumn, ..., '_n': #clips} KEYED by the cluster id (so two algorithms
    scored on the same clips can be paired cluster-by-cluster). `cols` = [(name, col_index)].
    Deliberately NO clip-level fallback for a single-group result: between-source variance is
    inestimable from one source (clips of one source are correlated, so resampling them would
    understate the real uncertainty). Callers render `[n/a]` and make no tie claims. Every
    registered dataset exposes real source clusters via get_group, so this never triggers in
    practice; it exists for degenerate inputs (e.g. a custom dataset that is one source)."""
    groups = {}
    for r in rows:
        groups.setdefault(r[1], []).append(r)          # column 1 is the cluster key
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
    """95% CI of the mean-over-datasets PAIRED metric delta (clean - degraded), in
    percentage points. `pairs` = per dataset (keyed_clean, keyed_degraded) cluster sums over the
    SAME probe clips; each bootstrap draw resamples clusters within each dataset (stratified) and
    applies the same picks to both sides, so shared clip difficulty cancels. Datasets with < 2
    common clusters cannot be resampled and are skipped. Returns (lo, hi, n_datasets_used)."""
    reduce_fn = FRAME_REDUCERS["pitch_f1"]
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
    """(errors, voicing) recomputed from summed sufficient statistics -- the same
    formulas the accumulator uses, so a bootstrap resample reproduces the reported
    value exactly."""
    tp, fp, fn = sum_col(dicts, "tp"), sum_col(dicts, "fp"), sum_col(dicts, "fn")
    errors = error_rates({c: sum_col(dicts, c) for c in FRAME_STAT_COLS})
    p, r, f1 = pitch_prf(tp, tp, fp, fn)
    return errors, {"precision": p, "recall": r, "f1": f1}


def _prf_from_sums(ds):
    return pitch_prf(sum_col(ds, "n_correct"), sum_col(ds, "tp"),
                     sum_col(ds, "fp"), sum_col(ds, "fn"))


# The stored per-clip sufficient statistics. Every one is a COUNT (or a sum of cents),
# so they are additive across clips; every reported rate is a ratio of these.
FRAME_STAT_COLS = ("frames", "n_exact", "n_correct", "n_wrong", "n_octave",
                   "sum_cents", "sum_signed_cents_correct", "tp", "fp", "fn")
FRAME_REDUCERS = {
    # unconditional pitch family (the headline)
    "pitch_f1": lambda ds: _prf_from_sums(ds)[2],
    "pitch_recall": lambda ds: _prf_from_sums(ds)[1],
    "pitch_precision": lambda ds: _prf_from_sums(ds)[0],
    # the same family with the pitch test removed
    "voicing_f1": lambda ds: agg_from_sums(ds)[1]["f1"],
    "voicing_recall": lambda ds: agg_from_sums(ds)[1]["recall"],
    # conditional diagnostics
    "correct_rate": lambda ds: agg_from_sums(ds)[0]["correct_rate"],
    "cents_mae": lambda ds: agg_from_sums(ds)[0]["cents_mae"],
}


# The metadata fields load_cells REQUIRES to route a cell to its key. Part of the stored
# format: renaming one is exactly as breaking as renaming a stat column, and the reader would
# otherwise accept the cell and then fail (or mis-route it) one layer down.
CELL_ENVELOPE = ("suite", "algorithm_name", "format")


def format_id():
    """A short id for the STORED cell format: the metadata fields the reader routes on, the
    per-clip columns, and the actual key structure a sweep entry carries. Stamped on every
    cell at write time and compared at read time, so a cell written under a different layout
    is refused rather than silently misread.

    One comparison, derived from the real objects, replaces two hand-maintained guards that
    between them covered the stat columns and one sweep key -- a renamed `errors.cents_mae`
    or `per_band.correct_rate` passed both, and every reader of those uses `.get()`, so the
    field simply vanished from the report with no error."""
    entry = sweep_summary([MetricAccumulator()], [0.5])[0]
    shape = (CELL_ENVELOPE, FRAME_STAT_COLS,
             tuple(sorted(entry["pitch"])), tuple(sorted(entry["voicing"])),
             tuple(sorted(entry["errors"])), tuple(sorted(next(iter(entry["per_band"].values())))))
    return hashlib.sha1(repr(shape).encode()).hexdigest()[:8]


def check_format(cell):
    """Refuse a cell whose stored format is not the current one."""
    got = (cell.get("metadata") or {}).get("format")
    if got != format_id():
        raise ValueError(
            f"cell format {got!r} is not the current {format_id()!r}: it was written under "
            "different metric definitions and must be regenerated "
            "(rm the cells and re-run)")


def frame_keyed(pc, theta_idx):
    """Keyed cluster sums (+ #clips) for one per_clip block at ONE threshold index. The v2
    per_clip block stores, per clip, the suff-stat row for every threshold; theta_idx picks
    the operating point (a report-time decision, typically the algorithm's theta*)."""
    # strict: clips and stats are stored as two independent lists, so a truncated write
    # would silently drop the tail clips from every bootstrap rather than fail
    rows = [list(meta[:2]) + list(stats[theta_idx])
            for meta, stats in zip(pc["clips"], pc["stats"], strict=True)]
    idx = [(c, 2 + i) for i, c in enumerate(FRAME_STAT_COLS)]
    return keyed_group_sums(rows, idx), len(rows)


def compare_keyed(keyed_a, keyed_b, metric="pitch_f1", n_boot=2000, seed=0):
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


# --------------------------------------------------------------------------- #
# Track scoring (the v2 question layer): pure functions over a load_cells dict.
# Tracks are SITUATIONS of use, never decomposition axes, so nothing is scored twice;
# every decomposition (voicing P/R, accuracy-on-voiced, octave, cents...) is a
# diagnostic rendered beneath its track. Threshold selection happens HERE (one global
# theta* per algorithm), never per cell.
# --------------------------------------------------------------------------- #
class StrEnum(str, Enum):
    """enum.StrEnum, which arrived in 3.11 and the algorithm matrix pins us below.

    The `__str__` is the whole point: without it a (str, Enum) member formats as its VALUE
    under an f-string but as `Class.MEMBER` under str() and %s, so the same member produces
    two different strings depending on how it is written out -- and a filename or an argv
    token built the wrong way is silently wrong rather than an error."""

    def __str__(self):
        return self.value


class Suite(StrEnum):
    """The four measurement libraries a run can invoke. Deliberately NOT called tracks:
    a TRACK is one of the six scored questions in the report. These four are independent
    and order-free -- any subset, in any order.

    THE declaration of the set: `evaluate.SUITES` is derived from it, so there is one place
    to add a fifth. str-valued so a value read from JSON compares and hashes equal to the
    member, which is what lets cells on disk key a dict of CellKeys."""

    FRAME = "frame"
    NOTE = "note"
    SYNTHETIC = "synthetic"
    SPEED = "speed"


@dataclass(frozen=True)
class Score:
    """A track score with its interval. `lo`/`hi` are None where the resampling unit is
    unsound (families, or a single measurement) -- absent rather than manufactured."""

    value: float | None
    lo: float | None = None
    hi: float | None = None


@dataclass(frozen=True)
class Delta:
    """A paired A - B comparison. `significant` is derived here so no consumer has to
    re-derive the rule (and get it subtly different)."""

    value: float
    lo: float
    hi: float

    @property
    def significant(self) -> bool:
        return self.lo > 0 or self.hi < 0


TRACK_ORDER = ("correctness", "noise", "signal_types", "tracking", "notes", "speed")


@dataclass(frozen=True)
class Tracks:
    """The six scored questions. ATTRIBUTES, not string keys: the set is closed, so a
    typo should fail at once instead of returning a KeyError or a silent .get() miss."""

    correctness: Score
    noise: Score
    signal_types: Score
    tracking: Score
    notes: Score
    speed: Score

    def items(self):
        """(name, Score) in report order."""
        return [(f, getattr(self, f)) for f in TRACK_ORDER]

    def values(self):
        return [s.value for _n, s in self.items()]


@dataclass(frozen=True, kw_only=True, slots=True)
class CellKey:
    """What identifies a measurement.

    Keyword-only on purpose: `subject` and `condition` are adjacent `str | None` fields, so a
    positional swap would build a key that matches no cell and read back as "no data" rather
    than as an error. Not a tuple, so it cannot be indexed, sliced, unpacked, or compared
    equal to one -- a key is a set of named parts, and any code treating it as a sequence is
    relying on field ORDER, which is not something this type promises.

    `cap` is a field because it distinguishes MEASUREMENTS, not just files: a 30-clip and a
    50-clip probe of the same (dataset, condition, algo) are two different measurements.
    It is also what keeps the clean baseline distinct without a pseudo-condition: the full
    clean cell is cap=None and a capped baseline is its own cap.
    """

    suite: "Suite"
    subject: str | None           # dataset name, or synthetic family; None for speed
    condition: str | None
    algo: str
    cap: tuple | None = None      # (max_clips, max_seconds); None = uncapped


def cap_of(cell):
    """The cap a cell was measured under, or None if uncapped. Read from the recorded
    parameters, never from the filename."""
    p = cell.get("parameters") or {}
    n, s = p.get("max_clips"), p.get("max_seconds")
    return (n, s) if (n or s) else None


SIGMA0 = 10.0        # cents; steady-jitter normalizer s = SIGMA0/(SIGMA0+jitter)

def _eligible(key, algo, datasets):
    """Shared row filter: right algorithm, inside any narrowing. Which datasets are in a run
    is the caller's choice -- a dataset enters only when its path is supplied -- so the scorer
    imposes no dataset policy of its own. A corpus with score-grade pitch GT (the notated note,
    not the performed f0) shifts theta* and every pitch table if included; that trade-off is the
    user's to make by supplying, or withholding, its path."""
    return (key.algo == algo
            and not (datasets and key.subject not in datasets))


def _readable(cell):
    """A cell that ran to completion and has something to score."""
    return (not cell.get("metadata", {}).get("crashed")
            and bool(cell.get("results", {}).get("sweep")))


def frame_cells(cells, algo, condition, datasets=None, *, any_cap=False):
    """(dataset, cap, cell) per dataset for this condition, preferring the UNCAPPED cell.

    A capped and an uncapped run of the same (dataset, condition) are different
    measurements and both live in `cells` under their own cap. The cap is yielded because
    every caller needs to know what a score was measured on; deriving it separately meant
    two copies of this preference rule that could disagree.

    If the UNCAPPED cell exists and crashed, the dataset yields NOTHING: a 30-clip probe
    must not silently stand in for the full run that segfaulted. `crashed_datasets` reports
    it and the track scores it 0. `any_cap=True` relaxes exactly that, for theta*, which
    needs an operating point even on a dataset the algorithm failed -- the crash still
    costs that dataset its score, only the threshold is recovered.
    """
    best, uncapped_crashed = {}, set()
    for key, cell in cells.items():
        if key.suite != Suite.FRAME or key.condition != condition or not _eligible(key, algo, datasets):
            continue
        if cell.get("metadata", {}).get("crashed"):
            if key.cap is None:
                uncapped_crashed.add(key.subject)
            continue
        if not _readable(cell):
            continue
        prev = best.get(key.subject)
        if prev is None or (prev[0] is not None and key.cap is None):
            best[key.subject] = (key.cap, cell)
    for ds, (cap, cell) in best.items():
        if ds in uncapped_crashed and not any_cap:
            continue
        yield ds, cap, cell


def frame_cell_at(cells, algo, dataset, condition, cap):
    """The completed cell for exactly this cap, or None. Distinct from frame_cells: an
    exact lookup, not a preference (the Noise track pairs against the clean baseline at
    the SAME cap)."""
    cell = cells.get(CellKey(suite="frame", subject=dataset, condition=condition,
                             algo=algo, cap=cap))
    return cell if cell is not None and _readable(cell) else None


def crashed_datasets(cells, algo, *, conditions=None, datasets=None):
    """Datasets whose cell RAN AND DIED, for the given conditions.

    These score 0 in their track's own domain. Skipping them instead would grade a tracker
    that segfaults on a corpus only on the corpora it survived, so two trackers covering
    different numbers of datasets would be printed side by side as if comparable.

    A cell that never RAN (an uninstalled backend writes no cell at all) is absent from
    `cells` entirely and stays missing, so the two cases remain distinguishable:
    metadata.crash_kind records `exit -N` / `timeout > Ns`, both attributable to the
    algorithm."""
    out = set()
    for key, cell in cells.items():
        if key.suite != Suite.FRAME or not cell.get("metadata", {}).get("crashed"):
            continue
        if conditions is not None and key.condition not in conditions:
            continue
        if not _eligible(key, algo, datasets):
            continue
        out.add(key.subject)
    return out


def _crashed_synthetic(cells, algo, kind):
    """Crashed synthetic families of ONE type, so a crash is charged to the track that
    owns that family and to no other. `metadata.family_type` is written from the family
    name BEFORE the run, so it survives the crash that empties `results`; the fallback
    re-derives it for cells written without it."""
    out = set()
    for key, cell in cells.items():
        if key.suite != Suite.SYNTHETIC or key.algo != algo \
                or not cell.get("metadata", {}).get("crashed"):
            continue
        fam = key.subject
        ftype = cell.get("metadata", {}).get("family_type")
        if ftype is None:
            from synthetic_benchmark import FAMILIES, family_type
            ftype = family_type(fam) if fam in FAMILIES else None
        if ftype == kind:
            out.add(fam)
    return out


def _trajectory_group(fam):
    """Which of the two capabilities the Tracking track weighs equally a family probes:
    holding a steady tone, or following a moving one."""
    from synthetic_benchmark import FAMILIES
    return "vib" if FAMILIES.get(fam, (None,))[0] == "vib" else "steady"


def _dead_score(dead, star=None):
    """The result a track returns when it has NO readable cell. Cells that were attempted
    and died score 0 -- a tracker that dies everywhere must not outrank one that merely
    scores badly. Nothing attempted at all stays missing (None)."""
    return {"score": 0.0 if dead else None, "n_completed": 0, "n_crashed": len(dead),
            **(star or {})}


def _with_zeros(values, n_crashed):
    """Mean over completed cells plus one 0 per crashed cell."""
    vals = list(values) + [0.0] * n_crashed
    return float(np.mean(vals)) if vals else None


def _sweep_f(cell, idx):
    """The cell's unconditional pitch F1 at one threshold index."""
    return cell["results"]["sweep"][idx]["pitch"]["f1"]


def theta_star(cells, algo, *, datasets=None):
    """The algorithm's ONE frozen operating point: argmax over the threshold grid of the
    equal-per-dataset mean pitch F1 on clean frame cells. Full (uncapped) clean cells
    are preferred; if only probe-sized clean baselines exist the fallback is used and
    stamped in the provenance. Ties resolve to the lowest threshold.
    Returns {"theta", "idx", "provenance", "n_datasets"}."""
    # frame_cells already prefers the uncapped cell per dataset; the caps tell us which
    # kind of measurement the pool actually is. A crashed full cell no longer hides the
    # capped baseline, because they are separate keys rather than competitors.
    # Pool-level, not per-dataset: an equal-per-dataset mean must not average a full
    # measurement against a 30-clip probe. Full cells win outright where any exist.
    everything, full = {}, {}
    for ds, cap, cell in frame_cells(cells, algo, "clean", datasets, any_cap=True):
        everything[ds] = cell
        if cap is None:
            full[ds] = cell
    pool, provenance = (full, "clean-full") if full else (everything, "clean-probe")
    if not pool:
        return {"theta": None, "idx": None, "provenance": "no-clean-cells", "n_datasets": 0}
    # the shortest grid in the pool: cells swept over different grids would otherwise
    # make the argmax depend on which cell happened to be iterated first
    n_thr = min(len(c["results"]["thresholds"]) for c in pool.values())
    means = [float(np.mean([_sweep_f(c, i) for c in pool.values()])) for i in range(n_thr)]
    idx = int(np.argmax(means))                       # argmax returns the FIRST max: lowest theta
    theta = next(iter(pool.values()))["results"]["thresholds"][idx]
    return {"theta": float(theta), "idx": idx, "provenance": provenance,
            "n_datasets": len(pool)}


def _pooled_sums(cells, algo, *, theta_idx, datasets=None, conditions=("clean",)):
    """Summed suff stats over the selected frame cells at one threshold index.
    conditions=None pools EVERY condition except the probe baseline."""
    if conditions is None:
        conditions = sorted({k.condition for k in cells
                             if k.suite == Suite.FRAME and k.algo == algo})
    total = None
    for cond in conditions:
        for _ds, _cap, cell in frame_cells(cells, algo, cond, datasets):
            pc = cell["results"].get("per_clip")
            if not pc:
                continue
            for stats in pc["stats"]:
                # strict: a row that does not match the schema would otherwise
                # zip-truncate and silently shift every field past the difference
                row = dict(zip(FRAME_STAT_COLS, stats[theta_idx], strict=True))
                total = row if total is None else {k: total[k] + row[k] for k in total}
    return total


def factorization(cells, algo, *, theta_idx, datasets=None, conditions=("clean",)):
    """Pooled pitch precision / recall / F1 plus voicing recall, so a report can show
    where a tracker's recall comes from: pitch recall = voicing recall x (the
    conditional correct rate, which the error table already reports as exact+close)."""
    s = _pooled_sums(cells, algo, theta_idx=theta_idx, datasets=datasets,
                     conditions=conditions)
    if not s:
        return None
    p, r, f = pitch_prf(s["n_correct"], s["tp"], s["fp"], s["fn"])
    vrec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
    return {"pitch_f1": f, "pitch_recall": r, "pitch_precision": p,
            "voicing_recall": vrec}


def track_correctness(cells, algo, *, datasets=None):
    """How correct is the output curve on clean real recordings? (pitch F1)"""
    star = theta_star(cells, algo, datasets=datasets)
    if star["idx"] is None:
        return _dead_score(crashed_datasets(cells, algo, conditions={"clean"},
                                    datasets=datasets), star)
    per_ds = {}
    for ds, cap, cell in frame_cells(cells, algo, "clean", datasets):
        e = cell["results"]["sweep"][star["idx"]]
        # capped-ness comes from the KEY, not from a metadata flag: two recordings of one
        # fact can disagree
        per_ds[ds] = {"f1": e["pitch"]["f1"], "cap": cap}
    crashed = crashed_datasets(cells, algo, conditions={"clean"}, datasets=datasets) - set(per_ds)
    if not per_ds and not crashed:
        return {"score": None, **star}
    return {"score": _with_zeros([d["f1"] for d in per_ds.values()], len(crashed)),
            "per_dataset": per_ds, "n_completed": len(per_ds),
            "n_crashed": len(crashed), **star}


def track_noise(cells, algo, *, datasets=None):
    """How well does it work UNDER real-world corruption? The equal-per-dataset mean
    pitch F1 over the degraded conditions -- an ABSOLUTE score, same construction as
    track_correctness with a different condition set.

    Deliberately NOT the retention ratio F_degraded/F_clean: that ratio rises when clean
    performance falls, so sandbagging your own clean output raises the score at fixed
    degraded performance, and it is floor-effect-prone. The ratio survives as
    `per_condition_retention`, reported
    and never scored, because "how much was lost" is still worth reading."""
    star = theta_star(cells, algo, datasets=datasets)
    if star["idx"] is None:
        return _dead_score(crashed_datasets(cells, algo, datasets=datasets), star)
    conds = sorted({k.condition for k in cells
                    if k.suite == Suite.FRAME and k.algo == algo and k.condition != "clean"})
    per_cond, retention = {}, {}
    for cond in conds:
        fs, ratios = [], []
        seen = set()
        for ds, cap, cell in frame_cells(cells, algo, cond, datasets):
            seen.add(ds)
            f = _sweep_f(cell, star["idx"])
            fs.append(f)
            # the retention ratio must divide by clean measured on the SAME clips, so
            # match the degraded cell's cap before falling back to the full clean cell
            base = (frame_cell_at(cells, algo, ds, "clean", cap)
                    or frame_cell_at(cells, algo, ds, "clean", None))
            if base is not None:
                fc = _sweep_f(base, star["idx"])
                if fc > 0:
                    ratios.append(f / fc)
        n_dead = len(crashed_datasets(cells, algo, conditions={cond}, datasets=datasets) - seen)
        s = _with_zeros(fs, n_dead)
        if s is not None:
            per_cond[cond] = s
        if ratios:
            retention[cond] = float(np.mean(ratios))
    if not per_cond:
        return {"score": None, **star}
    return {"score": float(np.mean(list(per_cond.values()))),
            "per_condition": per_cond,
            "per_condition_retention": retention,   # diagnostic only, never scored
            # the denominator, so two algorithms averaged over different numbers of
            # conditions are not read as if they were comparable
            "n_conditions": len(per_cond),
            **star}


def synthetic_cells(cells, algo, kind):
    """Completed synthetic cells of one family kind, as (family, sweep-entry-getter).

    The ONE place the synthetic result envelope is opened. Everything else asks by family
    kind and gets values, so a change to the stored shape lands here and nowhere else.
    """
    for key, cell in cells.items():
        if key.suite != Suite.SYNTHETIC or key.algo != algo \
                or cell.get("metadata", {}).get("crashed"):
            continue
        res = cell.get("results", {})
        if res.get("kind") == kind and res.get("sweep"):
            yield key.subject, res["sweep"]


def stationary_families(cells, algo):
    """The synthetic stationary families this algorithm has completed cells for."""
    return sorted(fam for fam, _sweep in synthetic_cells(cells, algo, "stationary"))


def synthetic_recall(cells, algo, family, theta_idx):
    """Unconditional pitch recall for one stationary family at a threshold index, or None
    if the algorithm has no completed cell for it."""
    if theta_idx is None:
        return None
    for fam, sweep in synthetic_cells(cells, algo, "stationary"):
        if fam == family:
            return sweep[theta_idx]["pitch"]["recall"]
    return None


def cost_summary(cells, algo):
    """Reliability facts about running one algorithm: how many cells finished and how the
    rest failed. Speed is measured separately and comparably by the controlled Speed track;
    a timing taken during the accuracy runs is neither controlled nor device-uniform (cells
    auto-resolve to cpu/mps/cuda and run concurrently), so it is not reported here."""
    attempted = completed = 0
    kinds = {}
    for key, cell in cells.items():
        if key.algo != algo:
            continue
        attempted += 1
        m = cell.get("metadata", {})
        if m.get("crashed"):
            k = m.get("crash_kind") or m.get("error") or "unknown"   # "error": pre-rename cells
            kinds[k] = kinds.get(k, 0) + 1
            continue
        completed += 1
    return {"attempted": attempted, "completed": completed, "crash_kinds": kinds}


def track_ci(cells, algo, track, *, datasets=None, n_boot=2000):
    """95% interval for a track score, or None where the resampling unit is unsound.

    Only `correctness` and `noise` have one: their unit is the CLIP, resampled within each
    dataset and averaged across datasets per replicate -- the same equal-per-dataset
    weighting the score itself uses. Signal types (~24 families), Tracking (5) and Notes
    (3 datasets) would be resampling a handful of units; Speed is a single measurement.
    An absent interval is honest; a wide unstable one invites false confidence.

    A crashed dataset contributes 0 to EVERY draw, because it contributes 0 to the score.
    Without that the interval would bracket a number the score never reports: a low
    absolute score paired with a completed-only interval sitting far above it.
    """
    if track not in ("correctness", "noise"):
        return None
    star = theta_star(cells, algo, datasets=datasets)
    if star["idx"] is None:
        return None
    conditions = ("clean",) if track == "correctness" else tuple(
        sorted({k.condition for k in cells if k.suite == Suite.FRAME and k.algo == algo
                and k.condition != "clean"}))
    # Mean WITHIN each condition, then across conditions -- exactly how track_noise
    # weights its score. A flat mean over all (condition, dataset) cells silently
    # reweights by how many datasets each condition happens to have, and produced an
    # interval that did not contain the score it was bracketing.
    by_cond, n_seen = {}, 0
    for cond in conditions:
        draws, seen = [], set()
        for ds, _cap, cell in frame_cells(cells, algo, cond, datasets):
            pc = cell["results"].get("per_clip")
            if not (pc and pc.get("stats")):
                continue
            seen.add(ds)
            keyed, _n = frame_keyed(pc, star["idx"])
            draws.append(boot_vals(list(keyed.values()), FRAME_REDUCERS["pitch_f1"],
                                   n_boot, seed=n_seen + len(draws)))
        dead = crashed_datasets(cells, algo, conditions={cond}, datasets=datasets) - seen
        draws += [np.zeros(n_boot)] * len(dead)
        if draws:
            n_seen += len(draws)
            by_cond[cond] = np.mean(draws, axis=0)
    if not by_cond:
        return None
    per_replicate = np.mean(list(by_cond.values()), axis=0)
    return float(np.percentile(per_replicate, 2.5)), float(np.percentile(per_replicate, 97.5))


def track_signal_types(cells, algo, *, datasets=None):
    """How much of the synthetic signal-class space does it handle? Mean over the
    stationary families of pitch RECALL at theta* (unconditional, so refusing to
    voice a family scores low rather than vanishing); each
    family is one probe question, equally weighted. The worst family is carried as the
    named diagnostic rather than scored as a min, because most real trackers have at least
    one dead family and a min would zero them all out, destroying downstream ranking."""
    star = theta_star(cells, algo, datasets=datasets)
    dead = _crashed_synthetic(cells, algo, "stationary")
    if star["idx"] is None:
        return _dead_score(dead, star)
    fams = {fam: sweep[star["idx"]]["pitch"]["recall"]
            for fam, sweep in synthetic_cells(cells, algo, "stationary")}
    dead -= set(fams)
    if not fams and not dead:
        return {"score": None, **star}
    scored = dict(fams, **{f: 0.0 for f in dead})     # a crash IS a failed family
    worst = min(scored, key=scored.get)
    return {"score": float(np.mean(list(scored.values()))), "worst_family": worst,
            "worst": float(scored[worst]), "per_family": scored,
            "n_completed": len(fams), "n_crashed": len(dead), **star}


def track_tracking(cells, algo, *, datasets=None):
    """Is a moving pitch followed faithfully? Steady tones (jitter, bias) + vibrato
    (depth retention x voiced fraction); the two families the pilot showed discriminate."""
    star = theta_star(cells, algo, datasets=datasets)
    dead = _crashed_synthetic(cells, algo, "trajectory")
    if star["idx"] is None:
        return _dead_score(dead, star)
    steady, vib = [], []
    detail = {}
    for key, cell in cells.items():
        fam = key.subject
        if key.suite != Suite.SYNTHETIC or key.algo != algo \
                or cell.get("metadata", {}).get("crashed"):
            continue
        res = cell.get("results", {})
        if res.get("kind") != "trajectory" or not res.get("sweep"):
            continue
        e = res["sweep"][star["idx"]]
        detail[fam] = e
        # A readout is None when the tracker voiced too few frames of that stimulus to
        # measure it. That is a FAILURE on this track, not missing data: score it 0
        # rather than skipping, or a tracker that declines the hard stimuli would be
        # rewarded with a higher mean over the ones it did attempt.
        if "jitter_cents" in e:
            j, b = e["jitter_cents"], e.get("bias_cents")
            # TOTAL error, not de-biased jitter: scoring jitter alone lets a tracker that
            # emits a CONSTANT (even an octave wrong) score perfectly, beating an honest
            # tracker with a little noise. Bias is folded in (hypot), not discarded.
            err = None if j is None else float(np.hypot(j, b or 0.0))
            steady.append(SIGMA0 / (SIGMA0 + err) if err is not None else 0.0)
        if "depth_ratio" in e:
            d = e["depth_ratio"]
            vf = e.get("voiced_fraction") or 0.0
            vib.append(np.clip(d, 0.0, 1.0) * vf if d is not None else 0.0)
    # A crashed family is a 0 inside ITS OWN group, not a 0 alongside the group means:
    # `parts` holds at most two entries, so charging a dead family there would give one
    # stimulus the weight of an entire capability.
    dead -= set(detail)
    for fam in dead:
        (vib if _trajectory_group(fam) == "vib" else steady).append(0.0)
    parts = [float(np.mean(x)) for x in (steady, vib) if x]
    if not parts:
        return _dead_score(dead, star)
    return {"score": float(np.mean(parts)), "per_family": detail,
            "n_completed": len(detail), "n_crashed": len(dead), **star}


def track_notes(cells, algo):
    """Is musical note structure recoverable? Mean COnP over note datasets (this track selects
    its own threshold x segmentation-cost internally, documented). Cap-aware like frame_cells:
    per dataset the UNCAPPED note cell is preferred over a capped probe, so a probe never stands
    in for -- nor masks a crash in -- the full run."""
    best = {}          # ds -> (cap, conp | None, crashed)
    for key, cell in cells.items():
        if key.suite != Suite.NOTE or key.algo != algo:
            continue
        crashed = bool(cell.get("metadata", {}).get("crashed"))
        v = None if crashed else (cell.get("results", {}) or {}).get("conp")
        if not crashed and (v is None or not np.isfinite(v)):
            continue
        prev = best.get(key.subject)
        if prev is None or (prev[0] is not None and key.cap is None):   # prefer uncapped
            best[key.subject] = (key.cap, v, crashed)
    vals = {ds: v for ds, (_c, v, crashed) in best.items() if not crashed}
    dead = {ds for ds, (_c, _v, crashed) in best.items() if crashed}
    if not vals and not dead:
        return {"score": None}
    return {"score": _with_zeros(list(vals.values()), len(dead)),
            "per_dataset": vals, "n_completed": len(vals), "n_crashed": len(dead)}


def track_speed(cells, algo):
    """Is it deployable? score = 1/(1+RTF_cpu)."""
    crashed = False
    for key, cell in cells.items():
        if key.suite != Suite.SPEED or key.algo != algo:
            continue
        if cell.get("metadata", {}).get("crashed"):
            crashed = True                 # keep looking: a good cell outranks a dead one
            continue
        dev = cell.get("results", {}).get("device_results", {}).get("cpu", {})
        ms = dev.get("absolute_time_ms")
        if ms is None:
            return {"score": None}
        sec = cell.get("parameters", {}).get("signal_length_seconds") or 1.0
        rtf = (ms / 1000.0) / sec
        return {"score": 1.0 / (1.0 + rtf), "rtf_cpu": rtf}
    if crashed:
        return {"score": 0.0, "n_completed": 0, "n_crashed": 1}
    return {"score": None}


def track_scores(cells, algo, *, datasets=None, intervals=True):
    """The six track scores as a `Tracks`, each a `Score` carrying its interval where one
    is sound. `intervals=False` skips the bootstrap when only the point estimates are
    wanted (the report renders many algorithms)."""
    raw = {
        "correctness": track_correctness(cells, algo, datasets=datasets)["score"],
        "noise": track_noise(cells, algo, datasets=datasets)["score"],
        "signal_types": track_signal_types(cells, algo, datasets=datasets)["score"],
        "tracking": track_tracking(cells, algo, datasets=datasets)["score"],
        "notes": track_notes(cells, algo)["score"],
        "speed": track_speed(cells, algo)["score"],
    }
    out = {}
    for name, value in raw.items():
        ci = track_ci(cells, algo, name, datasets=datasets) if (
            intervals and value is not None) else None
        out[name] = Score(value, *(ci or (None, None)))
    return Tracks(**out)


def harmonic_mean(values):
    """The one mean used for the Overall. Zero-propagating: a zero anywhere makes it 0,
    which is what "the weakest axis dominates" has to mean. Dropping zeros instead would
    make the mean ANTI-monotone in failure, since what survives the drop are exactly the
    tracker's best axes."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0 or (v <= 0).any():
        return 0.0
    return float(v.size / np.sum(1.0 / v))


def rank_key(overall_value, track_values):
    """The one ordering for every leaderboard: measured before unmeasured, Overall
    descending, then -- among the trackers a failed axis has pinned to 0 -- the harmonic
    mean of the tracks that did NOT fail, so failing one axis outranks failing everywhere."""
    survivors = [x for x in track_values if x]
    return (overall_value is None, -(overall_value or 0.0),
            -harmonic_mean(survivors) if survivors else 0.0)


def overall(cells, algo, *, datasets=None):
    """Harmonic mean of the six track scores, equal weights: the same mean family as
    the F-scores inside the tracks, and the one that punishes a weak axis hardest (HM
    is dominated by the smallest score). None if ANY track is missing (never a silent
    partial mean); a zero track makes it 0 (a failed axis sinks it)."""
    scores = track_scores(cells, algo, datasets=datasets, intervals=False)
    vals = scores.values()
    if any(v is None for v in vals):
        return None
    if any(v <= 0 for v in vals):
        return 0.0
    return harmonic_mean(vals)


# --------------------------------------------------------------------------- #
# Derived read-outs the report renders. They compute NUMBERS, so they live here
# (testable, and reachable by anything that imports the scoring layer) rather than
# inside the presentation layer.
# --------------------------------------------------------------------------- #
def error_classes(cells, algo, *, theta_idx, datasets=None):
    """The four disjoint magnitude classes over every scored frame, pooled across all
    conditions, plus the two cents summaries. exact+close+off+wrong == 1 by construction."""
    t = _pooled_sums(cells, algo, theta_idx=theta_idx, datasets=datasets, conditions=None)
    if not t or not t["frames"]:
        return None
    v = t["frames"]
    rates = error_rates(t)
    return {"exact": t["n_exact"] / v,
            "close": (t["n_correct"] - t["n_exact"]) / v,
            "off": (v - t["n_correct"] - t["n_wrong"]) / v,
            "wrong": t["n_wrong"] / v,
            "cents_mae": rates["cents_mae"], "cents_bias": rates["cents_bias"],
            "frames": v}


def band_aggregate(cells, algo, *, theta_idx, datasets=None):
    """Frame-weighted correct rate per ground-truth pitch band: catches a tracker that is
    fine overall and broken in one register. Frame counts come with it, so an empty band
    is distinguishable from a failed one."""
    agg = {b: {"frames": 0.0, "correct": 0.0} for b, _, _ in PITCH_BANDS}
    conds = sorted({k.condition for k in cells
                    if k.suite == Suite.FRAME and k.algo == algo})
    for cond in conds:
        for _ds, _cap, cell in frame_cells(cells, algo, cond, datasets):
            pb = (cell["results"]["sweep"][theta_idx] or {}).get("per_band") or {}
            for b, _lo, _hi in PITCH_BANDS:
                d = pb.get(b) or {}
                n = d.get("frames") or 0
                if n and d.get("correct_rate") is not None:
                    agg[b]["frames"] += n
                    agg[b]["correct"] += d["correct_rate"] * n
    return {b: (d["correct"] / d["frames"], int(d["frames"])) if d["frames"] else (None, 0)
            for b, d in agg.items()}


def ties(cells, *, datasets=None):
    """Per clean dataset: the best pitch F1 and every algorithm statistically tied with it.
    A leaderboard prints 0.878 next to 0.862 as though the gap were real; a paired cluster
    bootstrap over the clips both algorithms scored says whether it is."""
    per_ds = {}
    for algo in sorted({k.algo for k in cells}):
        star = theta_star(cells, algo, datasets=datasets)
        if star["idx"] is None:
            continue
        for ds, _cap, cell in frame_cells(cells, algo, "clean", datasets):
            pc = cell["results"].get("per_clip")
            if not (pc and pc.get("stats")):
                continue
            per_ds.setdefault(ds, {})[algo] = (
                _sweep_f(cell, star["idx"]), frame_keyed(pc, star["idx"])[0])
    out = {}
    for ds, entry in per_ds.items():
        if len(entry) < 2:
            continue
        best = max(entry, key=lambda a: entry[a][0])
        out[ds] = {"best": best, "f1": entry[best][0],
                   "tied": [a for a in sorted(entry) if a != best
                            and paired_tied(entry[a][1], entry[best][1],
                                            FRAME_REDUCERS["pitch_f1"])]}
    return out


def noise_drop_ci(cells, algo, *, datasets=None):
    """Paired CI on how much pitch F1 the degradations cost, in percentage points. Each
    (dataset, condition) contributes one clean/degraded pair over the clips both sides
    scored, so shared clip difficulty cancels."""
    star = theta_star(cells, algo, datasets=datasets)
    if star["idx"] is None:
        return float("nan"), float("nan"), 0
    pairs = []
    for key, cell in cells.items():
        if key.suite != Suite.FRAME or key.condition == "clean" or not _eligible(key, algo, datasets):
            continue
        # pair against the clean baseline at the SAME cap: literally the same clips,
        # which is what "paired" means. Fall back to the full cell only if there is none.
        clean = (frame_cell_at(cells, algo, key.subject, "clean", key.cap)
                 or frame_cell_at(cells, algo, key.subject, "clean", None))
        pc_d = (cell.get("results") or {}).get("per_clip")
        pc_c = ((clean or {}).get("results") or {}).get("per_clip")
        if not (pc_d and pc_d.get("stats") and pc_c and pc_c.get("stats")):
            continue
        pairs.append((frame_keyed(pc_c, star["idx"])[0],
                      frame_keyed(pc_d, star["idx"])[0]))
    return paired_delta_ci(pairs) if pairs else (float("nan"), float("nan"), 0)


def has_fixed_operating_point(cells, algo):
    """True when the sweep is flat on every clean cell: the tracker exposes no usable
    voicing confidence, so theta* is a formality rather than a tuned choice."""
    seen = False
    for _ds, _cap, cell in frame_cells(cells, algo, "clean"):
        f = [e["pitch"]["f1"] for e in cell["results"]["sweep"]]
        if not f:
            continue
        seen = True
        if max(f) - min(f) > 1e-9:
            return False
    return seen
