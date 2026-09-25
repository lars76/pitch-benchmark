import json
import math
import os
import zlib
from collections import Counter
from glob import glob
from itertools import combinations

import numpy as np

VOICED_THRESHOLD = 0.5

CORRECT_CENTS = 50
LADDER_CENTS = (10, 25, CORRECT_CENTS, 200)
OCTAVE_BRACKET_CENTS = (1100, 1300)
STAT_COLS = ("tp", "fp", "fn", "tp_trusted", *(f"n_lt_{c}" for c in LADDER_CENTS),
             "n_oct_up", "n_oct_down", "n_ref_trusted")
F1_COLS = (f"n_lt_{CORRECT_CENTS}", "fp", "tp_trusted", "n_ref_trusted")
F1_COL_IDX = tuple(STAT_COLS.index(c) for c in F1_COLS)
THRESHOLDS = np.linspace(0.0, 1.0, 41)
N_BOOT = 20000
_SE_FLOOR = 1e-12
BOOT_SEED = 20260826
BOOT_SCORED, BOOT_CAL = 2, 3


class Design:

    def __init__(self, identity, factors, carrier, panels):
        self.identity = str(identity)
        self.factors = tuple(factors)
        self.carrier = tuple(carrier)
        self.stages = {p: tuple(v) for p, v in panels.items()}
        if self.identity not in self.stages:
            raise ValueError(f"the design names {self.identity!r} as its reference panel but "
                             f"declares {sorted(self.stages)}")
        self.scored = tuple(p for p in self.stages if p != self.identity)
        self.panel_factors = {p: tuple(s for s in self.stages[p] if s in self.factors)
                              for p in self.scored}
        self.labels = {p: "+".join(f) or self.carrier[0]
                       for p, f in self.panel_factors.items()}
        self.effects = tuple(s for r in range(1, len(self.factors) + 1)
                             for s in combinations(self.factors, r))


def load_run(cells_dir):
    with open(os.path.join(cells_dir, "run.json")) as f:
        return json.load(f)


def is_voiced(periodicity):
    return periodicity >= VOICED_THRESHOLD


def cents(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(a / b)


def corpus_uid(name):
    return zlib.crc32(str(name).encode())


def columns(sums, names=STAT_COLS):
    return dict(zip(names, np.moveaxis(np.asarray(sums, dtype=np.float64), -1, 0)))


def pitch_f1(sums, names=STAT_COLS):
    cols = columns(sums, names)
    denom = cols["tp_trusted"] + cols["fp"] + cols["n_ref_trusted"]
    return np.divide(2.0 * cols[f"n_lt_{CORRECT_CENTS}"], denom,
                     out=np.zeros_like(denom), where=denom > 0)


def voicing_f1(sums):
    cols = columns(sums)
    tp, fp, fn = cols["tp"], cols["fp"], cols["fn"]
    denom = (tp + fp) + (tp + fn)
    return np.divide(2.0 * tp, denom, out=np.zeros_like(denom), where=denom > 0)


def in_scope_mask(true_pitch, auditable, fmin=None, fmax=None):
    p = np.asarray(true_pitch)
    if fmin is None and fmax is None:
        return np.ones(p.shape, dtype=bool)
    if fmin is None or fmax is None:
        raise ValueError(f"in_scope_mask needs both bounds or neither, got "
                         f"fmin={fmin!r}, fmax={fmax!r}")
    return ~(auditable & ((p < fmin) | (p > fmax)))


def label_masks(true_pitch, trusted=None, fmin=None, fmax=None):
    voiced = np.asarray(true_pitch) > 0
    auditable = voiced if trusted is None else voiced & np.asarray(trusted, dtype=bool)
    return voiced, auditable, in_scope_mask(true_pitch, auditable, fmin, fmax)


def sweep_clip_stats(results, true_pitch, trusted=None, fmin=None, fmax=None):
    true_pitch = np.asarray(true_pitch)
    tv, auditable, scope = label_masks(true_pitch, trusted, fmin, fmax)
    n_ref_trusted = int(np.sum(auditable & scope))
    rows = []
    for pred_pitch, pred_voicing in results:
        pred_pitch = np.asarray(pred_pitch)
        pv = np.asarray(pred_voicing).astype(bool) & (pred_pitch > 0)
        tp = int(np.sum(pv & tv))
        fp = int(np.sum(pv & ~tv))
        fn = int(np.sum(~pv & tv))
        mask = pv & auditable & scope
        signed = cents(pred_pitch[mask], true_pitch[mask])
        signed = signed[np.isfinite(signed)]
        abs_cents = np.abs(signed)
        lo, hi = OCTAVE_BRACKET_CENTS
        rows.append([tp, fp, fn, int(np.sum(mask))]
                    + [int(np.sum(abs_cents < c)) for c in LADDER_CENTS]
                    + [int(np.sum((signed >= lo) & (signed <= hi))),
                       int(np.sum((signed <= -lo) & (signed >= -hi))),
                       n_ref_trusted])
    return rows


def crashed_clip_stats(n_voiced, n_ref_trusted):
    row = dict.fromkeys(STAT_COLS, 0)
    row["fn"], row["n_ref_trusted"] = int(n_voiced), int(n_ref_trusted)
    return [[row[c] for c in STAT_COLS] for _ in range(len(THRESHOLDS))]


def json_key(k):
    if isinstance(k, (str, int, float, bool)) or k is None:
        return k
    return str(k)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {json_key(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(to_json_safe(obj), f, separators=(",", ":"))
    os.replace(tmp, path)


class Cell:

    def __init__(self, raw):
        m, r = raw["metadata"], raw["results"]
        self.dataset, self.panel, self.algo = m["dataset_name"], m["panel"], m["algorithm_name"]
        self.crashed, self.crash_kind = bool(m["crashed"]), m.get("crash_kind")
        self.clips = list(r["clips"])
        self.clip_crashed = list(r["crashed"])
        self.crash_kinds = list(r["crash_kinds"])
        self.stats = np.asarray(r["stats"], dtype=np.int64).reshape(
            -1, len(THRESHOLDS), len(STAT_COLS))
        roles = np.asarray(r["roles"])
        self.scored = np.flatnonzero(roles == "test")
        self.cal = np.flatnonzero(roles == "valid")

    def cal_keys(self):
        return [self.clips[i] for i in self.cal]

    def sums(self, picks):
        if not len(picks):
            return np.zeros((len(THRESHOLDS), len(STAT_COLS)), dtype=np.int64)
        return self.stats[picks].sum(axis=0)


def load_speed_cells(cells_dir):
    out = []
    for name in sorted(os.listdir(cells_dir)):
        if not (name.startswith("speed_") and name.endswith(".json")):
            continue
        with open(os.path.join(cells_dir, name)) as f:
            out.append(json.load(f))
    return out


def speed_factor(cell):
    results = cell["results"]
    if results["error"] or len(results["run_ms"]) < cell["parameters"]["rounds"]:
        return None
    return cell["parameters"]["signal_seconds"] * 1e3 / float(np.median(cell["results"]["run_ms"]))


def speed_sd(cell):
    return float(np.std(cell["parameters"]["signal_seconds"] * 1e3 / np.array(cell["results"]["run_ms"]), ddof=1))


def cpu_per_wall(cell):
    results = cell["results"]
    return float(np.median(np.array(results["run_ms"]) / np.array(results["wall_ms"])))


def load_cells(cells_dir):
    cells = []
    for path in sorted(glob(os.path.join(cells_dir, "*", "*", "*.json"))):
        with open(path) as f:
            cells.append(json.load(f))
    return cells


def index_cells(cells):
    if not cells:
        raise ValueError("no cells: there is nothing to score. A report built from an empty "
                         "matrix is not a weaker benchmark, it is not a benchmark -- run run.py "
                         "first, or point --cells at the directory that has the results.")
    parsed = [Cell(c) for c in cells]
    idx = {(c.dataset, c.panel, c.algo): c for c in parsed}
    check_pairing(idx)
    return idx


def check_pairing(idx):
    for corpus in sorted({c.dataset for c in idx.values()}):
        first = {}
        for (dataset, panel, algo), cell in sorted(idx.items()):
            if dataset != corpus or cell.crashed:
                continue
            for role, picks in (("test", cell.scored), ("valid", cell.cal)):
                keys = [cell.clips[i] for i in picks]
                if not keys:
                    continue
                seen = first.setdefault(role, (panel, algo, keys))
                if keys == seen[2]:
                    continue
                same = set(keys) == set(seen[2])
                raise ValueError(
                    f"{corpus}: {panel}/{algo} and {seen[0]}/{seen[1]} disagree on which clips "
                    f"are {role} ("
                    + (f"same {len(keys)} clips in a different order" if same
                       else f"{len(keys)} vs {len(seen[2])} clips")
                    + "). One bootstrap draw per corpus is applied to every panel and every "
                    "tracker under common random numbers, so pairing is by position: a "
                    "disagreement here makes the intervals and every pairwise comparison wrong "
                    "without making them look wrong. Re-measure into one directory from one "
                    "dataset.")


def _boot_rng(dataset, stream):
    return np.random.default_rng(np.random.SeedSequence([BOOT_SEED, stream, corpus_uid(dataset)]))


def calibration_sets(idx, algo, design):
    sets, dropped = [], {}
    for ds in sorted({d for (d, _p, a) in idx if a == algo}):
        cells = [idx.get((ds, p, algo)) for p in design.scored]
        bad = [p for p, c in zip(design.scored, cells) if c is None or c.crashed]
        if bad:
            dropped[ds] = bad
            continue
        if not cells[0].cal_keys():
            continue
        sets.append((ds, cells))
    return sets, dropped


def _calibration_curve(unit, design):
    return np.mean([pitch_f1(unit[p], F1_COLS) for p in design.scored], axis=0)


def select_threshold(idx, algo, design, sets):
    have = {}
    for (ds, panel, a) in idx:
        if a == algo and panel in design.scored:
            have.setdefault(ds, set()).add(panel)
    missing = sorted(ds for ds in {d for (d, _p, a) in idx if a == algo}
                     if set(design.scored) - have.get(ds, set()))
    if missing:
        raise KeyError(f"missing scored-panel cell for algo={algo} on {missing} -- "
                       f"theta cannot be selected on a partial matrix")
    units = [dict(zip(design.scored, (c.sums(c.cal)[:, F1_COL_IDX] for c in cells)))
             for _ds, cells in sets]
    if not units:
        return {"theta": None, "idx": None, "n_units": 0}
    i = int(np.argmax(np.mean([_calibration_curve(u, design) for u in units], axis=0)))
    return {"theta": float(THRESHOLDS[i]), "idx": i, "n_units": len(units)}


def panel_score(idx, algo, panel, theta_idx, datasets, metric=pitch_f1):
    vals = []
    for ds in datasets:
        cell = idx.get((ds, panel, algo))
        if cell is None:
            raise KeyError(f"missing cell: dataset={ds} panel={panel} algo={algo}")
        if not cell.crashed and not len(cell.scored):
            raise ValueError(
                f"{ds}/{panel}/{algo}: the cell exists but scored no clips. That is a missing "
                "measurement, not a zero, and averaging it in would quietly halve the score.")
        vals.append(float("nan") if cell.crashed
                    else float(metric(cell.sums(cell.scored))[theta_idx]))
    return float(np.mean(vals))


def headline(panel_scores, design):
    return float(np.mean([panel_scores[p] for p in design.scored]))


def ladder_profile(idx, algo, theta_idx, datasets, design):
    total = sum((cell.sums(cell.scored)[theta_idx].astype(float)
                 for ds in datasets for panel in design.scored
                 if (cell := idx.get((ds, panel, algo))) is not None and not cell.crashed),
                np.zeros(len(STAT_COLS)))
    cols = columns(total)
    trusted = float(cols["tp_trusted"])
    if trusted <= 0:
        return None
    rungs = {f"<{c}c": float(cols[f"n_lt_{c}"]) / trusted for c in LADDER_CENTS}
    return {"rungs": rungs, "tp_trusted": trusted,
            "octave_up": float(cols["n_oct_up"]) / trusted,
            "octave_down": float(cols["n_oct_down"]) / trusted}


def stage_effects(panel_scores, design):
    out = {}
    for subset in design.effects:
        total = 0.0
        for p, active in design.panel_factors.items():
            sign = 1
            for f in subset:
                sign = sign if f in active else -sign
            total += sign * panel_scores[p]
        out[subset] = float(total) / 2 ** (len(design.factors) - len(subset))
    return out


def _resample_sums(arr, picks):
    n_boot, k = picks.shape
    flat = picks + (np.arange(n_boot, dtype=np.int64) * k)[:, None]
    counts = np.bincount(flat.ravel(), minlength=n_boot * k).reshape(n_boot, k).astype(np.float64)
    return (counts @ arr.reshape(k, -1)).reshape(n_boot, *arr.shape[1:])


def _resample_matrix(cell, picks_idx):
    return cell.stats[picks_idx][..., F1_COL_IDX].astype(np.float64)


def replicate_scores(idx, algo, datasets, n_boot, design, sets):
    units = []
    for ds, cal_cells in sets:
        if ds not in datasets:
            continue
        arrs = [_resample_matrix(c, c.cal) for c in cal_cells]
        picks = _boot_rng(ds, BOOT_CAL).integers(0, len(arrs[0]), (n_boot, len(arrs[0])))
        units.append({p: _resample_sums(a, picks) for p, a in zip(design.scored, arrs)})
    if not units:
        return None
    theta_r = np.argmax(np.mean([_calibration_curve(u, design) for u in units], axis=0), axis=1)

    rows = np.arange(n_boot)
    per_panel = []
    for panel in design.scored:
        per_ds = []
        for ds in datasets:
            cell = idx.get((ds, panel, algo))
            if cell is None:
                raise KeyError(f"missing cell: dataset={ds} panel={panel} algo={algo}")
            if cell.crashed or not len(cell.scored):
                per_ds.append(np.full((n_boot, len(THRESHOLDS)), np.nan))
                continue
            picks = _boot_rng(ds, BOOT_SCORED).integers(
                0, len(cell.scored), (n_boot, len(cell.scored)))
            resampled = _resample_sums(_resample_matrix(cell, cell.scored), picks)
            per_ds.append(pitch_f1(resampled, F1_COLS))
        per_panel.append(np.mean(per_ds, axis=0)[rows, theta_r])
    return {"reps": np.mean(per_panel, axis=0), "theta_idx": theta_r}


def dominance(ordered_algos, scores, replicates, alpha=0.05):
    pairs = [(a, b) for i, a in enumerate(ordered_algos) for b in ordered_algos[i + 1:]]
    if not pairs:
        return {"crit": float("nan"), "sep": {}, "n_pairs": 0}
    d_hat, se, t = {}, {}, []
    for a, b in pairs:
        d = replicates[a] - replicates[b]
        s = float(np.std(d, ddof=1))
        d_hat[(a, b)] = float(scores[a] - scores[b])
        se[(a, b)] = s
        t.append((d - d.mean()) / (s + _SE_FLOOR))
    crit = float(np.percentile(np.max(np.abs(np.stack(t)), axis=0), 100 * (1 - alpha)))
    sep = {p: se[p] > _SE_FLOOR
              and not (d_hat[p] - crit * se[p] <= 0.0 <= d_hat[p] + crit * se[p])
           for p in pairs}
    return {"crit": crit, "sep": sep, "n_pairs": len(pairs)}


def crash_stats(idx, algo):
    out = {"cells_crashed": 0, "cells": 0, "clips_crashed": 0, "clips": 0,
           "kinds": Counter()}
    for cell in idx.values():
        if cell.algo != algo:
            continue
        out["cells"] += 1
        if cell.crashed:
            out["cells_crashed"] += 1
            out["kinds"][cell.crash_kind or "unknown"] += 1
            continue
        out["clips"] += len(cell.clip_crashed)
        out["clips_crashed"] += sum(1 for f in cell.clip_crashed if f)
        out["kinds"].update(k for k in cell.crash_kinds if k)
    out["kinds"] = dict(out["kinds"])
    return out


def score_all(cells, design, n_boot=N_BOOT):
    idx = index_cells(cells)
    datasets = sorted({k[0] for k in idx})
    algos = sorted({k[2] for k in idx})

    nan = float("nan")
    out, reps = {}, {}
    for algo in algos:
        crash = crash_stats(idx, algo)
        sets, dropped = calibration_sets(idx, algo, design)
        star = select_threshold(idx, algo, design, sets)
        if star["idx"] is None:
            out[algo] = {"theta": None, "theta_idx": None, "n_units": 0,
                         "score": nan, "panels": dict.fromkeys(design.scored, nan),
                         "identity": nan,
                         "effects": stage_effects(dict.fromkeys(design.scored, nan), design),
                         "ladder": None, "crash": crash,
                         "cal_dropped": dropped, "complete": False}
            continue
        ti = star["idx"]
        panels = {p: panel_score(idx, algo, p, ti, datasets) for p in design.scored}
        boot = replicate_scores(idx, algo, datasets, n_boot, design, sets)
        out[algo] = {
            "theta": star["theta"], "theta_idx": ti, "n_units": star["n_units"],
            "score": headline(panels, design), "panels": panels,
            "identity": panel_score(idx, algo, design.identity, ti, datasets),
            "effects": stage_effects(panels, design),
            "ladder": ladder_profile(idx, algo, ti, datasets, design),
            "voicing_f1": headline({p: panel_score(idx, algo, p, ti, datasets, metric=voicing_f1)
                                    for p in design.scored}, design),
            "crash": crash, "cal_dropped": dropped,
            "complete": crash["cells_crashed"] == 0,
        }
        if boot is not None:
            reps[algo] = boot["reps"]
            out[algo]["ci"] = (float(np.percentile(boot["reps"], 2.5)),
                               float(np.percentile(boot["reps"], 97.5)))
    ordered = sorted((a for a in algos
                      if out[a]["complete"] and np.isfinite(out[a]["score"])),
                     key=lambda a: -out[a]["score"])
    incomplete = sorted((a for a in algos if a not in ordered),
                        key=lambda a: (-out[a]["crash"]["cells_crashed"], a))
    ranked = [a for a in ordered if a in reps]
    dom = dominance(ranked, {a: out[a]["score"] for a in ranked}, reps)
    for a in ranked:
        beaten = sum(1 for b in ranked if dom["sep"].get((a, b)))
        loses = sum(1 for b in ranked if dom["sep"].get((b, a)))
        out[a]["dominance"] = {"beats": beaten, "loses_to": loses,
                               "undetermined": len(ranked) - 1 - beaten - loses}
    return {"algos": out, "order": ordered, "incomplete": incomplete, "design": design,
            "datasets": datasets, "dominance": dom, "reps": reps, "n_boot": n_boot}
