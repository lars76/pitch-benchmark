#!/usr/bin/env python3
"""The v2 report: one-glance leaderboard (Overall + seven question-tracks), then one
section per track with its diagnostics beneath. Renders whatever cells exist; a missing
track is shown as a gap, never silently averaged over.

    .venv/bin/python generate_report.py --results results --out benchmark_report.md
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import metrics
from datasets.augment import CONDITION_FAMILIES
from evaluate import load_cells
from metrics import (
    FRAME_REDUCERS,
    PITCH_BANDS,
    SIGMA0,
    T_MAX,
    TRACKS_SCORED,
    band_label,
    ci_cell,
    cluster_bootstrap,
)

# Datasets whose pitch/note ground truth is SCORE-GRADE (the notated note, not the performed
# f0): reliable for VOICING, but their pitch scores are a GT artifact and must not enter the
# accuracy leaderboard.
VOICING_ONLY = frozenset({"M4Singer"})
# Sparse-voiced corpora: long clinical/lab sessions that are mostly UNVOICED (~13-21% voiced);
# precision-type denominators are dominated by silence false positives and are not comparable
# to the dense corpora. Flagged in place, never dropped.
SPARSE_VOICED = frozenset({"OSFGlottis", "AVID"})

_EGG_SPEECH = ["PTDB", "MOCHA", "CMUArctic", "AVID", "OSFGlottis", "SVD", "APLAWD", "KEELE", "FDA"]
DATASET_GROUPS = {
    "By Origin": {
        "Synthetic": ["Bach10Synth", "MDBStemSynth", "SpeechSynth", "NSynth"],
        "Real": ["MIR1K", "Vocadito", "URMP", *_EGG_SPEECH],
    },
    "By Domain": {
        "Speech": ["SpeechSynth", *_EGG_SPEECH],
        "Music": ["Bach10Synth", "MDBStemSynth", "NSynth", "Vocadito", "MIR1K", "URMP"],
    },
    "By Cross-Dimension": {
        "Synthetic + Speech": ["SpeechSynth"],
        "Synthetic + Music": ["Bach10Synth", "MDBStemSynth", "NSynth"],
        "Real + Speech": _EGG_SPEECH,
        "Real + Music": ["Vocadito", "MIR1K", "URMP"],
    },
}

TRACK_LABELS = [("accuracy", "Accuracy"), ("noise", "Noise"), ("signals", "Signals"),
                ("stability", "Stability"), ("dynamics", "Dynamics"), ("notes", "Notes"),
                ("speed", "Speed")]


def _is_bad(v):
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "---|" * len(headers) + "\n"
    for row in rows:
        out += "| " + " | ".join(row) + " |\n"
    return out


def fmt_best(value, column_values, fmt="{:.3f}", lower=False, na="n/a"):
    """Format `value`, bolding it if it is the best (max, or min if lower) of column_values."""
    if _is_bad(value):
        return na
    valid = [v for v in column_values if not _is_bad(v)]
    best = (min if lower else max)(valid) if valid else None
    s = fmt.format(value)
    if best is not None and abs(value - best) < 1e-9:
        s = f"**{s}**"
    return s


def _quarantine_corrupt(file_path):
    """Rename a corrupt result to <name>.corrupt and warn loudly, so the runner's
    exists()-based skip regenerates the cell instead of skipping a broken one forever."""
    q = str(file_path) + ".corrupt"
    try:
        os.replace(file_path, q)
    except OSError:
        q = file_path
    print(f"Warning: corrupt result JSON quarantined -> {q}")


def load_all_results(results_dir):
    """All JSON result files split into (frame, speed, synthetic) by benchmark_type.
    Kept for the dedup utilities and tests; the report itself scores via load_cells."""
    frame, speed, synth = [], [], []
    for file_path in Path(results_dir).glob("*.json"):
        try:
            with open(file_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _quarantine_corrupt(file_path)
            continue
        except OSError as e:
            print(f"Warning: skipping {file_path}: {e}")
            continue
        m = data.get("metadata", {})
        if m.get("track") == "notes":
            continue
        bt = m.get("benchmark_type")
        if bt == "speed":
            speed.append(data)
        elif bt == "synthetic":
            synth.append(data)
        elif m.get("dataset_name"):
            frame.append(data)
    return frame, speed, synth


def _dedupe_prefer_cpu(results, key_fn):
    """One result per key: non-crashed beats crashed, then cpu (the reproducible
    reference) beats gpu. Device lives in the filename so variants coexist on disk."""
    def rank(r):
        m = r.get("metadata", {})
        return (not m.get("crashed"), m.get("device") == "cpu")

    chosen = {}
    for r in results:
        k = key_fn(r.get("metadata", {}), r.get("parameters", {}))
        cur = chosen.get(k)
        if cur is None or rank(r) > rank(cur):
            chosen[k] = r
    return list(chosen.values())


def _pitch_key(m, p):
    return (m.get("algorithm_name"), m.get("dataset_name"), m.get("condition", "clean"),
            bool(m.get("probe")), m.get("seed"),
            p.get("sample_rate"), p.get("hop_size"), p.get("max_samples"), p.get("max_seconds"))


def _ood_key(m, _p):
    return (m.get("algorithm_name"), m.get("family"))


def _algos(cells):
    return sorted({k[3] for k in cells})


def _star(cells, algo, cache={}):
    if algo not in cache:
        cache[algo] = metrics.theta_star(cells, algo)
    return cache[algo]


def _is_fixed_operating_point(cells, algo):
    """A flat F@50 sweep on every clean cell = no usable confidence curve (binary voicing)."""
    seen = False
    for (track, _ds, cond, a), cell in cells.items():
        if track == "frame" and a == algo and cond == "clean":
            f = [e["pitch_f"]["f50"] for e in cell.get("results", {}).get("sweep", [])]
            if not f:
                continue
            seen = True
            if max(f) - min(f) > 1e-9:
                return False
    return seen


# --------------------------------------------------------------------------- #
# 1. Leaderboard
# --------------------------------------------------------------------------- #
def section_leaderboard(cells):
    algos = _algos(cells)
    scores = {a: metrics.track_scores(cells, a) for a in algos}
    overalls = {a: metrics.overall(cells, a) for a in algos}

    def _hm_positive(a):
        vals = [v for v in scores[a].values() if v]
        return len(vals) / np.sum(1.0 / np.asarray(vals)) if vals else 0.0

    # zeros rank below every positive Overall but keep an order among themselves
    # (HM of their non-zero tracks), so a failed tracker is still comparable
    order = sorted(algos, key=lambda a: (overalls[a] is None, -(overalls[a] or 0),
                                         -_hm_positive(a)))
    headers = ["**Algorithm**", "**Overall**"] + [f"**{lbl}**" for _k, lbl in TRACK_LABELS]
    cols = {k: [scores[a][k] for a in algos] for k, _l in TRACK_LABELS}
    rows = []
    for a in order:
        dag = "†" if _is_fixed_operating_point(cells, a) else ""
        if overalls[a] == 0.0:
            killers = ", ".join(l for k, l in TRACK_LABELS if scores[a][k] == 0.0)
            ov = f"0 ({killers})"
        else:
            ov = fmt_best(overalls[a], list(overalls.values()))
        row = [a + dag, ov]
        row += [fmt_best(scores[a][k], cols[k]) for k, _l in TRACK_LABELS]
        rows.append(row)
    s = "## Leaderboard\n\n"
    s += ("Overall = harmonic mean of the seven track scores, equal weights. A missing "
          "track makes the Overall n/a (never a silent partial mean); a zero track "
          "sinks it to 0, annotated with the failing track; zero-Overall trackers are "
          "ordered among themselves by the harmonic mean of their non-zero tracks. `†` = fixed operating point (no confidence curve to tune).\n\n")
    s += md_table(headers, rows)
    gaps = {a: [l for (k, l) in TRACK_LABELS if scores[a][k] is None] for a in algos}
    gaps = {a: g for a, g in gaps.items() if g}
    if gaps:
        s += "\nMissing tracks: " + "; ".join(f"{a}: {', '.join(g)}" for a, g in gaps.items()) + ".\n"
    s += "\n### Operating points (theta\\*)\n\n"
    rows = []
    for a in order:
        st = _star(cells, a)
        rows.append([a, "n/a" if st["theta"] is None else f"{st['theta']:.1f}",
                     st["provenance"], str(st["n_datasets"])])
    s += md_table(["**Algorithm**", "**theta\\***", "**selected on**", "**#datasets**"], rows)
    return s


# --------------------------------------------------------------------------- #
# 2. Methodology
# --------------------------------------------------------------------------- #
def section_methodology():
    return f"""## Methodology

### The contract
An algorithm is `(f0(t), q(t))`: a pitch estimate and a voicing confidence per frame.
`f0 <= 0` or `NaN` is a **voicing claim, never a pitch estimate**: at threshold theta, a
frame counts as voiced-with-pitch iff `q >= theta` and f0 is finite-positive. Abstaining on a truly
voiced frame costs recall; it never fabricates a cents error.

### The core metric: the pitch F-score
With counts at threshold theta and cents tolerance T -- `n_ok(T)` = frames truly voiced,
voiced by the tracker, and within T cents; `tp/fp/fn` = the voicing confusion --

    pitch recall    R(T) = n_ok / (tp + fn)
    pitch precision P(T) = n_ok / (tp + fp)
    pitch F-score   F(T) = 2 n_ok / ((tp+fp) + (tp+fn))

Pure counts, no weights. The identity `R = voicing_recall x accuracy_on_voiced`
displays both classic RPA definitions as factors of one product. `T -> inf` recovers
voicing F1; the headline tolerance is 50 cents; the **tolerance AUC** integrates F over
T in [0, {T_MAX:.0f}] and equals `1 - truncated-MAE/{T_MAX:.0f}` on the scored frames.

### The operating point
One global **theta\\*** per algorithm: argmax over the threshold grid of the
equal-per-dataset mean F@50 on clean cells (full cells preferred, probe fallback
stamped; ties -> lowest theta). Frozen; every track reads it. There is no per-cell
threshold selection anywhere.

### The tracks (questions, not decompositions)
Each column is a complete situation of use, so nothing is scored twice. Diagnostics
(voicing P/R/F1, accuracy-on-voiced, RPA@50, cents, RCA, octave/gross rates,
per-band, smoothness, latency, coverage) explain the columns and are never scored.

1. **Accuracy** -- how correct is the output curve on clean real recordings? (F@50)
2. **Noise** -- how much survives corruption? (F@50 degraded / clean, paired probe clips)
3. **Signals** -- how much of the signal-class space is handled? (mean over stationary families; worst family named)
4. **Stability** -- does one threshold work everywhere? (F(theta*)/F(oracle))
5. **Dynamics** -- is a moving pitch followed faithfully? (steady jitter + vibrato retention)
6. **Notes** -- is musical structure recoverable? (COnP)
7. **Speed** -- is it deployable? (1/(1+RTF))

### Weight audit
The only named conventions: tolerances (10/25/50 cents), T_MAX={T_MAX:.0f},
steady-jitter normalizer sigma0={SIGMA0:.0f} cents, speed mapping 1/(1+RTF), the HM's
equal track weights, and equal-per-dataset pooling for track scores (the frame-pooled
variant is shown as a diagnostic). Everything else is counts.

### Statistics
All CIs are 95% paired cluster bootstraps over per-clip sufficient statistics (clusters
= speaker/singer/piece), and every scored value is exactly recomputable from summed
per-clip stats -- the CI machinery resamples and recomputes the same formula it reports.
"""


# --------------------------------------------------------------------------- #
# 3. Datasets
# --------------------------------------------------------------------------- #
def section_datasets(repo_root):
    path = os.path.join(repo_root, "dataset_stats.json")
    s = "## Datasets\n\n"
    if not os.path.exists(path):
        return s + ("Dataset statistics not found. Run "
                    "`scripts/dataset_stats.py --json dataset_stats.json` and commit the "
                    "JSON to render this section.\n")
    rows = json.load(open(path))
    hdr = ["**Dataset**", "**Domain**", "**Clips**", "**Hours**", "**Avg len (s)**",
           "**Voiced %**", "**f0 p5-p50-p95 (Hz)**", "**Band coverage**"]
    body = []
    for r in rows:
        cov = ", ".join(f"{b} {r['bands'][b]:.0f}%" for b, _, _ in PITCH_BANDS
                        if r["bands"].get(b, 0) >= 1.0)
        flag = " (sparse-voiced)" if r["name"] in SPARSE_VOICED else \
               " (voicing-only GT)" if r["name"] in VOICING_ONLY else ""
        body.append([r["name"] + flag, r["domain"], str(r["n"]), f"{r['hours']:.1f}",
                     f"{r['avg_len']:.1f}", f"{r['voiced_pct']:.0f}",
                     f"{r['p5']:.0f}-{r['p50']:.0f}-{r['p95']:.0f}", cov])
    s += md_table(hdr, body)
    s += ("\nBands: " + ", ".join(f"{n} ({band_label(lo, hi)})" for n, lo, hi in PITCH_BANDS)
          + ". f0 statistics are over in-window voiced frames.\n")
    return s


# --------------------------------------------------------------------------- #
# 4. Accuracy
# --------------------------------------------------------------------------- #
def _clean_cells(cells, algo):
    for (track, ds, cond, a), cell in cells.items():
        if track == "frame" and a == algo and cond == "clean" \
                and not cell.get("metadata", {}).get("crashed") \
                and cell.get("results", {}).get("sweep"):
            yield ds, cell


def section_accuracy(cells):
    algos = _algos(cells)
    s = "## Track 1: Accuracy\n\n"
    s += ("Clean real recordings at each algorithm's theta\\*. Datasets flagged "
          "voicing-only are excluded from the score; sparse-voiced corpora are flagged "
          "(precision denominators dominated by silence).\n\n")
    any_rows = False
    for algo in algos:
        st = _star(cells, algo)
        if st["idx"] is None:
            continue
        rows = []
        for ds, cell in sorted(_clean_cells(cells, algo)):
            e = cell["results"]["sweep"][st["idx"]]
            pc = cell["results"].get("per_clip")
            f50 = e["pitch_f"]["f50"]
            if pc and pc.get("stats"):
                keyed, _n = metrics.frame_keyed(pc, st["idx"])
                lo, hi = cluster_bootstrap(list(keyed.values()), FRAME_REDUCERS["pitch_f"])
                f50s = ci_cell(f50, lo, hi)
            else:
                f50s = f"{f50:.3f}"
            flag = " (sparse)" if ds in SPARSE_VOICED else \
                   " (voicing-only, unscored)" if ds in VOICING_ONLY else ""
            probe = " [probe]" if cell.get("metadata", {}).get("probe") else ""
            rows.append([ds + flag + probe, f50s,
                         f"{e['pitch_f']['r50']:.3f}", f"{e['pitch_f']['p50']:.3f}",
                         f"{e['pitch_f']['auc']:.3f}", f"{e['pitch']['rpa']:.3f}",
                         f"{e['pitch']['coverage']:.3f}"])
        if not rows:
            continue
        any_rows = True
        s += f"### {algo}\n\n"
        s += md_table(["**Dataset**", "**F@50 [95% CI]**", "**R@50**", "**P@50**",
                       "**AUC**", "**RPA@50 (voiced)**", "**Coverage**"], rows)
        fac = metrics.factorization(cells, algo, theta_idx=st["idx"])
        if fac:
            s += (f"\nFactorization (frame-pooled): pitch recall {fac['pitch_recall']:.3f} "
                  f"= voicing recall {fac['voicing_recall']:.3f} x accuracy-on-voiced "
                  f"{fac['accuracy_on_voiced']:.3f}; pitch F {fac['pitch_f']:.3f}.\n\n")
    if not any_rows:
        s += "No clean frame cells.\n"
    return s


# --------------------------------------------------------------------------- #
# 5. Noise
# --------------------------------------------------------------------------- #
def section_noise(cells):
    algos = _algos(cells)
    s = "## Track 2: Noise robustness\n\n"
    s += ("Retention ratio F@50(degraded)/F@50(clean) on the SAME probe clips, at "
          "theta\\*, equal-per-dataset. **Floor-effect caveat**: a tracker that is "
          "already poor on clean has little to lose; read this column next to "
          "Accuracy, never alone.\n\n")
    conds = sorted({k[2] for k in cells if k[0] == "frame"
                    and k[2] not in ("clean", "clean_probe")})
    if not conds:
        return s + "No degradation cells.\n"
    rows = []
    per_algo = {}
    for algo in algos:
        nz = metrics.track_noise(cells, algo)
        per_algo[algo] = nz.get("per_condition", {})
        row = [algo, "n/a" if nz["score"] is None else f"{nz['score']:.3f}"]
        row += [f"{per_algo[algo][c]:.3f}" if c in per_algo[algo] else "n/a" for c in conds]
        rows.append(row)
    s += md_table(["**Algorithm**", "**Mean**"] + [f"**{c}**" for c in conds], rows)
    fams = {f: [c for c in members if c in conds]
            for f, members in CONDITION_FAMILIES.items()}
    fams = {f: cs for f, cs in fams.items() if len(cs) >= 2}
    if fams:
        rows = []
        for algo in algos:
            row = [algo]
            for f, cs in fams.items():
                vals = [per_algo[algo][c] for c in cs if c in per_algo[algo]]
                row.append(f"{np.mean(vals):.3f}" if vals else "n/a")
            rows.append(row)
        s += "\nBy condition family:\n\n"
        s += md_table(["**Algorithm**"] + [f"**{f}**" for f in fams], rows)
    return s


# --------------------------------------------------------------------------- #
# 6. Signals
# --------------------------------------------------------------------------- #
def section_signals(cells):
    algos = _algos(cells)
    s = "## Track 3: Signal robustness\n\n"
    s += ("Stationary synthetic families with exact labels; per-family accuracy = "
          "coverage-aware pitch recall@50 at theta\\*. Score = the MEAN over families "
          "(each family is one probe question, equally weighted); the worst family is "
          "named beside it as the diagnostic. A worst-family SCORE was measured and "
          "rejected: 6 of 7 surveyed trackers have at least one exactly-dead family, "
          "so a min would zero almost the whole field. Controls (no pitch present) "
          "report false-positive rate as a diagnostic.\n\n")
    fams = sorted({k[1] for k, c in cells.items() if k[0] == "synthetic"
                   and c.get("results", {}).get("kind") == "stationary"})
    if not fams:
        return s + "No synthetic stationary cells.\n"
    rows = []
    for algo in algos:
        sig = metrics.track_signals(cells, algo)
        pf = sig.get("per_family", {})
        row = [algo, "n/a" if sig["score"] is None else
               f"**{sig['score']:.2f}** (worst {sig['worst_family']} {sig['worst']:.2f})"]
        row += [f"{pf[f]:.2f}" if f in pf else "n/a" for f in fams]
        rows.append(row)
    s += md_table(["**Algorithm**", "**Score (worst)**"] + [f"**{f}**" for f in fams], rows)
    ctl = sorted({k[1] for k, c in cells.items() if k[0] == "synthetic"
                  and c.get("results", {}).get("kind") == "control"})
    if ctl:
        rows = []
        for algo in algos:
            row = [algo]
            for f in ctl:
                cell = cells.get(("synthetic", f, None, algo))
                fp = cell.get("results", {}).get("false_positive_rate") if cell else None
                row.append("n/a" if _is_bad(fp) else f"{fp:.3f}")
            rows.append(row)
        s += "\nControls (false-positive rate, lower is better; diagnostic only):\n\n"
        s += md_table(["**Algorithm**"] + [f"**{f}**" for f in ctl], rows)
    return s


# --------------------------------------------------------------------------- #
# 7. Stability
# --------------------------------------------------------------------------- #
def section_stability(cells):
    algos = _algos(cells)
    s = "## Track 4: Operating stability\n\n"
    s += ("F@50 at the frozen theta\\* divided by F@50 at each cell's oracle threshold "
          "(1.0 = the one global threshold loses nothing anywhere). Fixed-operating-"
          "point trackers are trivially 1.0 (nothing to mistune) and flagged `†` on the "
          "leaderboard.\n\n")
    rows = []
    for algo in algos:
        st = metrics.track_stability(cells, algo)
        star = _star(cells, algo)
        flat = []
        for _ds, cell in _clean_cells(cells, algo):
            f = [e["pitch_f"]["f50"] for e in cell["results"]["sweep"]]
            if f and star["idx"] is not None:
                flat.append(max(f) - f[star["idx"]])
        rows.append([algo,
                     "n/a" if st["score"] is None else f"{st['score']:.3f}",
                     f"{max(flat):.3f}" if flat else "n/a"])
    s += md_table(["**Algorithm**", "**Efficiency**", "**Worst peak-vs-theta\\* gap**"], rows)
    return s


# --------------------------------------------------------------------------- #
# 8. Dynamics
# --------------------------------------------------------------------------- #
def section_dynamics(cells):
    algos = _algos(cells)
    s = "## Track 5: Tracking dynamics\n\n"
    s += (f"Trajectory families with exact labels, read at theta\\*. Steady tones: "
          f"jitter (cents std) and bias, scored sigma0/(sigma0+jitter) with "
          f"sigma0={SIGMA0:.0f}c. Vibrato: modulation-depth retention x voiced "
          f"coverage. Track score = mean of the two family groups.\n\n")
    fams = sorted({k[1] for k, c in cells.items() if k[0] == "synthetic"
                   and c.get("results", {}).get("kind") == "trajectory"})
    if not fams:
        return s + "No trajectory cells.\n"
    rows = []
    for algo in algos:
        dyn = metrics.track_dynamics(cells, algo)
        row = [algo, "n/a" if dyn["score"] is None else f"{dyn['score']:.3f}"]
        for f in fams:
            e = dyn.get("per_family", {}).get(f)
            if not e:
                row.append("n/a")
            elif "jitter_cents" in e:
                j, b = e.get("jitter_cents"), e.get("bias_cents")
                row.append("n/a" if _is_bad(j) else f"jit {j:.1f}c, bias {b:+.1f}c")
            else:
                row.append(f"ret {e.get('depth_retention', 0):.2f} x cov "
                           f"{e.get('coverage', 0):.2f}")
        rows.append(row)
    s += md_table(["**Algorithm**", "**Score**"] + [f"**{f}**" for f in fams], rows)
    return s


# --------------------------------------------------------------------------- #
# 9. Notes
# --------------------------------------------------------------------------- #
def section_notes(cells):
    algos = _algos(cells)
    s = "## Track 6: Notes\n\n"
    s += ("Note transcription (COnP / COnPOff). This track selects its own threshold "
          "and segmentation cost internally -- the one deliberate exception to the "
          "global-theta rule, documented here.\n\n")
    note_ds = sorted({k[1] for k in cells if k[0] == "note"})
    if not note_ds:
        return s + "No note cells.\n"
    rows = []
    for algo in algos:
        row = [algo]
        for ds in note_ds:
            cell = cells.get(("note", ds, "clean", algo))
            r = cell.get("results", {}) if cell else {}
            conp, conpoff = r.get("conp"), r.get("conpoff")
            row.append("n/a" if _is_bad(conp) else f"{conp:.3f} / {conpoff:.3f}")
        rows.append(row)
    s += md_table(["**Algorithm**"] + [f"**{d}** (COnP/COnPOff)" for d in note_ds], rows)
    return s


# --------------------------------------------------------------------------- #
# 10. Speed
# --------------------------------------------------------------------------- #
def section_speed(cells):
    algos = _algos(cells)
    s = "## Track 7: Speed\n\n"
    rows = []
    for algo in algos:
        sp = metrics.track_speed(cells, algo)
        rows.append([algo,
                     "n/a" if sp.get("rtf_cpu") is None else f"{sp['rtf_cpu']:.3f}",
                     "n/a" if sp["score"] is None else f"{sp['score']:.3f}"])
    if not any(r[1] != "n/a" for r in rows):
        return s + "No speed cells.\n"
    s += md_table(["**Algorithm**", "**RTF (cpu)**", "**Score 1/(1+RTF)**"], rows)
    return s


# --------------------------------------------------------------------------- #
# 11. Caveats
# --------------------------------------------------------------------------- #
def section_caveats():
    return """## Caveats

- **Floor effect** (Noise track): retention is only meaningful next to absolute
  Accuracy; a tracker that is poor on clean audio has little room to drop.
- **Sparse-voiced corpora** (OSFGlottis, AVID): mostly-unvoiced sessions; precision
  denominators are dominated by silence false positives. Flagged in the tables.
- **Score-grade ground truth** (M4Singer): notated pitch, not performed f0; voicing GT
  is reliable, pitch scores are not; excluded from Accuracy.
- **Training-data leakage**: learned trackers may have trained on these public corpora.
  A clean-only advantage that collapses under degradation is a leakage signature;
  degraded and synthetic tracks move inputs away from anything seen verbatim.
- **Why v1 scored pitch conditionally, and v2 does not**: v1 computed RPA only where
  both sides agreed on voicing, to dodge GT voicing-label errors and to accommodate
  trackers that output no pitch on unvoiced frames. v2 scores the joint event instead:
  labels are now exact-by-construction or consensus-derived, paired comparisons cancel
  the shared residue, and the (f0, q) contract makes abstention a voicing claim (a
  recall cost) rather than a fabricated pitch error. The conditional quantity survives
  as the accuracy-on-voiced diagnostic in the factorization.
"""


# --------------------------------------------------------------------------- #
# 12. Appendix
# --------------------------------------------------------------------------- #
def section_appendix(cells):
    algos = _algos(cells)
    s = "## Appendix\n\n### Accuracy by dataset group\n\n"
    per_algo_ds = {}
    for algo in algos:
        st = _star(cells, algo)
        if st["idx"] is None:
            continue
        per_algo_ds[algo] = {ds: cell["results"]["sweep"][st["idx"]]["pitch_f"]["f50"]
                             for ds, cell in _clean_cells(cells, algo)
                             if ds not in VOICING_ONLY}
    grouped = {d for t in DATASET_GROUPS.values() for names in t.values() for d in names}
    present = {d for v in per_algo_ds.values() for d in v}
    for d in sorted(present - grouped):
        print(f"Warning: dataset {d} not covered by DATASET_GROUPS")
    for title, groups in DATASET_GROUPS.items():
        rows = []
        for algo, by_ds in per_algo_ds.items():
            row = [algo]
            for gname, members in groups.items():
                vals = [by_ds[d] for d in members if d in by_ds]
                row.append(f"{np.mean(vals):.3f}" if vals else "n/a")
            rows.append(row)
        if rows:
            s += f"**{title}**\n\n"
            s += md_table(["**Algorithm**"] + [f"**{g}**" for g in groups], rows) + "\n"
    s += "### Aggregation sensitivity\n\n"
    s += ("The Overall uses the harmonic mean (one mean family everywhere; dominated by "
          "the weakest track). The alternatives on the same track scores:\n\n")
    rows = []
    for algo in algos:
        sc = [v for v in metrics.track_scores(cells, algo).values() if v is not None]
        if len(sc) != len(TRACKS_SCORED) or any(v <= 0 for v in sc):
            rows.append([algo, "n/a", "n/a", "n/a"])
            continue
        am = float(np.mean(sc))
        gm = float(np.exp(np.mean(np.log(sc))))
        hm = float(len(sc) / np.sum(1.0 / np.asarray(sc)))
        rows.append([algo, f"{am:.3f}", f"{gm:.3f}", f"{hm:.3f}"])
    s += md_table(["**Algorithm**", "**AM**", "**GM**", "**HM (used)**"], rows)
    return s


# --------------------------------------------------------------------------- #
def update_readme_table(readme_path, cells):
    """Replace the table between '## Overall Results' and the next '##' heading."""
    if not os.path.exists(readme_path):
        return
    algos = _algos(cells)
    overalls = {a: metrics.overall(cells, a) for a in algos}
    complete = {a: v for a, v in overalls.items() if v is not None}
    order = sorted(complete, key=lambda a: -complete[a])
    scores = {a: metrics.track_scores(cells, a) for a in order}
    lines = ["", "The overall score is the harmonic mean of the seven track scores "
                 "(see the benchmark report for definitions and diagnostics).", ""]
    if order:
        hdr = ["**Algorithm**", "**Overall**"] + [f"**{l}**" for _k, l in TRACK_LABELS]
        rows = [[a, f"{complete[a]:.3f}"]
                + [f"{scores[a][k]:.3f}" for k, _l in TRACK_LABELS] for a in order]
        lines.append(md_table(hdr, rows).rstrip())
    partial = sorted(set(algos) - set(order))
    if partial:
        lines += ["", "Awaiting complete v2 runs: " + ", ".join(partial) + "."]
    lines.append("")
    src = open(readme_path).read()
    head = "## Overall Results"
    i = src.find(head)
    if i < 0:
        return
    j = src.find("\n## ", i + len(head))
    j = len(src) if j < 0 else j
    open(readme_path, "w").write(src[:i + len(head)] + "\n" + "\n".join(lines) + src[j:])
    print(f"README table updated -> {readme_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="benchmark_report.md")
    ap.add_argument("--readme", default=None,
                    help="optional README.md whose Overall Results table is regenerated")
    args = ap.parse_args()

    cells = load_cells(args.results)
    if not cells:
        raise SystemExit(f"no cells in {args.results}")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    parts = [
        "# Pitch Benchmark Report\n",
        section_leaderboard(cells),
        section_methodology(),
        section_datasets(repo_root),
        section_accuracy(cells),
        section_noise(cells),
        section_signals(cells),
        section_stability(cells),
        section_dynamics(cells),
        section_notes(cells),
        section_speed(cells),
        section_caveats(),
        section_appendix(cells),
    ]
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"report -> {args.out}")
    if args.readme:
        update_readme_table(args.readme, cells)


if __name__ == "__main__":
    main()
