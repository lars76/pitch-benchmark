#!/usr/bin/env python3
"""The v2 report: results first, definitions in METRICS.md.

Presentation only -- every number here comes from metrics.py, so a table can never
disagree with a score. Renders whatever cells exist; a track with no cells is one
sentence, never a column of blanks.

    .venv/bin/python generate_report.py --results results --out benchmark_report.md
"""

import argparse
import json
import os

import numpy as np

import metrics
from evaluate import load_cells
from metrics import PITCH_BANDS, SIGMA0, band_label, cluster_bootstrap, FRAME_REDUCERS

# Sparse-voiced corpora: long clinical/lab sessions that are mostly UNVOICED; precision
# denominators are dominated by silence false positives and are not
# comparable to the dense corpora. Flagged in place, never dropped -- unlike
# metrics.PITCH_INELIGIBLE, which the scorer excludes outright.
SPARSE_VOICED = frozenset({"OSFGlottis", "AVID"})

# The split a reader actually chooses on. Finer groupings were measured to be exact means
# of the per-dataset numbers already printed, so they are not rendered.
DOMAIN_GROUPS = {
    "Speech": ["SpeechSynth", "PTDB", "MOCHA", "CMUArctic", "AVID", "OSFGlottis",
               "SVD", "APLAWD", "KEELE", "FDA"],
    "Music": ["Bach10Synth", "MDBStemSynth", "NSynth", "Vocadito", "MIR1K", "URMP"],
}

TRACK_LABELS = [("correctness", "Correctness"), ("noise", "Noise"),
                ("signal_types", "Signal types"), ("tracking", "Tracking"),
                ("notes", "Notes"), ("speed", "Speed")]


def _and_list(items):
    """'a', 'a and b', 'a, b and c' -- so a list of missing tracks reads as English."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _is_bad(v):
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "---|" * len(headers) + "\n"
    for row in rows:
        out += "| " + " | ".join(row) + " |\n"
    return out


def fmt_best(value, column_values, fmt="{:.3f}", lower=False, na="n/a"):
    """Format `value`, bolding it if it is the best (max, or min if lower) of the column."""
    if _is_bad(value):
        return na
    valid = [v for v in column_values if not _is_bad(v)]
    best = (min if lower else max)(valid) if valid else None
    s = fmt.format(value)
    return f"**{s}**" if best is not None and abs(value - best) < 1e-9 else s


def ci_cell(value, lo, hi):
    return f"{value:.3f} [n/a]" if _is_bad(lo) else f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def _algos(cells):
    return sorted({k.algo for k in cells})


def _live_tracks(cells, algos):
    """Only the tracks something actually measured. A track with no cells anywhere is
    reported once, in a sentence, instead of as a column of n/a in every table."""
    # Tracks is attribute-addressed; flatten to {name: value} once here so the
    # rendering below stays a plain lookup by the track name it is iterating.
    scored = {a: dict(metrics.track_scores(cells, a, intervals=False).items())
              for a in algos}
    scored = {a: {n: s.value for n, s in d.items()} for a, d in scored.items()}
    live = [(k, l) for k, l in TRACK_LABELS
            if any(scored[a][k] is not None for a in algos)]
    missing = [l for k, l in TRACK_LABELS if (k, l) not in live]
    return scored, live, missing


# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #
def section_leaderboard(cells):
    algos = _algos(cells)
    scored, live, missing = _live_tracks(cells, algos)
    overalls = {a: metrics.overall(cells, a) for a in algos}
    order = sorted(algos, key=lambda a: metrics.rank_key(overalls[a], scored[a].values()))

    s = "## Leaderboard\n\n"
    if missing:
        s += (f"**{_and_list(missing)} {'are' if len(missing) > 1 else 'is'} not measured "
              f"in this run**, so Overall is not computed; rank by Correctness. "
              f"Definitions for every column are in [METRICS.md](METRICS.md).\n\n")
    else:
        s += ("Overall is the harmonic mean of the six track scores, so a tracker is only "
              "as good as its weakest situation of use. Definitions for every column are "
              "in [METRICS.md](METRICS.md).\n\n")

    cols = {k: [scored[a][k] for a in algos] for k, _l in live}
    rows = []
    for a in order:
        dag = " †" if metrics.has_fixed_operating_point(cells, a) else ""
        ov = overalls[a]
        if ov == 0.0:
            killers = ", ".join(l for k, l in live if scored[a][k] == 0.0)
            ov_cell = f"0 ({killers})"
        else:
            ov_cell = fmt_best(ov, list(overalls.values()))
        star = metrics.theta_star(cells, a)
        rows.append([a + dag, ov_cell,
                     "n/a" if star["theta"] is None else f"{star['theta']:.1f}"]
                    + [fmt_best(scored[a][k], cols[k]) for k, _l in live])
    s += md_table(["**Algorithm**", "**Overall**", "**theta\\***"]
                  + [f"**{l}**" for _k, l in live], rows)
    if any(metrics.has_fixed_operating_point(cells, a) for a in algos):
        s += "\n† no usable voicing confidence: the sweep is flat, so theta\\* is nominal.\n"
    s += _headline(cells, order, scored, live)
    return s


def _headline(cells, order, scored, live):
    """The sentence a reader came for: who wins, and what it costs."""
    if len(order) < 2:
        return ""
    best = order[0]
    per_track_best = {l: max((a for a in order if scored[a][k] is not None),
                             key=lambda a: scored[a][k], default=None)
                      for k, l in live}
    others = sorted({l: a for l, a in per_track_best.items() if a and a != best}.items())
    if not others:
        return f"\n**{best}** leads every measured track.\n"
    trade = "; ".join(f"{a} leads {l}" for l, a in others)
    return f"\n**{best}** leads overall; {trade}.\n"


# --------------------------------------------------------------------------- #
# Track 1: Correctness
# --------------------------------------------------------------------------- #
def section_correctness(cells):
    s = "## Track 1: Correctness\n\n"
    s += ("Pitch F on clean real recordings at each algorithm's theta\\*, averaged with "
          "equal weight per dataset. Datasets whose pitch ground truth is score-grade are "
          "excluded by the scorer; sparse-voiced corpora are flagged in place.\n\n")
    any_rows = False
    for algo in _algos(cells):
        star = metrics.theta_star(cells, algo)
        if star["idx"] is None:
            continue
        acc = metrics.track_correctness(cells, algo)
        rows = []
        for ds, cap, cell in sorted(metrics.frame_cells(cells, algo, "clean")):
            e = cell["results"]["sweep"][star["idx"]]
            pc = cell["results"].get("per_clip")
            if pc and pc.get("stats"):
                keyed, _n = metrics.frame_keyed(pc, star["idx"])
                lo, hi = cluster_bootstrap(list(keyed.values()), FRAME_REDUCERS["pitch_f1"])
                fs = ci_cell(e["pitch"]["f1"], lo, hi)
            else:
                fs = f"{e['pitch']['f1']:.3f}"
            flag = " (sparse)" if ds in SPARSE_VOICED else ""
            probe = f" [capped n{cap[0]}/t{cap[1]}]" if cap else ""
            rows.append([ds + flag + probe, fs, f"{e['pitch']['recall']:.3f}",
                         f"{e['pitch']['precision']:.3f}"])
        # a crashed dataset scores 0 and must be VISIBLE, or the track score cannot be
        # reconciled with the rows above it
        for ds in sorted(metrics.crashed_datasets(cells, algo, conditions={"clean"})):
            rows.append([ds, "0.000 (crashed)", "0.000", "0.000"])
        if not rows:
            continue
        any_rows = True
        s += f"### {algo}\n\n"
        s += md_table(["**Dataset**", "**pitch F1 [95% CI]**", "**pitch recall**",
                       "**pitch precision**"], rows)
        if acc["score"] is not None:
            s += f"\nTrack score (equal per dataset): **{acc['score']:.3f}**"
            ci = metrics.track_ci(cells, algo, "correctness")
            if ci:
                s += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
            if acc.get("n_crashed"):
                s += f" over {acc['n_completed']} completed + {acc['n_crashed']} crashed"
            s += ".\n"
        fac = metrics.factorization(cells, algo, theta_idx=star["idx"])
        if fac:
            s += (f"\nWhere the recall comes from: pitch recall {fac['pitch_recall']:.3f} "
                  f"= voicing recall {fac['voicing_recall']:.3f} x the correct rate "
                  f"(exact+close, below).\n\n")
    if not any_rows:
        s += "No clean frame cells.\n"
    return s + _error_breakdown(cells) + _band_table(cells) + _ties_table(cells)


def _error_breakdown(cells):
    """WHERE the errors are: is it precision, or is it picking the wrong pitch?"""
    rows = []
    for algo in _algos(cells):
        star = metrics.theta_star(cells, algo)
        if star["idx"] is None:
            continue
        e = metrics.error_classes(cells, algo, theta_idx=star["idx"])
        if e:
            rows.append([algo, f"{e['exact']:.3f}", f"{e['close']:.3f}",
                         f"{e['off']:.3f}", f"{e['wrong']:.3f}",
                         f"{e['cents_mae']:.1f}",
                         "n/a" if _is_bad(e['cents_bias']) else f"{e['cents_bias']:+.1f}"])
    if not rows:
        return ""
    return ("\n### Error breakdown\n\nOn frames scored for pitch, at theta\\*, pooled "
            "over all conditions. The four classes are disjoint and cover every scored "
            "frame, so a row reads as a sentence: *exact on X, loses tuning on Y, wanders "
            "on Z, picks the wrong pitch on W*. `cents bias` is the mean SIGNED error over the "
            "in-tolerance frames only (so its denominator differs from MAE's): 0 means "
            "symmetric scatter, non-zero means the tracker sits systematically sharp (+) "
            "or flat (-) and can be corrected by subtracting it. It is the only signed "
            "column; every other one is blind to the direction of the error.\n\n"
            + md_table(["**Algorithm**", "**exact** (<10c)", "**close** (10-50c)",
                        "**off** (50-200c)", "**wrong** (>200c)", "**cents MAE**",
                        "**cents bias**"], rows))


def _band_table(cells):
    bands = [b for b, _, _ in PITCH_BANDS]
    rows = []
    for algo in _algos(cells):
        star = metrics.theta_star(cells, algo)
        if star["idx"] is None:
            continue
        agg = metrics.band_aggregate(cells, algo, theta_idx=star["idx"])
        if not any(n for _r, n in agg.values()):
            continue
        rows.append([algo] + [f"{r:.2f} ({n:,})" if r is not None else "n/a"
                              for r, n in (agg[b] for b in bands)])
    if not rows:
        return ""
    return ("\n### By pitch band\n\nCorrect rate (exact+close) with the frame count it "
            "rests on, by ground-truth band. A register collapse shows up here and "
            "nowhere else in the aggregate; a small count means the cell is not "
            "evidence.\n\n"
            + md_table(["**Algorithm**"] + [f"**{b}** ({band_label(lo, hi)})"
                                            for b, lo, hi in PITCH_BANDS], rows))


def _ties_table(cells):
    t = metrics.ties(cells)
    if not t:
        return ""
    rows = [[ds, f"{d['best']} ({d['f1']:.3f})", ", ".join(d["tied"]) or "none"]
            for ds, d in sorted(t.items())]
    return ("\n### Statistical ties\n\nPer clean dataset: the best pitch F1, and every "
            "algorithm whose paired 95% CI of the difference over the clips both scored "
            "includes 0. A tied algorithm is not beaten on that dataset, whatever the "
            "third decimal suggests. Many pairs are compared with no multiple-comparisons "
            "correction, so read a single tie as descriptive.\n\n"
            + md_table(["**Dataset**", "**best pitch F1**", "**tied with best**"], rows))


# --------------------------------------------------------------------------- #
# Track 2: Noise
# --------------------------------------------------------------------------- #
def section_noise(cells):
    algos = _algos(cells)
    s = "## Track 2: Noise\n\n"
    s += ("Absolute pitch F1 on the degraded conditions at theta\\*, equal weight per "
          "dataset. This is how well the tracker works in noise, not how much of its "
          "clean performance it kept: a retention ratio rises when clean performance "
          "falls, so it rewards being bad on clean and is not scored.\n\n")
    conds = sorted({k.condition for k in cells if k.suite == metrics.Suite.FRAME
                    and k.condition != "clean"})
    if not conds:
        return s + "No degradation cells.\n"
    rows = []
    for algo in algos:
        nz = metrics.track_noise(cells, algo)
        per = nz.get("per_condition", {})
        lo, hi, n = metrics.noise_drop_ci(cells, algo)
        ci = metrics.track_ci(cells, algo, "noise")
        score_cell = ("n/a" if nz["score"] is None else
                      f"{nz['score']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
        rows.append([algo, score_cell,
                     str(nz.get("n_conditions", 0)),
                     "n/a" if not n else f"{lo:.1f} to {hi:.1f}"]
                    + [f"{per[c]:.3f}" if c in per else "n/a" for c in conds])
    s += md_table(["**Algorithm**", "**Score**", "**#conditions**",
                   "**drop pp [95% CI]**"] + [f"**{c}**" for c in conds], rows)
    s += ("\n`drop` is the paired clean-minus-degraded difference in pitch F1, in "
          "percentage points, resampled over the clips both sides scored. `#conditions` "
          "is the denominator of the score: two algorithms averaged over different "
          "numbers of conditions are not directly comparable.\n")
    return s


# --------------------------------------------------------------------------- #
# Track 3: Signal types
# --------------------------------------------------------------------------- #
def section_signal_types(cells):
    algos = _algos(cells)
    s = "## Track 3: Signal types\n\n"
    s += ("Synthetic families with exact labels. Per family: unconditional pitch recall "
          "at theta\\*, so a family the tracker refuses to voice scores low rather than "
          "vanishing. Score = mean over families, each family one equally-weighted probe "
          "question. A dead family is one scoring exactly 0 -- the capability is absent, "
          "not merely weak.\n\n")
    rows, any_row = [], False
    for algo in algos:
        sig = metrics.track_signal_types(cells, algo)
        if sig["score"] is None:
            continue
        any_row = True
        pf = sig.get("per_family", {})
        dead = sorted(f for f, v in pf.items() if v == 0.0)
        rows.append([algo, f"{sig['score']:.3f}",
                     f"{sig['worst_family']} ({sig['worst']:.2f})",
                     str(len(dead)), ", ".join(dead) if dead else "none"])
    if not any_row:
        return s + "No synthetic stationary cells.\n"
    s += md_table(["**Algorithm**", "**Score**", "**worst family**",
                   "**dead**", "**which**"], rows)
    s += _controls_line(cells, algos)
    return s


def _controls_line(cells, algos):
    """Controls carry no pitch, so a false positive is the only thing to report. One line:
    the column is 0.000 for nearly everyone and a table would be mostly zeros."""
    ctl = sorted({k.subject for k, c in cells.items() if k.suite == metrics.Suite.SYNTHETIC
                  and (c.get("results") or {}).get("kind") == "control"})
    if not ctl:
        return ""
    hits = []
    for algo in algos:
        for f in ctl:
            cell = cells.get(metrics.CellKey(suite="synthetic", subject=f,
                                             condition=None, algo=algo))
            fp = (cell.get("results") or {}).get("false_positive_rate") if cell else None
            if not _is_bad(fp) and fp > 0:
                hits.append(f"{algo} {f} {fp:.3f}")
    tail = "; ".join(hits) if hits else "no tracker fires on any control"
    return (f"\nControls ({', '.join(ctl)}) contain no pitch, so the only readout is the "
            f"false-positive rate (diagnostic, never scored): {tail}.\n")


# --------------------------------------------------------------------------- #
# Track 4: Tracking
# --------------------------------------------------------------------------- #
def section_tracking(cells):
    s = "## Track 4: Tracking\n\n"
    s += (f"Trajectory families with exact labels, read at theta\\*. Steady tones score "
          f"sigma0/(sigma0+error) on the TOTAL error hypot(jitter, bias) with "
          f"sigma0={SIGMA0:.0f} cents, so a constant output that is an octave off cannot "
          f"win by having no jitter. Vibrato scores depth ratio x voiced fraction. Track "
          f"score = mean of the two capabilities.\n\n")
    fams = sorted({k.subject for k, c in cells.items() if k.suite == metrics.Suite.SYNTHETIC
                   and (c.get("results") or {}).get("kind") == "trajectory"})
    if not fams:
        return s + "No trajectory cells.\n"
    rows = []
    for algo in _algos(cells):
        trk = metrics.track_tracking(cells, algo)
        if trk["score"] is None:
            continue
        row = [algo, f"{trk['score']:.3f}"]
        for f in fams:
            e = trk.get("per_family", {}).get(f)
            if not e:
                row.append("n/a")
            elif "jitter_cents" in e:
                j, b = e.get("jitter_cents"), e.get("bias_cents")
                row.append("n/a" if _is_bad(j)
                           else f"jitter {j:.1f}c, bias {0.0 if _is_bad(b) else b:+.1f}c")
            else:
                row.append(f"depth {e.get('depth_ratio') or 0:.2f} x "
                           f"voiced {e.get('voiced_fraction') or 0:.2f}")
        rows.append(row)
    if not rows:
        return s + "No trajectory cells.\n"
    return s + md_table(["**Algorithm**", "**Score**"] + [f"**{f}**" for f in fams], rows)


# --------------------------------------------------------------------------- #
# Track 5: Notes   /   Track 6: Speed
# --------------------------------------------------------------------------- #
def section_notes(cells):
    s = "## Track 5: Notes\n\n"
    note_ds = sorted({k.subject for k in cells if k.suite == metrics.Suite.NOTE})
    if not note_ds:
        return s + "Not measured in this run.\n"
    s += ("Note transcription (COnP = correct onset and pitch; COnPOff also requires the "
          "offset). This track selects its own threshold and segmentation cost "
          "internally, the one deliberate exception to the global theta\\* rule.\n\n")
    rows = []
    for algo in _algos(cells):
        row = [algo]
        for ds in note_ds:
            cell = cells.get(metrics.CellKey(suite="note", subject=ds,
                                             condition="clean", algo=algo))
            r = (cell or {}).get("results", {})
            conp, conpoff = r.get("conp"), r.get("conpoff")
            row.append("n/a" if _is_bad(conp) else f"{conp:.3f} / {conpoff:.3f}")
        rows.append(row)
    return s + md_table(["**Algorithm**"] + [f"**{d}** (COnP/COnPOff)" for d in note_ds],
                        rows)


def section_speed(cells):
    s = "## Track 6: Speed\n\n"
    rows = []
    for algo in _algos(cells):
        sp = metrics.track_speed(cells, algo)
        rows.append([algo, "n/a" if sp.get("rtf_cpu") is None else f"{sp['rtf_cpu']:.4f}",
                     "n/a" if sp.get("score") is None else f"{sp['score']:.3f}"])
    if not any(r[1] != "n/a" for r in rows):
        return s + "Not measured in this run.\n"
    s += ("Real-time factor on cpu under controlled conditions (serial, isolated, "
          "repeated): seconds of compute per second of audio, so lower is faster. "
          "Score = 1/(1+RTF).\n\n")
    return s + md_table(["**Algorithm**", "**RTF (cpu)**", "**Score**"], rows)


# --------------------------------------------------------------------------- #
# Reliability, datasets, caveats
# --------------------------------------------------------------------------- #
def section_reliability(cells):
    """Reliability, not scored: did each cell finish, and how did the rest fail.

    The numbers come from metrics.cost_summary; this only formats them."""
    rows = []
    for algo in _algos(cells):
        c = metrics.cost_summary(cells, algo)
        if not c["attempted"]:
            continue
        detail = "" if not c["crash_kinds"] else " (" + ", ".join(
            f"{v}x {k}" for k, v in sorted(c["crash_kinds"].items())) + ")"
        rows.append([algo, f"{c['completed']}/{c['attempted']}{detail}"])
    if not rows:
        return ""
    return ("## Reliability\n\nNothing here is scored -- these are facts about running the "
            "code. A crashed cell still counts against the tracker's scores (it contributes "
            "0 to its track), so `completed/attempted` explains a low score rather than "
            "excusing it. Speed is measured separately, and comparably, by the Speed "
            "track.\n\n"
            + md_table(["**Algorithm**", "**completed/attempted**"], rows))


def section_by_domain(cells):
    """Speech vs music: the one grouping a reader chooses on. Finer groupings were
    measured to be exact means of the per-dataset numbers already printed."""
    rows = []
    for algo in _algos(cells):
        star = metrics.theta_star(cells, algo)
        if star["idx"] is None:
            continue
        by_ds = {ds: cell["results"]["sweep"][star["idx"]]["pitch"]["f1"]
                 for ds, _cap, cell in metrics.frame_cells(cells, algo, "clean")}
        row = [algo]
        for _g, members in DOMAIN_GROUPS.items():
            vals = [by_ds[d] for d in members if d in by_ds]
            row.append(f"{np.mean(vals):.3f}" if vals else "n/a")
        if any(c != "n/a" for c in row[1:]):
            rows.append(row)
    if not rows:
        return ""
    return ("\n### Correctness by domain\n\nClean pitch F1, equal weight per dataset "
            "within each domain.\n\n"
            + md_table(["**Algorithm**"] + [f"**{g}**" for g in DOMAIN_GROUPS], rows))


def section_datasets(repo_root):
    path = os.path.join(repo_root, "dataset_stats.json")
    s = "## Datasets\n\n"
    if not os.path.exists(path):
        return s + ("Dataset statistics not found. Run "
                    "`scripts/dataset_stats.py --json dataset_stats.json` and commit the "
                    "JSON to render this section.\n")
    rows = json.load(open(path))
    body = []
    for r in rows:
        by_band = ", ".join(f"{b} {r['bands'][b]:.0f}%" for b, _, _ in PITCH_BANDS
                            if r["bands"].get(b, 0) >= 1.0)
        flag = " (sparse-voiced)" if r["name"] in SPARSE_VOICED else \
               " (score-grade pitch GT, unscored)" if r["name"] in metrics.PITCH_INELIGIBLE \
               else ""
        body.append([r["name"] + flag, r["domain"], str(r["n"]), f"{r['hours']:.1f}",
                     f"{r['voiced_pct']:.0f}",
                     f"{r['p5']:.0f}-{r['p50']:.0f}-{r['p95']:.0f}", by_band])
    s += md_table(["**Dataset**", "**Domain**", "**Clips**", "**Hours**", "**Voiced %**",
                   "**f0 p5-p50-p95 (Hz)**", "**f0 by band**"], body)
    s += ("\nBands: " + ", ".join(f"{n} ({band_label(lo, hi)})" for n, lo, hi in PITCH_BANDS)
          + ". f0 statistics are over in-window voiced frames.\n")
    return s


def section_caveats():
    return """## Caveats

- **Sparse-voiced corpora** (OSFGlottis, AVID): mostly-unvoiced sessions, so precision
  denominators are dominated by silence false positives. Flagged in the tables, never
  dropped.
- **Score-grade pitch ground truth** (M4Singer): the notated note, not the performed f0.
  Its voicing labels are sound, so the corpus is kept for voicing, but it is excluded
  from every pitch number including theta\\* selection.
- **Training-data leakage**: learned trackers may have trained on these public corpora.
  A clean-only advantage that collapses under degradation is a leakage signature; the
  degraded and synthetic tracks move inputs away from anything seen verbatim.
"""


# --------------------------------------------------------------------------- #
def update_readme_table(readme_path, cells):
    """Replace the table between '## Overall Results' and the next '##' heading."""
    if not os.path.exists(readme_path):
        return
    algos = _algos(cells)
    scored, live, missing = _live_tracks(cells, algos)
    overalls = {a: metrics.overall(cells, a) for a in algos}
    complete = {a: v for a, v in overalls.items() if v is not None}
    order = sorted(complete, key=lambda a: -complete[a])
    note = ("The overall score is the harmonic mean of the six track scores "
            "(see [METRICS.md](METRICS.md) for definitions).")
    if missing:
        note = (f"{_and_list(missing)} {'are' if len(missing) > 1 else 'is'} not yet "
                f"measured, so Overall is not computed; the table ranks by Correctness. "
                f"See [METRICS.md](METRICS.md) for definitions.")
        order = sorted(algos, key=lambda a: -(scored[a]["correctness"] or 0))
    lines = ["", note, ""]
    if order:
        hdr = ["**Algorithm**"] + ([] if missing else ["**Overall**"]) \
            + [f"**{l}**" for _k, l in live]
        rows = []
        for a in order:
            row = [a] + ([] if missing else [f"{complete[a]:.3f}"])
            row += ["n/a" if scored[a][k] is None else f"{scored[a][k]:.3f}"
                    for k, _l in live]
            rows.append(row)
        lines.append(md_table(hdr, rows).rstrip())
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
        section_correctness(cells),
        section_by_domain(cells),
        section_noise(cells),
        section_signal_types(cells),
        section_tracking(cells),
        section_notes(cells),
        section_speed(cells),
        section_reliability(cells),
        section_datasets(repo_root),
        section_caveats(),
    ]
    with open(args.out, "w") as f:
        f.write("\n".join(p for p in parts if p))
    print(f"report -> {args.out}")
    if args.readme:
        update_readme_table(args.readme, cells)


if __name__ == "__main__":
    main()
