import argparse
import math
from pathlib import Path

import score


def corpora_rows(run, datasets):
    corpora = run["corpora"]
    rows = []
    for name in datasets:
        st = corpora[name]
        bands = ", ".join(f"{b} {v:.0f}%" for b, v in st["bands"].items()
                          if v >= 1.0)
        avg_len_s = st["hours"] * 3600.0 / max(st["clips"], 1)
        rows.append([name, st["clips"], f"{st['cal_clips']} / {st['cal_groups']}",
                     f"{avg_len_s:.1f}", f"{st['voiced_pct']:.0f}",
                     f"{st['f0_p5']:.0f}-{st['f0_p50']:.0f}-{st['f0_p95']:.0f}",
                     f"{st['fmin']:.0f}-{st['fmax']:.0f}",
                     f"{st['out_of_window_pct']:.1f}", bands])
    return rows


def md_table(rows, header):
    rows = [list(row) for row in rows]
    for col, label in enumerate(header):
        if "↑" not in label and "↓" not in label:
            continue
        values = {i: float(str(row[col]).split()[0]) for i, row in enumerate(rows)
                  if row[col] != "-"}
        if not values:
            continue
        best = (max if "↑" in label else min)(values.values())
        for i, value in values.items():
            if value == best:
                number, space, rest = str(rows[i][col]).partition(" ")
                rows[i][col] = f"**{number}**{space}{rest}"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def dash_row(algo, n):
    return [algo, *["-"] * n]


def _f(v, digits=3):
    return "-" if v is None or math.isnan(v) else f"{v:.{digits}f}"


def splice(text, tag, block):
    start_tag, end_tag = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    start, end = text.find(start_tag), text.find(end_tag)
    if text.count(start_tag) != 1 or text.count(end_tag) != 1 or end < start:
        raise ValueError(f"report requires one ordered pair of {start_tag} and {end_tag}")
    return text[:start + len(start_tag)] + "\n" + block + "\n" + text[end:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", default="cells")
    p.add_argument("--out", default="BENCHMARK.md")
    args = p.parse_args()
    document = Path(args.out).read_text()
    for tag in ("dataset-stats", "report"):
        splice(document, tag, "")

    run = score.load_run(args.cells)
    cells = score.load_cells(args.cells)
    res = score.score_all(cells, score.Design(**run["design"]))
    idx = score.index_cells(cells)
    algos, order = res["algos"], res["order"]
    incomplete = res["incomplete"]
    shown = {a: f"{a} (not installed)" if set(algos[a]["crash"]["kinds"]) == {"not installed"}
             else f"{a} (crashed)" for a in incomplete}
    design = res["design"]
    panels = list(design.scored)
    effects = design.effects
    speed_cells = {c["metadata"]["algorithm_name"]: c
                   for c in score.load_speed_cells(args.cells)}

    corpus_rows = corpora_rows(run, res["datasets"])
    dataset_parts = ["### Sample size\n"]
    dataset_parts.append(md_table([r[:4] for r in corpus_rows],
                                 ["Corpus", "Test clips", "Calibration clips / groups",
                                  "Mean clip length (s)"]))
    dataset_parts.append(
        "\nEach source clip is counted once, before rendering the recording conditions. "
        "Calibration clips in `valid/` come from separate speakers or recording groups "
        "and are used only to select the confidence threshold. Corpora without calibration clips "
        "use the threshold selected on the other corpora.\n")
    dataset_parts.append("### Pitch distribution\n")
    dataset_parts.append(md_table([[r[0], *r[4:]] for r in corpus_rows],
                                 ["Corpus", "Voiced (%)", "f0 p5 / p50 / p95 (Hz)",
                                  "Search range (Hz)", "Out of range (%)", "f0 by band"]))
    distribution = ", ".join(
        f"{band} {sum(run['corpora'][ds]['bands'].get(band, 0) for ds in res['datasets']) / len(res['datasets']):.1f}%"
        for band in ("bass", "low", "mid", "high", "vhigh"))
    dataset_parts.append(f"\nPitch bands across voiced frames, with equal weight per corpus: "
                         f"{distribution}.\n")
    dataset_parts.append(
        "Statistics describe the test clips. p5, p50 and p95 are the 5th, 50th and 95th pitch "
        "percentiles. Out of range is the percentage of verified voiced reference frames "
        "outside the search range. These frames are excluded from pitch F1. "
        "Bands: bass <80 Hz, "
        "low 80–260, mid 260–650, high 650–1050, vhigh ≥1050 Hz. Bands under 1% are omitted.\n")
    dataset_parts.append(
        "Trackers use the search range to restrict pitch candidates where supported. "
        "Otherwise, predictions outside the range are clamped to its nearest boundary.\n")

    parts = ["## Results\n", "### Overall performance\n"]
    dom = res["dominance"]
    rows = []
    for a in order:
        st = algos[a]
        lo, hi = st.get("ci", (None, None))
        ci = f"{st['score']:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else _f(st["score"])
        d = st.get("dominance") or {}
        rows.append([a, ci, _f(st["voicing_f1"]), d.get("beats", "-"),
                     d.get("loses_to", "-"), d.get("undetermined", "-")])
    rows += [dash_row(shown[a], 5) for a in incomplete]
    parts.append(md_table(rows, ["Tracker", f"Pitch F1@{score.CORRECT_CENTS}c ↑ [95% CI]", "Voicing F1 ↑", "Beats ↑",
                                 "Loses to ↓", "Undetermined"]))
    parts.append(
        "\nVoicing F1 measures voiced/unvoiced detection, regardless of pitch accuracy.\n")
    if dom["n_pairs"]:
        parts.append(
            "Brackets show 95% confidence intervals. Beats and loses to count statistically "
            "significant wins and losses after adjusting for all pairwise comparisons. "
            "Undetermined means the data do not resolve the difference.\n")
    else:
        parts.append(f"\nNo pairwise comparison: {len(order)} tracker(s) completed every cell.\n")

    for a in incomplete:
        cr = algos[a]["crash"]
        if set(cr["kinds"]) == {"not installed"}:
            parts.append(f"{a} was not installed and is unranked.\n")
            continue
        parts.append(
            f"{a} crashed in {cr['cells_crashed']} of {cr['cells']} runs and is unranked "
            "because its results are incomplete.\n")

    lost = [(a, algos[a]["crash"]) for a in order if algos[a]["crash"]["clips_crashed"]]
    if lost:
        parts.append("Clip failures, scored as missed voiced frames: "
                     + ", ".join(f"{a} {cr['clips_crashed']}/{cr['clips']}"
                                 for a, cr in lost) + ".\n")

    parts.append("### Performance by dataset\n")
    rows = [[a, *[_f(score.headline(
        {p: score.panel_score(idx, a, p, algos[a]["theta_idx"], [ds]) for p in panels}, design))
        for ds in res["datasets"]]] for a in order]
    rows += [dash_row(shown[a], len(res["datasets"])) for a in incomplete]
    parts.append(md_table(rows, ["Tracker", *[f"{ds} ↑" for ds in res["datasets"]]]))
    parts.append(
        f"\nPitch F1 averaged over the {len(panels)} scored conditions within each corpus, using the "
        "same threshold as the overall score. Rows follow the overall ranking.\n")

    parts.append("### Performance by recording condition\n")
    rows = [[a, _f(algos[a]["identity"]), *[_f(algos[a]["panels"][p]) for p in panels]]
            for a in order]
    rows += [dash_row(shown[a], 1 + len(panels)) for a in incomplete]
    parts.append(md_table(rows, ["Tracker", "Clean ↑",
                                 *[f"{design.labels[p]} ↑" for p in panels]]))
    parts.append(
        "\nPitch F1 averaged equally over the corpora. Clean (`identity`) is the unmodified "
        "audio and is excluded from the overall score. `level` applies only level normalization. "
        "All scored conditions use the same normalization.\n")

    parts.append("### Factor effects and interactions\n")
    rows = [[a, *[_f(algos[a]["effects"][e]) for e in effects]] for a in order]
    rows += [dash_row(shown[a], len(effects)) for a in incomplete]
    parts.append(md_table(rows, ["Tracker", *[" x ".join(e) for e in effects]]))
    parts.append(
        "\nThe scene, room and mic columns show the change in pitch F1 when that factor is "
        "enabled, averaged over all settings of the other factors. Negative values mean a loss. "
        "For pair interactions, a negative value means the combined loss exceeds the sum of "
        "the separate losses. A positive value means the combined loss is smaller. "
        "Pair interactions are averaged over both settings of the third factor. "
        "The three-factor interaction shows how the scene × room interaction changes "
        "when mic is enabled.\n")

    if any(algos[a].get("ladder") for a in order):
        parts.append("### Pitch accuracy\n")
        rungs = [f"<{c}c" for c in score.LADDER_CENTS]
        rows = []
        for a in order:
            lad = algos[a].get("ladder")
            if not lad:
                continue
            rows.append([a] + [f"{lad['rungs'][r] * 100:.1f}" for r in rungs]
                        + [f"{lad['octave_up'] * 100:.2f}", f"{lad['octave_down'] * 100:.2f}"])
        rows += [dash_row(shown[a], len(rungs) + 2) for a in incomplete]
        parts.append(md_table(rows, ["Tracker", *[f"{r} ↑" for r in rungs],
                                    "Octave up ↓", "Octave down ↓"]))
        parts.append(
            "\nPercentages among frames where both the tracker and reference are voiced and "
            "the reference pitch is verified and within the search range. Counts are pooled "
            "across all scored conditions and corpora at each tracker's selected threshold. "
            "Octave up and down count errors "
            f"{score.OCTAVE_BRACKET_CENTS[0]}–{score.OCTAVE_BRACKET_CENTS[1]} cents "
            "above and below the reference pitch, respectively.\n")
        if "BasicPitch" in order:
            parts.append("BasicPitch uses its note output, which has semitone resolution.\n")

    parts.append("## Properties\n")
    parts.append("Frame alignment and runtime do not contribute to the overall score.\n")

    if run.get("probe"):
        parts.append("### Frame alignment (chirp probe, ms)\n")
        offsets = {a: max((abs(b["offset_ms"]) for b in run["probe"].get(a, [])
                           if b["offset_ms"] is not None), default=None)
                   for a in [*order, *incomplete]}
        alignment_order = sorted(offsets, key=lambda a: (
            math.inf if offsets[a] is None else offsets[a], a))
        rows = [[shown.get(a, a), _f(offsets[a], 2)] for a in alignment_order]
        parts.append(md_table(rows, ["Tracker", "Worst measured alignment error (ms) ↓"]))
        parts.append(
            "\nLargest absolute time offset measured across chirp bands. A larger offset "
            "(for example, above 2 ms) can indicate an algorithm error or a model that learned "
            "to place pitch estimates too early or too late from misaligned training labels.\n")

    parts.append("### Speed\n")
    speeds = {a: score.speed_factor(speed_cells[a]) if a in speed_cells else None
            for a in [*order, *incomplete]}
    speed_order = sorted(speeds, key=lambda a: (math.inf if speeds[a] is None else -speeds[a], a))
    rows = [[shown.get(a, a), _f(speeds[a], 1)] for a in speed_order]
    parts.append(md_table(rows, ["Tracker", "Speed (× real time) ↑"]))
    machines = {c["parameters"].get("cpu") for c in speed_cells.values()
                if c["parameters"].get("cpu")}
    parts.append(
        "\nAudio duration divided by median processing time after warm-up: 20× means "
        "20 seconds of audio processed per second. "
        "Speed is measured separately by [speed.py](speed.py), including for trackers with "
        "incomplete accuracy results. CPU: "
        + (", ".join(sorted(machines)) if machines else "an unrecorded cpu")
        + ".\n")

    document = splice(document, "dataset-stats", "\n".join(dataset_parts))
    document = splice(document, "report", "\n".join(parts))
    Path(args.out).write_text(document)
    print(f"updated {args.out}")


if __name__ == "__main__":
    main()
