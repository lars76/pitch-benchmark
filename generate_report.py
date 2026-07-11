#!/usr/bin/env python3
"""Generate a markdown report (tables + insights) from pitch detection benchmark results."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets.augment import (
    CONDITION_FAMILIES,  # single source of truth for condition families
)
from metrics import PITCH_BANDS, band_label


def _is_bad(v):
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table from header strings + rows (each a list of cell strings)."""
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "---|" * len(headers) + "\n"
    for row in rows:
        out += "| " + " | ".join(row) + " |\n"
    return out


def fmt_best(value, column_values, fmt="{:.3f}", lower=False, na="N/A") -> str:
    """Format `value`, bolding it if it is the best (max, or min if lower) of column_values."""
    if _is_bad(value):
        return na
    valid = [v for v in column_values if not _is_bad(v)]
    best = (min if lower else max)(valid) if valid else None
    s = fmt.format(value)
    if best is not None and abs(value - best) < 1e-9:
        s = f"**{s}**"
    return s


def load_all_results(results_dir: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Load all JSON result files, split into (pitch, speed, ood) by benchmark_type."""
    pitch_results = []
    speed_results = []
    ood_results = []

    for file_path in Path(results_dir).glob("*.json"):
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Route by benchmark_type FIRST: OOD cells also carry combined_score, so they must be
            # split off here or they would pollute the accuracy leaderboard.
            bt = data.get("metadata", {}).get("benchmark_type")
            if bt == "speed":
                speed_results.append(data)
            elif bt == "ood":
                ood_results.append(data)
            elif "results" in data and "combined_score" in data.get("results", {}):
                pitch_results.append(data)
            else:
                print(f"Warning: Unrecognized result format in {file_path.name}")

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return pitch_results, speed_results, ood_results


def _dedupe_prefer_cpu(results: list[dict], key_fn) -> list[dict]:
    """Keep one result per key_fn(metadata, parameters). A non-crashed result always beats a crashed
    one for the same cell -- a valid result on disk must not be shadowed by a stale crash cell (e.g. an
    `_auto_` cell written before a backend was installed, which lingers under a different device string
    than the later `_mps_`/`_cuda_` result). Among non-crashed results prefer `device=="cpu"` (the
    reproducible reference), else whatever ran; a crash is kept only if EVERY cell for the key crashed
    (it then counts as 0). Device lives in the filename so cpu/gpu results coexist on disk."""
    def rank(r):   # higher wins: (valid over crashed, then cpu over gpu)
        m = r.get("metadata", {})
        return (not m.get("crashed"), m.get("device") == "cpu")

    chosen: dict = {}
    for r in results:
        k = key_fn(r.get("metadata", {}), r.get("parameters", {}))
        cur = chosen.get(k)
        if cur is None or rank(r) > rank(cur):
            chosen[k] = r
    return list(chosen.values())


def _pitch_key(m, p):
    # Everything the result filename encodes EXCEPT device, so only device variants collapse (a
    # different seed/sample-rate/hop/probe-cap stays its own row).
    return (m.get("algorithm_name"), m.get("dataset_name"), m.get("condition", "clean"),
            bool(m.get("probe")), m.get("seed"),
            p.get("sample_rate"), p.get("hop_size"), p.get("max_samples"), p.get("max_seconds"))


def _ood_key(m, _p):
    return (m.get("algorithm_name"), m.get("family"))


# Synthetic OOD families (kept in sync with ood_benchmark.py FAMILIES): two axes.
OOD_MECHANISMS = ["missing_f0", "unresolved", "irn", "vibrato_fast", "glide"]
OOD_MECH_AXIS = {
    "missing_f0": "fundamental presence (no energy at f0)",
    "unresolved": "harmonic resolvability (high harmonics only)",
    "irn": "noise-based periodicity (rippled noise)",
    "vibrato_fast": "temporal dynamics (fast FM)",
    "glide": "temporal dynamics (monotonic octave sweep)",
}
OOD_RANGE_SIGNALS = ["sine", "harm", "tilt"]          # family-name prefixes of the pitch-range axis
OOD_SIGNAL_LABEL = {
    "sine": "sine (pure tone)", "harm": "harmonic (normal tone)", "tilt": "tilt (bright)",
}
OOD_BANDS = ["bass", "low", "mid", "high", "vhigh"]   # range-table column order
OOD_CONTROL_ORDER = ["noise", "whisper"]


def _ood_summary_vals(cells, fams):
    """Values feeding the mean/worst summary. A crashed cell (present but None) counts as **0.0** - a
    crash is a total failure on that signal (the process dies), so the summary must not hide it. A
    family never run (absent from `cells`) is dropped. Per-cell rendering still shows N/A for crashes,
    so 'crashed' and 'ran but scored 0' stay distinguishable in the detail row."""
    out = []
    for f in fams:
        if f in cells:
            v = cells[f]
            out.append(0.0 if _is_bad(v) else float(v))
    return out


def _ood_collect(ood_results):
    """-> (voiced[algo][family]=acc|None, octv[algo][family]=octave_rate, control[algo][family]=fp|None)."""
    voiced, octv, control = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for r in ood_results:
        m, res = r.get("metadata", {}), r.get("results", {})
        algo, fam, ftype = m.get("algorithm_name"), m.get("family"), m.get("family_type")
        if not algo or not fam:
            continue
        if ftype == "control":
            control[algo][fam] = None if m.get("crashed") else res.get("false_positive_rate")
        elif m.get("crashed"):
            voiced[algo][fam] = None                          # N/A = crashed on this signal
        else:
            acc = res.get("ood_accuracy")
            voiced[algo][fam] = 0.0 if _is_bad(acc) else acc  # ran-but-untrackable is a real 0
            octv[algo][fam] = (res.get("pitch_accuracy") or {}).get("octave_error_rate")
    return voiced, octv, control


def _ood_mechanism_section(voiced) -> str:
    fams = [f for f in OOD_MECHANISMS if any(f in voiced[a] for a in voiced)]
    summ = {a: (float(np.mean(v)), float(min(v)))
            for a in voiced for v in [_ood_summary_vals(voiced[a], fams)] if v}
    if not summ:
        return ""
    order = sorted(summ, key=lambda a: -summ[a][0])
    means, worsts = [summ[a][0] for a in order], [summ[a][1] for a in order]
    col_vals = {f: [voiced[a].get(f) for a in order] for f in fams}
    headers = ["**Algorithm**", "**mean** ↑", "**worst** ↑"] + [f"**{f}**" for f in fams]
    rows = [[a, fmt_best(summ[a][0], means, "{:.2f}"), fmt_best(summ[a][1], worsts, "{:.2f}")]
            + [fmt_best(voiced[a].get(f), col_vals[f], "{:.2f}") for f in fams] for a in order]
    return (
        "### OOD mechanisms (spectral structure; probed at low-mid pitch)\n\n"
        + md_table(headers, rows) + "\n"
        + "A **crash counts as 0** in mean/worst (a crash is a total failure); the detail cell stays "
        + "`N/A`. Axis: " + "; ".join(f"`{f}` = {OOD_MECH_AXIS[f]}" for f in fams) + "\n\n"
    )


def _ood_range_section(voiced, octv) -> str:
    present = {s: [b for b in OOD_BANDS if any(f"{s}_{b}" in voiced[a] for a in voiced)]
               for s in OOD_RANGE_SIGNALS}
    present = {s: b for s, b in present.items() if b}
    if not present:
        return ""
    out = (
        "### Pitch-range coverage (one tone swept across bands)\n\n"
        "F0 accuracy per pitch band for a tone swept bass->vhigh; `floor`/`ceil` = lowest/highest band "
        "with accuracy >= 0.5 (the tracker's usable register). This fills the high/very-high registers "
        "the real datasets (all bass/low/mid) never exercise. Comparing **sine vs harmonic** separates "
        "a true range limit (both fail) from harmonic-dependence (sine fails, harmonic holds).\n\n"
    )
    for sig, bands in present.items():
        def _m(a, sig=sig, bands=bands):
            v = _ood_summary_vals(voiced[a], [f"{sig}_{b}" for b in bands])
            return np.mean(v) if v else float("nan")
        algos = [a for a in voiced if any(f"{sig}_{b}" in voiced[a] for b in bands)]
        algos.sort(key=lambda a: (np.isnan(_m(a)), -(0.0 if np.isnan(_m(a)) else _m(a))))
        col_vals = {b: [voiced[a].get(f"{sig}_{b}") for a in algos] for b in bands}
        headers = ["**Algorithm**"] + [f"**{b}**" for b in bands] + ["**floor**", "**ceil**"]
        rows = []
        for a in algos:
            cells = [fmt_best(voiced[a].get(f"{sig}_{b}"), col_vals[b], "{:.2f}") for b in bands]
            ok = [b for b in bands
                  if isinstance(voiced[a].get(f"{sig}_{b}"), (int, float)) and voiced[a][f"{sig}_{b}"] >= 0.5]
            rows.append([a, *cells, ok[0] if ok else "-", ok[-1] if ok else "-"])
        out += f"**{OOD_SIGNAL_LABEL[sig]}**\n\n" + md_table(headers, rows) + "\n"
    return out + _ood_range_notes(voiced, octv, present)


def _ood_range_notes(voiced, octv, present) -> str:
    out = ""
    for b in [x for x in ("high", "vhigh") if x in present.get("sine", []) and x in present.get("harm", [])]:
        rng, tim = [], []
        for a in sorted(voiced):
            sv, hv = voiced[a].get(f"sine_{b}"), voiced[a].get(f"harm_{b}")
            if not (isinstance(sv, (int, float)) and isinstance(hv, (int, float))):
                continue
            if sv < 0.5 and hv < 0.5:
                rng.append(a)
            elif sv < 0.5 <= hv:
                tim.append(a)
        parts = []
        if rng:
            parts.append(f"true range ceiling (sine and harmonic both fail): {', '.join(rng)}")
        if tim:
            parts.append(f"harmonic-dependent (sine fails, harmonic holds): {', '.join(tim)}")
        if parts:
            out += f"At **{b}** - " + "; ".join(parts) + ".\n\n"
    oct_hi = {}
    for a in octv:
        vals = [octv[a].get(f"{s}_{b}") for s in OOD_RANGE_SIGNALS for b in ("high", "vhigh")
                if isinstance(octv[a].get(f"{s}_{b}"), (int, float))]
        if vals and np.mean(vals) > 0.1:
            oct_hi[a] = float(np.mean(vals))
    if oct_hi:
        out += ("High-band failures that are **octave errors** (reports f0/2), not silence: "
                + ", ".join(f"{a} ({oct_hi[a]:.2f})" for a in sorted(oct_hi, key=lambda a: -oct_hi[a]))
                + ".\n\n")
    return out


def _ood_controls_section(control) -> str:
    fams = [f for f in OOD_CONTROL_ORDER if any(f in control[a] for a in control)]
    if not fams:
        return ""
    def _cm(a):                                                # lower false-positive rate = better
        vs = [v for v in (control[a].get(f) for f in fams) if not _is_bad(v)]
        return float(np.mean(vs)) if vs else float("inf")
    order = sorted(control, key=_cm)
    col_vals = {f: [control[a].get(f) for a in order] for f in fams}
    headers = ["**Algorithm**"] + [f"**{f}**" for f in fams]
    rows = [[a] + [fmt_best(control[a].get(f), col_vals[f], "{:.2f}", lower=True) for f in fams]
            for a in order]
    return (
        "### Unvoiced controls (false-positive rate)\n\n"
        "Correct answer is no pitch; value is the fraction of frames wrongly voiced (lower is better).\n\n"
        + md_table(headers, rows) + "\n"
    )


def generate_ood_analysis(ood_results: list[dict]) -> str:
    """OOD section: a spectral-mechanism matrix + a pitch-range matrix + unvoiced controls."""
    if not ood_results:
        return ""
    voiced, octv, control = _ood_collect(ood_results)
    out = "## Generalization / OOD (synthetic signals)\n\n"
    out += (
        "Out-of-distribution probe on synthetic signals with **exact labels** (no pitch detector in the "
        "label loop). **F0 accuracy** = fraction of ground-truth-voiced frames the tracker both voices "
        "AND places within 50 cents of the known f0 (drops for wrong pitch *and* for refusing to voice); "
        "below 0.50 = cannot follow. `N/A` = the tracker crashed on that signal (recorded, not fatal). "
        "Two axes below: spectral **mechanisms** and **pitch range**.\n\n"
    )
    return out + _ood_mechanism_section(voiced) + _ood_range_section(voiced, octv) \
        + _ood_controls_section(control)


def aggregate_pitch_results(
    pitch_results: list[dict],
) -> dict[str, dict[str, list[float]]]:
    """Aggregate pitch results by algorithm and dataset.

    A recorded crash (metadata.crashed) counts as 0.0 -- a crash is a total failure, not missing data,
    so it is scored, not dropped (same policy as the OOD table; shared metadata.crashed convention with
    pitch_benchmark/ood_benchmark). A non-crashed NaN (a deterministically empty run, e.g. all frames
    unvoiced) is genuinely no data and is dropped."""
    # Structure: {algorithm: {dataset: [scores]}}
    aggregated = defaultdict(lambda: defaultdict(list))

    for result in pitch_results:
        try:
            algo_name = result["metadata"]["algorithm_name"]
            dataset_name = result["metadata"]["dataset_name"]

            if result["metadata"].get("crashed"):
                aggregated[algo_name][dataset_name].append(0.0)
                continue

            combined_score = result["results"]["combined_score"]
            if combined_score is not None and not np.isnan(combined_score):
                aggregated[algo_name][dataset_name].append(combined_score)
        except KeyError as e:
            print(f"Warning: Missing key in pitch result: {e}")
            continue

    return aggregated


def aggregate_speed_results(
    speed_results: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate speed results by algorithm."""
    # Structure: {algorithm: {metric: value}}
    aggregated = {}

    for result in speed_results:
        try:
            algo_name = result["metadata"]["algorithm_name"]
            cpu_perf = result["results"]["device_performance"].get("cpu", {})

            if cpu_perf.get("supported", False):
                aggregated[algo_name] = {
                    "absolute_time_ms": cpu_perf["absolute_time_ms"],
                    "relative_speed": cpu_perf.get("relative_speed", 1.0),
                    "supports_cuda": result["results"]["supports_cuda"],
                }
        except KeyError as e:
            print(f"Warning: Missing key in speed result: {e}")
            continue

    return aggregated


def collect_detailed_metrics(
    pitch_results: list[dict],
) -> dict[str, dict[str, list[float]]]:
    """Collect detailed metrics for in-depth analysis."""
    # Structure: {algorithm: {metric: [values]}}
    metrics = defaultdict(lambda: defaultdict(list))

    for result in pitch_results:
        try:
            algo_name = result["metadata"]["algorithm_name"]
            results_data = result["results"]

            # Voicing detection metrics
            voicing = results_data.get("voicing_detection", {})
            for metric in ["precision", "recall", "f1"]:
                if metric in voicing and voicing[metric] is not None:
                    metrics[algo_name][f"voicing_{metric}"].append(
                        voicing[metric]
                    )

            # Pitch accuracy metrics
            pitch = results_data.get("pitch_accuracy", {})
            for metric in [
                "rpa",
                "rca",
                "cents_error",
                "rmse",
                "octave_error_rate",
                "gross_error_rate",
            ]:
                if metric in pitch and pitch[metric] is not None:
                    metrics[algo_name][f"pitch_{metric}"].append(pitch[metric])

            # Smoothness metrics
            smoothness = results_data.get("smoothness_metrics", {})
            for metric in ["relative_smoothness", "continuity_breaks"]:
                if metric in smoothness and smoothness[metric] is not None:
                    metrics[algo_name][metric].append(smoothness[metric])

            # Optimal threshold
            if (
                "optimal_threshold" in results_data
                and results_data["optimal_threshold"] is not None
            ):
                metrics[algo_name]["optimal_threshold"].append(
                    results_data["optimal_threshold"]
                )

            # Combined score for consistency analysis
            if (
                "combined_score" in results_data
                and results_data["combined_score"] is not None
            ):
                metrics[algo_name]["combined_score"].append(
                    results_data["combined_score"]
                )

            # Execution time
            if "execution_time_seconds" in result["metadata"]:
                metrics[algo_name]["execution_time"].append(
                    result["metadata"]["execution_time_seconds"]
                )

        except KeyError as e:
            print(f"Warning: Missing key in detailed metrics: {e}")
            continue

    return metrics


def generate_dataset_descriptions() -> str:
    """Generate dataset descriptions section."""
    descriptions = "## Dataset Descriptions\n\n"

    descriptions += "The benchmark evaluates algorithms across diverse datasets covering speech, music, synthetic, and real-world conditions:\n\n"

    descriptions += (
        "| **Dataset** | **Domain** | **Type** | **Description** |\n"
    )
    descriptions += "|---|---|---|---|\n"

    dataset_info = [
        (
            "NSynth",
            "Music",
            "Synthetic",
            "Single-note synthetic audio from musical instruments with accurate pitch labels. Inharmonic families (organ, mallet, synth_lead) and multiphonic-tagged notes are excluded by default, since their MIDI pitch is not a reliable single-f0 ground truth. Lacks temporal/spectral complexity of real-world environments.",
        ),
        (
            "PTDB",
            "Speech",
            "Real",
            "Speech recordings with simultaneous laryngograph (EGG). Ground truth is a consensus over three independent EGG estimators (Praat, dEGG, Harvest), keeping frames where at least two agree within 50 cents and gating silence by mic energy. Frames where estimators voice but disagree on the f0 value are kept voiced (counted in F1) but excluded from pitch accuracy.",
        ),
        (
            "MIR1K",
            "Music",
            "Real",
            "Vocal excerpts with pitch contours initially extracted algorithmically (e.g., YIN) followed by manual correction. Labels still reflect some algorithmic biases.",
        ),
        (
            "MDBStemSynth",
            "Music",
            "Synthetic",
            "Musically structured synthetic audio with accurate pitch annotations. Valuable for controlled evaluation but lacks real-world acoustic variability.",
        ),
        (
            "Vocadito",
            "Music",
            "Real",
            "Solo vocal recordings with pitch annotations derived from pYIN algorithm, refined through manual verification process.",
        ),
        (
            "Bach10Synth",
            "Music",
            "Synthetic",
            "High-quality pitch labels for synthesized musical performances. Similar to MDB-STEM-Synth but focused on Bach compositions.",
        ),
        (
            "SpeechSynth",
            "Speech",
            "Synthetic",
            "Synthetic Mandarin speech generated using LightSpeech TTS. The conditioned f0 is rendered by a neural vocoder and is faithful for smooth contours, so labels are accurate to within a few tens of cents (near-exact, not exact to the cent).",
        ),
        (
            "MOCHA",
            "Speech",
            "Real",
            "MOCHA-TIMIT speech with a simultaneous laryngograph (.lar) channel. Consensus f0 on the EGG (same three-estimator scheme as PTDB); no shipped author f0.",
        ),
        (
            "CMUArctic",
            "Speech",
            "Real",
            "CMU ARCTIC prompts recorded with simultaneous EGG (stereo speech+EGG). Consensus f0 on the EGG; no shipped author f0.",
        ),
        (
            "AVID",
            "Speech",
            "Real",
            "Aalto Vocal Intensity Database: calibrated stereo speech+EGG recordings at graded vocal intensities. Consensus f0 on the EGG; no shipped author f0.",
        ),
        (
            "OSFGlottis",
            "Speech",
            "Real",
            "OSF 'Physical Models of the Glottis' subjects reading Harvard sentences with simultaneous EGG (10 kHz). Consensus f0 on the EGG; no shipped author f0.",
        ),
        (
            "SVD",
            "Speech",
            "Real",
            "Saarbruecken Voice Database (healthy subset): German connected-speech phrases with simultaneous EGG. Consensus f0 on the EGG; no shipped author f0.",
        ),
        (
            "APLAWD",
            "Speech",
            "Real",
            "APLAWD speech with a simultaneous laryngograph (.egg) channel. Consensus f0 on the EGG; no shipped author f0.",
        ),
        (
            "KEELE",
            "Speech",
            "Real",
            "KEELE pitch database: speech with simultaneous EGG. Consensus f0 on the EGG by default; also ships an author reference (label_source=\"reference\").",
        ),
        (
            "FDA",
            "Speech",
            "Real",
            "FDA (Edinburgh) speech with simultaneous EGG. Consensus f0 on the EGG by default; also ships an author reference (label_source=\"reference\").",
        ),
        (
            "URMP",
            "Music",
            "Real",
            "URMP multi-instrument recordings with gold frame-level f0 and note (Notes_*.txt) annotations per stem. Music reference, no EGG.",
        ),
    ]

    for dataset, domain, type_str, desc in dataset_info:
        descriptions += f"| **{dataset}** | {domain} | {type_str} | {desc} |\n"

    # Corpus statistics: measured once (exact full pass, sr 16 kHz, hop 256) and assumed stable, so
    # the report never needs the raw datasets. f0 stats are over in-window voiced frames; band
    # coverage = % of voiced frames per band. If the datasets change, re-measure with
    # `uv run python scripts/dataset_stats.py` and paste its `corpus_stats` block over the list below.
    descriptions += "\n### Corpus Statistics\n\n"
    descriptions += (
        "*f0 in Hz over in-window voiced frames; band coverage = % of voiced frames per "
        "band (bass <80, low 80-260, mid 260-650, high 650-1050, vhigh >=1050 Hz).*\n\n"
    )
    descriptions += (
        "| **Dataset** | **Clips** | **Hours** | **Avg len (s)** | **Voiced %** | "
        "**f0 p5-p50-p95** | **Band coverage** |\n"
    )
    descriptions += "|---|--:|--:|--:|--:|---|---|\n"
    corpus_stats = [
        ("NSynth", 3319, 3.7, 4.0, 53, "73-277-1480", "bass 8, low 39, mid 29, high 12, vhigh 12"),
        ("PTDB", 4718, 9.6, 7.3, 25, "84-155-234", "bass 3, low 96, mid 1"),
        ("MIR1K", 1000, 2.2, 8.0, 70, "121-231-390", "low 63, mid 37"),
        ("MDBStemSynth", 230, 15.6, 243.6, 40, "73-209-704", "bass 11, low 50, mid 32, high 6"),
        ("Vocadito", 40, 0.2, 20.4, 66, "113-219-364", "low 69, mid 31"),
        ("Bach10Synth", 40, 0.4, 33.4, 92, "111-296-579", "low 38, mid 59, high 2"),
        ("SpeechSynth", 219, 0.1, 1.9, 48, "125-211-272", "low 91, mid 9"),
    ]
    for name, clips, hours, avglen, voiced, f0r, bands in corpus_stats:
        descriptions += (
            f"| **{name}** | {clips} | {hours} | {avglen} | {voiced} | {f0r} | {bands} |\n"
        )
    descriptions += (
        "\nThe high/vhigh bands are populated almost entirely by NSynth; the speech sets "
        "(PTDB, SpeechSynth) sit almost entirely in the low band.\n"
    )

    descriptions += "\n**Key Characteristics:**\n"
    descriptions += "- **Synthetic datasets** provide highly precise ground truth (exact by construction for the DSP-resynthesized sets; within ~tens of cents for the neural-vocoded SpeechSynth) but may lack real-world complexity\n"
    descriptions += "- **Real datasets** capture natural acoustic variations but have imperfect ground truth annotations\n"
    descriptions += (
        "- **Speech datasets** focus on vocal pitch tracking challenges\n"
    )
    descriptions += "- **Music datasets** encompass instrumental and vocal music scenarios\n"
    descriptions += "- **SpeechSynth** provides synthetic speech with vocoder-rendered pitch labels faithful to within a few tens of cents; treat smaller RPA/cents gaps on it as within that rendering floor\n\n"

    return descriptions


def generate_methodology_section() -> str:
    """Generate methodology and metric definitions section."""
    methodology = "## Benchmark Methodology\n\n"

    methodology += "### Evaluation Setup\n"
    methodology += "Algorithms are evaluated on **clean** audio across multiple datasets (synthetic and real, "
    methodology += "speech and music). This is the headline leaderboard. **Robustness** is measured separately on a "
    methodology += "small deterministic probe (a capped, duration-truncated sample across datasets) run through each "
    methodology += "degradation, and reported as Δ-from-clean per condition. Per-pitch-band metrics (RPA, octave/gross "
    methodology += "error) are reported by ground-truth f0 band. Noise/degradations are a pure function of (seed, index), "
    methodology += "so results are reproducible regardless of run order.\n\n"

    methodology += "### Performance Metric Definition\n"
    methodology += "The **Overall Performance Rankings** show the **Harmonic Mean (HM)** score as percentages, computed from six complementary components:\n\n"
    methodology += "**HM = 6 / (1/RPA + 1/CA + 1/P + 1/R + 1/OA + 1/GEA)**\n\n"
    methodology += "Where:\n"
    methodology += "- **RPA** (Raw Pitch Accuracy): fraction of frames within 50 cents of ground truth, scored where **both** the ground truth and the algorithm are voiced\n"
    methodology += "- **CA** (Cents Accuracy): exp(-mean_cents_error/500), penalizing larger deviations exponentially\n"
    methodology += "- **P** (Voicing Precision): TP/(TP+FP), fraction of predicted voiced frames that are truly voiced\n"
    methodology += "- **R** (Voicing Recall): TP/(TP+FN), fraction of truly voiced frames detected\n"
    methodology += "- **OA** (Octave Accuracy): exp(-10×octave_error_rate), robustness against octave errors\n"
    methodology += "- **GEA** (Gross Error Accuracy): exp(-5×gross_error_rate), penalizing deviations >200 cents\n\n"
    methodology += (
        "Zero or undefined components are floored to a small ε (not dropped), so the score "
        "collapses toward 0 if an algorithm fully fails on any one axis, as a true harmonic mean should.\n\n"
    )

    methodology += (
        "Pitch (RPA/CA/octave/gross) is scored only where both agree on voicing, not on every "
        "ground-truth-voiced frame, because (1) where the ground-truth voicing is wrong, scoring every "
        "reference frame penalizes a correct algorithm, and (2) some algorithms do not output a pitch on "
        "every frame. Voicing quality is captured separately by P/R/F1.\n\n"
    )

    methodology += "### Speed Benchmark Details\n"
    methodology += "CPU timing measurements are performed on 1-second audio signals at 22.05 kHz sample rate with 256-sample hop length. "
    methodology += "The reported **CPU Time (ms)** represents the average processing time per 1-second audio segment across multiple runs. "
    methodology += "**Relative Speed** shows performance relative to CREPE as the baseline algorithm.\n\n"

    methodology += "### Optimal Threshold Analysis\n"
    methodology += "The **Optimal Threshold** refers to the voicing confidence threshold that maximizes the Harmonic Mean score. "
    methodology += "Algorithms test multiple thresholds (0.0 to 1.0 in steps of 0.1) and select the one yielding the highest combined score. "
    methodology += "**CV** stands for Coefficient of Variation (std/mean), measuring consistency across datasets.\n\n"

    return methodology


def generate_combined_score_table(
    aggregated_results: dict[str, dict[str, list[float]]],
    robustness: dict[str, float] | None = None,
) -> str:
    """Generate the main performance table showing combined scores as percentages."""
    if not aggregated_results:
        return "No pitch benchmark results found.\n"

    all_datasets = sorted({d for a in aggregated_results.values() for d in a})

    # Per-algorithm percentage score per dataset (rounded to .1 so display ties bold together).
    table_data = []
    for algo in sorted(aggregated_results):
        scores = {
            d: (round(float(np.mean(aggregated_results[algo][d])) * 100, 1)
                if aggregated_results[algo].get(d) else None)
            for d in all_datasets
        }
        present = [v for v in scores.values() if v is not None]
        avg = round(float(np.mean(present)), 1) if present else None
        table_data.append({"algo": algo, "scores": scores, "avg": avg})

    table_data.sort(key=lambda r: (r["avg"] is None, -(r["avg"] or 0)))

    col_vals = {d: [r["scores"][d] for r in table_data] for d in all_datasets}
    avgs = [r["avg"] for r in table_data]
    robust_vals = (
        [round(robustness.get(r["algo"]), 1) if robustness.get(r["algo"]) is not None else None
         for r in table_data]
        if robustness else None
    )

    headers = (
        ["**Algorithm**"]
        + [f"**{d}**" for d in all_datasets]
        + ["**Average**"]
        + (["**Robust Δ ↓**"] if robustness else [])
    )
    rows = []
    for i, r in enumerate(table_data):
        name = f"**{r['algo']}**" if i == 0 else r["algo"]  # best overall (sorted first)
        cells = [name] + [fmt_best(r["scores"][d], col_vals[d], "{:.1f}%") for d in all_datasets]
        cells.append(fmt_best(r["avg"], avgs, "{:.1f}%"))
        if robustness:
            rv = robustness.get(r["algo"])
            rv = round(rv, 1) if rv is not None else None
            cells.append(fmt_best(rv, robust_vals, "{:.1f}", lower=True))
        rows.append(cells)

    return "## Overall Performance Rankings\n\n" + md_table(headers, rows) + "\n"


def generate_speed_table(speed_results: dict[str, dict[str, float]]) -> str:
    """Generate CPU speed performance table."""
    if not speed_results:
        return "No speed benchmark results found.\n"

    # Sort by absolute time (ascending - faster is better)
    sorted_algos = sorted(
        speed_results.items(), key=lambda x: x[1]["absolute_time_ms"]
    )
    times = [m["absolute_time_ms"] for _, m in sorted_algos]
    speeds = [m["relative_speed"] for _, m in sorted_algos]

    rows = []
    for algo_name, metrics in sorted_algos:
        name = f"**{algo_name}**" if metrics is sorted_algos[0][1] else algo_name
        rows.append(
            [
                name,
                fmt_best(metrics["absolute_time_ms"], times, "{:.1f}", lower=True),
                fmt_best(metrics["relative_speed"], speeds, "{:.2f}x"),
            ]
        )

    return (
        "## Speed Performance (CPU)\n\n"
        + md_table(
            ["**Algorithm**", "**CPU Time (ms) ↓**", "**Relative Speed ↑**"], rows
        )
        + "\n"
    )


def _avg(d, key):
    vals = [v for v in d.get(key, []) if not _is_bad(v)]
    return float(np.mean(vals)) if vals else None


def _cv(values):
    """Coefficient of variation (std/mean); None if < 2 values or non-positive mean."""
    v = [x for x in values if not _is_bad(x)]
    return float(np.std(v)) / float(np.mean(v)) if len(v) > 1 and np.mean(v) > 0 else None


def _metric_table(detailed, title, desc, columns):
    """Render a per-algorithm metric table. columns: (header, value_fn, fmt, lower)."""
    algos = sorted(detailed)
    col_vals = {h: [fn(detailed[a]) for a in algos] for h, fn, _, _ in columns}
    headers = ["**Algorithm**"] + [f"**{h}**" for h, _, _, _ in columns]
    rows = [
        [a] + [fmt_best(fn(detailed[a]), col_vals[h], fmt, lower=lo) for h, fn, fmt, lo in columns]
        for a in algos
    ]
    return f"### {title}\n{desc}\n\n" + md_table(headers, rows) + "\n"


def generate_detailed_analysis(
    detailed_metrics: dict[str, dict[str, list[float]]],
) -> str:
    """Detailed per-algorithm metric tables (averaged across datasets)."""
    if not detailed_metrics:
        return "No detailed metrics available.\n"

    out = "## Detailed Performance Analysis\n\n"
    out += _metric_table(
        detailed_metrics,
        "Voicing Detection Performance",
        "How well algorithms distinguish voiced (pitched) from unvoiced frames.",
        [
            ("Precision ↑", lambda d: _avg(d, "voicing_precision"), "{:.3f}", False),
            ("Recall ↑", lambda d: _avg(d, "voicing_recall"), "{:.3f}", False),
            ("F1 ↑", lambda d: _avg(d, "voicing_f1"), "{:.3f}", False),
        ],
    )
    out += _metric_table(
        detailed_metrics,
        "Pitch Accuracy Metrics",
        "Pitch-estimation accuracy across error types.",
        [
            ("RPA ↑", lambda d: _avg(d, "pitch_rpa"), "{:.3f}", False),
            ("RCA ↑", lambda d: _avg(d, "pitch_rca"), "{:.3f}", False),
            ("Cents Error ↓", lambda d: _avg(d, "pitch_cents_error"), "{:.1f}", True),
            ("RMSE (Hz) ↓", lambda d: _avg(d, "pitch_rmse"), "{:.1f}", True),
            ("Octave Error ↓", lambda d: _avg(d, "pitch_octave_error_rate"), "{:.3f}", True),
            ("Gross Error ↓", lambda d: _avg(d, "pitch_gross_error_rate"), "{:.3f}", True),
        ],
    )
    out += _metric_table(
        detailed_metrics,
        "Pitch Contour Smoothness",
        "Temporal stability of the pitch track (diagnostic).",
        [
            ("Relative Smoothness ↓", lambda d: _avg(d, "relative_smoothness"), "{:.3f}", True),
            ("Continuity Breaks ↓", lambda d: _avg(d, "continuity_breaks"), "{:.3f}", True),
        ],
    )

    # Threshold table: mean (plain), std (bold-best), range (string).
    algos = sorted(detailed_metrics)
    stds = []
    for a in algos:
        thr = [t for t in detailed_metrics[a].get("optimal_threshold", []) if not _is_bad(t)]
        stds.append(float(np.std(thr)) if thr else None)
    rows = []
    for a in algos:
        thr = [t for t in detailed_metrics[a].get("optimal_threshold", []) if not _is_bad(t)]
        if thr:
            rows.append([
                a,
                f"{np.mean(thr):.3f}",
                fmt_best(float(np.std(thr)), stds, "{:.3f}", lower=True),
                f"{min(thr):.2f}-{max(thr):.2f}",
            ])
        else:
            rows.append([a, "N/A", "N/A", "N/A"])
    out += (
        "### Optimal Threshold Analysis\n"
        "Voicing-confidence threshold that maximizes the combined score.\n\n"
        + md_table(
            ["**Algorithm**", "**Mean Threshold**", "**Std Dev ↓**", "**Range**"], rows
        )
        + "\n"
    )

    out += _metric_table(
        detailed_metrics,
        "Algorithm Consistency",
        "Stability across datasets (CV = std/mean, lower = more consistent).",
        [
            ("Performance CV ↓", lambda d: _cv(d.get("combined_score", [])), "{:.3f}", True),
            ("Threshold CV ↓", lambda d: _cv(d.get("optimal_threshold", [])), "{:.3f}", True),
        ],
    )
    return out


# Dataset groupings for the subset analysis (single source; descriptions derived from this).
# The EGG (laryngograph) speech corpora are all Real + Speech; URMP is Real + Music. Keep these in
# sync with datasets/__init__._PITCH_REGISTRY -- _check_group_coverage() below warns on any registered
# dataset that appears in the results but is missing from every group here (a silent-omission guard).
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


def _check_group_coverage(aggregated_results: dict) -> None:
    """Warn if any dataset present in the results is not covered by DATASET_GROUPS, so a newly added
    corpus can never be silently dropped from the subset analysis (the bug this guards against)."""
    grouped = {d for title in DATASET_GROUPS.values() for names in title.values() for d in names}
    present = {d for algo in aggregated_results.values() for d in algo}
    missing = sorted(present - grouped)
    if missing:
        print(
            f"WARNING: datasets in results but absent from DATASET_GROUPS (omitted from subset "
            f"analysis): {', '.join(missing)}",
            file=sys.stderr,
        )


def _subcat_score(algo_results, datasets):
    """Pooled percentage score over the datasets in a subcategory (None if none present)."""
    pooled = []
    for dataset in datasets:
        if dataset in algo_results:  # dataset_name in the JSON is always the exact registry key
            pooled.extend(algo_results[dataset])
    return round(float(np.mean(pooled)) * 100, 1) if pooled else None


def generate_subset_analysis(
    aggregated_results: dict[str, dict[str, list[float]]],
) -> str:
    """Performance by dataset subset (origin / domain / cross-dimension)."""
    if not aggregated_results:
        return ""

    _check_group_coverage(aggregated_results)
    analysis = "## Performance by Dataset Subsets\n\n"
    for title, subcats in DATASET_GROUPS.items():
        analysis += f"### {title}\n"
        for name, datasets in subcats.items():
            analysis += f"- **{name}**: {', '.join(datasets)}\n"
        analysis += "\n"

        names = list(subcats)
        table = {
            algo: {n: _subcat_score(aggregated_results[algo], subcats[n]) for n in names}
            for algo in sorted(aggregated_results)
        }
        col_vals = {n: [table[a][n] for a in table] for n in names}
        headers = ["**Algorithm**"] + [f"**{n}**" for n in names]
        rows = [
            [algo] + [fmt_best(table[algo][n], col_vals[n], "{:.1f}%") for n in names]
            for algo in table
        ]
        analysis += md_table(headers, rows) + "\n"
    return analysis


def _band_weighted(pitch_results, bands, metric):
    """{algorithm: {band: frame-weighted metric}} across datasets (None if no data)."""
    agg = defaultdict(lambda: defaultdict(list))  # {algo: {band: [(value, frames)]}}
    for result in pitch_results:
        algo = result.get("metadata", {}).get("algorithm_name")
        per_band = result.get("results", {}).get("per_band")
        if not algo or not per_band:
            continue
        for band in bands:
            cell = per_band.get(band) or {}
            v = cell.get(metric)
            if v is not None and not np.isnan(v):
                agg[algo][band].append((v, cell.get("valid_frames") or 0))

    table = {}
    for algo, by_band in agg.items():
        table[algo] = {}
        for band in bands:
            vals = by_band.get(band, [])
            total = sum(f for _, f in vals)
            if total > 0:
                table[algo][band] = sum(v * f for v, f in vals) / total
            elif vals:
                table[algo][band] = float(np.mean([v for v, _ in vals]))
            else:
                table[algo][band] = None
    return table


def _band_subtable(table, bands, band_ranges, title, better, pct):
    """Render one algorithms x bands sub-table; `better` = 'max' or 'min'; pct -> '%' else rate."""
    if not table:
        return ""
    lower = better == "min"
    fmt = "{:.1%}" if pct else "{:.3f}"
    headers = ["**Algorithm**"] + [f"**{name}**<br>{rng}" for name, rng in band_ranges]
    rows = []
    for algo in sorted(table):
        def col(b):
            return [table[a][b] for a in table]
        cells = [fmt_best(table[algo][b], col(b), fmt, lower=lower) for b in bands]
        rows.append([algo, *cells])
    return f"### {title}\n\n" + md_table(headers, rows) + "\n"


def generate_band_analysis(pitch_results: list[dict]) -> str:
    """Per-pitch-band RPA + octave/gross error tables (clean audio), frame-weighted.

    Octave/gross are split out from RPA because the documented high-register failure is an
    octave/gross spike that RPA (which stays high) hides.
    """
    band_ranges = [(name, band_label(lo, hi)) for name, lo, hi in PITCH_BANDS]
    bands = [name for name, _ in band_ranges]
    subtables = [
        ("rpa", "Raw Pitch Accuracy (RPA &uarr;)", "max", True),
        ("octave", "Octave-error rate (&darr;)", "min", False),
        ("gross", "Gross-error rate (&darr;)", "min", False),
    ]
    rendered = [
        _band_subtable(
            _band_weighted(pitch_results, bands, metric), bands, band_ranges, title, better, pct
        )
        for metric, title, better, pct in subtables
    ]
    if not any(rendered):
        return ""
    return "## Performance by Pitch Band\n\n*Clean audio, by ground-truth f0 band.*\n\n" + "".join(
        rendered
    )


# Per-family generalization aggregates rendered after the main robustness table. Additive precedes
# convolutional (insertion order). `filtering` is intentionally absent: a singleton family needs no
# mean/worst aggregate, and the >= 2-present guard in _family_aggregate_section is the general net.
FAMILY_SECTIONS = {
    "additive": (
        "Additive-noise generalization",
        "Across the additive provenances: mean drop and the **worst-provenance** drop (max over "
        "sources). A low mean with a high worst-provenance flags a tracker robust only on some noise "
        "types -- a generalization/overfitting signal, since a tracker's training noise is unknown.",
    ),
    "convolutional": (
        "Reverberation generalization",
        "Across the convolutional conditions (synthetic reverb + physical room): mean drop and the "
        "**worst-case** drop (max). A low mean with a high worst flags sensitivity to one reverb "
        "character.",
    ),
}


def _family_aggregate_section(member_conds, present_conds, table, heading, blurb):
    """Render a mean-Δ + worst-provenance-Δ table for one condition family. Returns '' unless at
    least two of the family's conditions are present, so singletons and partial runs are skipped."""
    conds = [c for c in member_conds if c in present_conds]
    if len(conds) < 2:
        return ""
    fam_mean, fam_worst = {}, {}
    for algo in table:
        ds = [table[algo][c] for c in conds if table[algo][c] is not None]
        fam_mean[algo] = float(np.mean(ds)) if ds else None
        fam_worst[algo] = float(np.max(ds)) if ds else None   # worst provenance = largest drop
    order = sorted(fam_worst, key=lambda a: (fam_worst[a] is None, fam_worst[a] or 0.0))
    mvals, wvals = [fam_mean[a] for a in order], [fam_worst[a] for a in order]
    headers = ["**Algorithm**", "**mean Δ** ↓", "**worst-provenance Δ** ↓"]
    rows = [
        [a, fmt_best(fam_mean[a], mvals, "{:.1f}", lower=True),
         fmt_best(fam_worst[a], wvals, "{:.1f}", lower=True)]
        for a in order
    ]
    return f"\n### {heading}\n\n{blurb}\n\n" + md_table(headers, rows) + "\n"


def generate_robustness_analysis(probe_results: list[dict]) -> tuple[str, dict[str, float]]:
    """Δ-from-clean (combined-score percentage points) per degradation, on the probe.

    Returns (markdown_section, {algorithm: mean_delta}) where mean_delta is the average drop
    across conditions (lower = more robust), surfaced as a column in the leaderboard.
    """
    scores = defaultdict(lambda: defaultdict(dict))  # {algo: {condition: {dataset: score}}}
    for result in probe_results:
        m = result.get("metadata", {})
        algo, ds, cond = (
            m.get("algorithm_name"),
            m.get("dataset_name"),
            m.get("condition"),
        )
        sc = result.get("results", {}).get("combined_score")
        if algo and ds and cond and sc is not None and not np.isnan(sc):
            scores[algo][cond][ds] = sc

    conditions = sorted({c for a in scores for c in scores[a] if c != "clean"})
    if not scores or not conditions:
        return "", {}

    table, mean_delta = {}, {}
    for algo, by_cond in scores.items():
        clean = by_cond.get("clean", {})
        table[algo] = {}
        deltas = []
        for cond in conditions:
            cd = by_cond.get(cond, {})
            paired = [clean[d] - cd[d] for d in cd if d in clean]
            if paired:
                delta = float(np.mean(paired)) * 100.0
                table[algo][cond] = delta
                deltas.append(delta)
            else:
                table[algo][cond] = None
        mean_delta[algo] = float(np.mean(deltas)) if deltas else None

    analysis = "## Robustness (By Condition)\n\n"
    analysis += (
        "Δ-from-clean in combined-score percentage points (**lower = more robust**), measured "
        "on the robustness probe. All conditions are label-preserving (ground-truth f0 stays "
        "valid).\n\n"
        "⚠ Δ-from-clean is **floor-effect-prone**: a tracker that already scores poorly on "
        "clean audio has little room to drop, so a small mean Δ is only meaningful next to a "
        "high clean score (leaderboard). Compare absolute per-condition scores before ranking "
        "trackers by Δ.\n\n"
    )
    order = sorted(table, key=lambda a: (mean_delta[a] is None, mean_delta[a] or 0.0))
    col_vals = {c: [table[a][c] for a in order] for c in conditions}
    md_vals = [mean_delta[a] for a in order]
    headers = ["**Algorithm**"] + [f"**{c}** ↓" for c in conditions] + ["**mean Δ** ↓"]
    rows = []
    for algo in order:
        cells = [algo] + [
            fmt_best(table[algo][c], col_vals[c], "{:.1f}", lower=True) for c in conditions
        ]
        cells.append(fmt_best(mean_delta[algo], md_vals, "{:.1f}", lower=True))
        rows.append(cells)

    # Per-family generalization aggregates (additive, convolutional). The per-condition columns are
    # already above; each family section surfaces the mean and the WORST-case (max) drop over its
    # members. Uniformly-low = genuinely robust; low mean but high worst = likely overfit to one
    # provenance/character -- the fair generalization signal, since a tracker's training data is
    # unknown. Rendered only when >= 2 of a family's conditions are present (see FAMILY_SECTIONS).
    family_sections = "".join(
        _family_aggregate_section(CONDITION_FAMILIES[fam], conditions, table, heading, blurb)
        for fam, (heading, blurb) in FAMILY_SECTIONS.items()
    )
    return analysis + md_table(headers, rows) + "\n" + family_sections, mean_delta


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive analysis report from benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_report.md",
        help="Output markdown file path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        return 1

    print(f"Loading results from {args.results_dir}...")
    pitch_results, speed_results, ood_results = load_all_results(args.results_dir)
    pitch_results = _dedupe_prefer_cpu(pitch_results, _pitch_key)   # one row per cell, prefer cpu
    ood_results = _dedupe_prefer_cpu(ood_results, _ood_key)         # same for OOD (no device mixing)

    print(
        f"Found {len(pitch_results)} pitch benchmark results and {len(speed_results)} speed benchmark results."
    )

    # The leaderboard reports CLEAN accuracy only; degraded conditions are summarized separately
    # (robustness). Results without a condition default to clean.
    clean_results = [
        r
        for r in pitch_results
        if r.get("metadata", {}).get("condition", "clean") == "clean"
        and not r.get("metadata", {}).get("probe")
    ]
    probe_results = [
        r for r in pitch_results if r.get("metadata", {}).get("probe")
    ]
    print(
        f"Using {len(clean_results)} clean pitch results for the leaderboard, "
        f"{len(probe_results)} probe results for robustness."
    )

    # Diagnostics for the "results present but nothing renders" gotchas. A run capped with
    # --max-samples / --max-seconds is tagged as a robustness PROBE (probe=True), so it never feeds
    # the leaderboard; and a probe with only the clean condition can't produce a robustness table
    # (that needs paired degraded runs). Surface both clearly instead of failing silently.
    report_note = ""
    probe_conditions = {
        r.get("metadata", {}).get("condition", "clean") for r in probe_results
    }
    if not clean_results:
        if probe_results:
            report_note += (
                f"> **Note:** found {len(probe_results)} result(s) but **0 clean full-dataset "
                "results**, so the leaderboard is empty. Runs with `--max-samples`/`--max-seconds` "
                "are tagged as robustness **probes** and do not feed the leaderboard; re-run "
                "**without** those caps to get a leaderboard entry.\n\n"
            )
        else:
            report_note += (
                "> **Note:** no pitch results were found in the results directory.\n\n"
            )
    if probe_results and probe_conditions <= {"clean"}:
        report_note += (
            "> **Note:** the robustness table is empty: the probe runs are all the `clean` "
            "baseline. Robustness needs paired **degraded** probe runs "
            "(`--degradation white|pink|chime|demand|telephone|reverb|room`).\n\n"
        )
    if report_note:
        print("WARNING:\n" + report_note.replace("> ", "  ").replace("**", ""))

    aggregated_pitch = aggregate_pitch_results(clean_results)
    aggregated_speed = aggregate_speed_results(speed_results)
    detailed_metrics = collect_detailed_metrics(clean_results)
    robustness_section, robustness_score = generate_robustness_analysis(probe_results)

    print("Generating analysis report...")

    report = "# Pitch Detection Algorithm Benchmark Report\n\n"
    report += report_note  # surfaces the "probe-only / empty" gotchas in the report itself
    report += generate_methodology_section()
    report += generate_dataset_descriptions()
    # main performance table carries the robustness mean-Δ column
    report += generate_combined_score_table(aggregated_pitch, robustness=robustness_score)
    report += generate_band_analysis(clean_results)
    report += robustness_section
    report += generate_ood_analysis(ood_results)
    report += generate_speed_table(aggregated_speed)
    report += generate_detailed_analysis(detailed_metrics)
    report += generate_subset_analysis(aggregated_pitch)

    # Write report
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report generated: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
