"""Measure per-dataset corpus statistics for the benchmark's dataset introduction.

Exact (full pass over every clip, no sampling): clip count, total hours, average
clip length, voiced-frame fraction, ground-truth f0 distribution (p5/p50/p95 + min/max
over voiced frames), and per-pitch-band coverage. Writes a Markdown table to --out.

Note on ranges: datasets default to clip_pitch=False, so ground-truth f0 outside a
dataset's [fmin, fmax] window is marked UNVOICED (excluded), not clamped. Reported f0
stats are therefore over in-window voiced frames; p5-p95 is the real content spread.

Usage (run from the repo root):
    uv run python scripts/dataset_stats.py --out dataset_stats.md

Also prints a ready-to-paste `corpus_stats` Python block for generate_report.py (the report keeps a
hardcoded copy so it never needs the raw datasets at report time); paste it over the existing list.
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

# Repo root on sys.path so `datasets`, `metrics` import when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_pitch_dataset
from metrics import PITCH_BANDS, band_label, is_voiced

SR, HOP = 16000, 256

# (name, domain, data-dir). Order matches generate_report.py:corpus_stats so the paste is 1:1. Paths
# are relative to the repo root (mirroring run.sh); edit them for your machine.
DATASETS = [
    ("NSynth", "Music", "../datasets/nsynth-test"),
    ("PTDB", "Speech", "../datasets/SPEECH DATA"),
    ("MIR1K", "Music", "../datasets/MIR-1K"),
    ("MDBStemSynth", "Music", "../datasets/MDB-stem-synth"),
    ("Vocadito", "Music", "../datasets/vocadito"),
    ("Bach10Synth", "Music", "../datasets/Bach10Synth/Bach10-mf0-synth"),
    ("SpeechSynth", "Speech", "datasets/speechsynth.pt"),
    # EGG (laryngograph) speech corpora + KEELE/FDA/MOCHA/CMUArctic (consensus f0)
    ("KEELE", "Speech", "../datasets/KEELE/KEELE"),
    ("FDA", "Speech", "../datasets/FDA"),
    ("MOCHA", "Speech", "../datasets/MOCHA"),
    ("CMUArctic", "Speech", "../datasets/cmu_arctic_egg"),
    ("SVD", "Speech", "../datasets/svd_zenodo/healthy"),
    ("APLAWD", "Speech", "../datasets/aplawd/APLAWDW"),
    ("OSFGlottis", "Speech", "../datasets/osf_glottis"),
    ("AVID", "Speech", "../datasets/avid"),
    # singing (M4Singer = voicing-only: score-grade pitch GT) + multi-instrument
    ("M4Singer", "Music", "../datasets/m4singer"),
    ("URMP", "Music", "../datasets/URMP"),
]


def measure(name, data_dir):
    """Full-pass exact stats for one dataset. Returns a dict (or None on load failure)."""
    try:
        ds = get_pitch_dataset(name)(root_dir=data_dir, sample_rate=SR, hop_size=HOP)
    except Exception as e:
        print(f"  {name}: SKIP ({str(e)[:70]})")
        return None

    n = len(ds)
    n_ok = 0
    total_samples = 0
    voiced_frames = total_frames = 0
    f0_chunks = []
    band_counts = {b[0]: 0 for b in PITCH_BANDS}

    bar = tqdm(range(n), desc=name, unit="clip")   # full-pass bar; ETA/rate come free
    for i in bar:
        try:
            it = ds[i]
        except Exception as e:
            bar.write(f"  {name}[{i}]: item error ({str(e)[:50]}) -- skipping clip")
            continue
        n_ok += 1
        audio = np.asarray(it["audio"]).reshape(-1)
        pitch = np.asarray(it["pitch"]).reshape(-1)
        voiced = is_voiced(np.asarray(it["periodicity"]).reshape(-1))
        total_samples += audio.size
        total_frames += voiced.size
        m = voiced & (pitch > 0)
        voiced_frames += int(m.sum())
        if m.any():
            f = pitch[m].astype(np.float64)
            f0_chunks.append(f)
            for bname, lo, hi in PITCH_BANDS:
                band_counts[bname] += int(np.sum((f >= lo) & (f < hi)))

    f0 = np.concatenate(f0_chunks) if f0_chunks else np.array([np.nan])
    hours = total_samples / SR / 3600.0
    p5, p50, p95 = np.percentile(f0, [5, 50, 95])
    nv = max(voiced_frames, 1)
    bands = {b: 100.0 * band_counts[b] / nv for b in band_counts}
    if n_ok < n:
        print(f"  {name}: {n - n_ok}/{n} clips failed to load; stats cover the {n_ok} loadable clips")
    return {
        "n": n, "hours": hours, "avg_len": total_samples / SR / max(n_ok, 1),
        "voiced_pct": 100.0 * voiced_frames / max(total_frames, 1),
        "p5": p5, "p50": p50, "p95": p95, "fmin_obs": np.nanmin(f0), "fmax_obs": np.nanmax(f0),
        "window": (ds.fmin, ds.fmax), "bands": bands,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dataset_stats.md")
    args = ap.parse_args()

    band_ranges = {name: band_label(lo, hi) for name, lo, hi in PITCH_BANDS}
    rows = []
    for name, domain, data_dir in DATASETS:
        print(f"measuring {name} ...")
        s = measure(name, data_dir)
        if s:
            rows.append((name, domain, s))

    lines = ["# Dataset Corpus Statistics\n",
             "*Exact, full-pass measurement. f0 stats are over in-window voiced frames "
             "(out-of-[fmin,fmax] ground truth is marked unvoiced, not clamped). "
             "Band coverage = % of voiced frames whose f0 falls in each band.*\n",
             "| **Dataset** | **Domain** | **Clips** | **Hours** | **Avg len (s)** | "
             "**Voiced %** | **f0 p5-p50-p95 (Hz)** | **Window [fmin-fmax]** | **Band coverage** |",
             "|---|---|--:|--:|--:|--:|---|---|---|"]
    for name, domain, s in rows:
        cov = ", ".join(f"{b} {s['bands'][b]:.0f}%" for b, _, _ in PITCH_BANDS if s["bands"][b] >= 1.0)
        f0r = f"{s['p5']:.0f}-{s['p50']:.0f}-{s['p95']:.0f}"
        win = f"{s['window'][0]:.0f}-{s['window'][1]:.0f}"
        lines.append(
            f"| {name} | {domain} | {s['n']} | {s['hours']:.1f} | {s['avg_len']:.1f} | "
            f"{s['voiced_pct']:.0f} | {f0r} | {win} | {cov} |"
        )
    lines.append("")
    lines.append("Bands: " + ", ".join(f"{n} ({band_ranges[n]})" for n, _, _ in PITCH_BANDS) + ".")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    print("\n".join(lines))

    # Ready-to-paste block for generate_report.py:corpus_stats (its schema, its column order), so
    # updating the report's hardcoded table is a single copy-paste with no manual reshaping.
    print("\n# ---- paste over generate_report.py `corpus_stats` ----")
    print("    corpus_stats = [")
    for name, _domain, s in rows:
        cov = ", ".join(f"{b} {s['bands'][b]:.0f}" for b, _, _ in PITCH_BANDS if s["bands"][b] >= 1.0)
        f0r = f"{s['p5']:.0f}-{s['p50']:.0f}-{s['p95']:.0f}"
        print(
            f'        ("{name}", {s["n"]}, {s["hours"]:.1f}, {s["avg_len"]:.1f}, '
            f'{round(s["voiced_pct"])}, "{f0r}", "{cov}"),'
        )
    print("    ]")


if __name__ == "__main__":
    main()
