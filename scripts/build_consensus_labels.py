"""Generate cross-family consensus labels for laryngograph speech corpora and write one
``<NAME>.npz`` per corpus into ``datasets/laryngograph/`` (read at benchmark runtime by the
LaryngographSpeechDataset subclasses there).

For each utterance: run four EGG f0 estimators across three families (correlation = Praat+REAPER as
one vote; period = dEGG; instantaneous-frequency = Harvest) on the laryngograph signal, sampled on the
benchmark frame grid (16 kHz / hop 256). A mic-energy silence gate removes EGG-bandpass silence
artifacts. A frame is:
  - voiced (F1 positive) iff >=2 of the 3 families voice it;
  - pitch-confident iff >=2 families also AGREE on the f0 (<50 cents) -> store the consensus f0;
  - voiced-but-pitch-uncertain (>=2 voice, families disagree on value) -> f0=0, voiced=1
    (the benchmark's finite-frame rule drops f0=0 frames from RPA while F1 keeps them);
  - unvoiced -> f0=0, voiced=0.

The estimators are the SAME first-class trackers the benchmark uses, run on the EGG channel: Praat,
REAPER and Harvest come from algorithms/ via get_algorithm; dEGG is algorithms.degg (EGG-only, so it
is imported directly and is deliberately not in the benchmark registry). The shared algorithm base
class aligns each estimator onto the eval grid (resampling.resample_to_grid), so the consensus is built
on exactly the grid the runtime uses -- there is no bespoke resampling here.

GENERATION-TIME ONLY. Needs the praat/reaper/harvest backends (scipy/numpy/torch/torchaudio are core):

    uv sync --extra praat --extra reaper --extra harvest
    uv run python scripts/build_consensus_labels.py --dataset PTDB --data-dir "../datasets/SPEECH DATA"
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# Repo root on sys.path so algorithms/, metrics, resampling and datasets import when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms import (
    get_algorithm,
)
from algorithms.degg import (
    DEGGPitchAlgorithm,
)
from datasets.base import frame_rms
from metrics import cents

SR, HOP = 16000, 256
OUT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "laryngograph"


# --- per-dataset EGG resolvers: yield (mic_wav, egg_path, stem) -------------
def ptdb_items(root: Path):
    for gender in ("MALE", "FEMALE"):
        mic_dir = root / gender / "MIC"
        if not mic_dir.exists():
            continue
        for wav in sorted(mic_dir.rglob("*.wav")):
            lar = Path(str(wav).replace("/MIC/", "/LAR/"))
            lar = lar.with_name(wav.name.replace("mic_", "lar_"))
            if lar.exists():
                yield wav, lar, wav.stem


# name -> (item-iterator, fmin, fmax)
DATASETS = {
    "PTDB": (ptdb_items, 65.0, 300.0),
}


def _load_16k(wav: Path) -> np.ndarray:
    """Decode (soundfile) + resample to 16 kHz with torchaudio.functional.resample. Used for BOTH the
    mic (silence gate) and the EGG (estimation), so their frame grids (len // hop) match frame for
    frame and the sample count matches the benchmark loader exactly (same DSP resampler)."""
    y, sr = sf.read(str(wav))
    y = np.asarray(y, float)
    if y.ndim > 1:
        y = y[:, 0]
    t = torch.from_numpy(y).float()
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t.numpy()


def silence_keep(mic16k: np.ndarray, nfr: int, thr: float = 0.05) -> np.ndarray:
    """True where the mic frame is loud enough (>= thr of the per-file peak RMS). Uses the shared
    per-frame RMS primitive (datasets.base.frame_rms) with CENTERED windows, matching both the grid
    contract and the runtime energy_voicing_gate, so the consensus labels are gated the same way
    the author path is. NOTE: npz files generated before the centered-gate fix (and before the
    REAPER timestamp fix) carry the old conventions -- regenerate to pick both up."""
    a = torch.from_numpy(np.ascontiguousarray(mic16k, dtype=np.float32))
    rms = frame_rms(a, HOP, nfr, center=True).numpy()
    return rms / (rms.max() + 1e-12) >= thr


def _family_track(algo, egg16: np.ndarray, threshold: float):
    """Run one estimator on the 16 kHz EGG at a fixed operating point; return (pitch_hz, voiced_bool).
    voiced == pitch > 0 reproduces the historical per-family voicing (every family used f0 > 0). For
    Praat, threshold 0.0 makes voicing all-True so pitch survives wherever to_pitch itself voiced it
    (its strength threshold would otherwise cut voicing); REAPER/Harvest/dEGG use 0.5 (REAPER -> its
    default unvoiced_cost 0.9; Harvest/dEGG carry a binary confidence)."""
    pitch, _voicing, _notes = algo.extract_pitch(egg16, thresholds=[threshold], compute_notes=False)[0]
    pitch = np.asarray(pitch, dtype=float)
    return pitch, pitch > 0


def consensus(algos, egg16: np.ndarray, gate: np.ndarray):
    """Three per-frame confidence/value arrays over the 3 EGG families (correlation = Praat+REAPER as
    one vote; period = dEGG; instantaneous-frequency = Harvest), all gated by mic-energy silence:
      voicing_conf = (# families that voice the frame) / 3                     in {0, 1/3, 2/3, 1}
      pitch_hz     = geometric (log-Hz) median of the voiced families' f0 (0 if none) -- always kept
      pitch_conf   = (largest set of voiced families PAIRWISE within 50c) / 3  in {0, 1/3, 2/3, 1}
    """
    praat, reaper, harvest, degg = algos
    pr, pv = _family_track(praat, egg16, 0.0)
    re, rv = _family_track(reaper, egg16, 0.5)
    ha, hv = _family_track(harvest, egg16, 0.5)
    dg, dv = _family_track(degg, egg16, 0.5)
    L = min(len(pr), len(re), len(dg), len(ha), len(gate))
    pr, pv, re, rv, dg, dv, ha, hv, gg = (x[:L] for x in (pr, pv, re, rv, dg, dv, ha, hv, gate))

    # The correlation family's value is the GEOMETRIC mean of Praat & REAPER (pitch is compared in the
    # log domain, so the geomean is the right average); corr_v requires both voiced, so where it is
    # used both pr,re > 0 and sqrt(pr*re) is well-defined.
    corr_v = (pv & gg) & (rv & gg)
    Vs = np.stack([corr_v, dv & gg, hv & gg]).astype(float)       # (3, L)
    Fs = np.stack([np.sqrt(pr * re), dg, ha])                     # (3, L)

    nvoiced = Vs.sum(0)                                           # 0..3
    voicing_conf = nvoiced / 3.0

    # Consensus f0 in LOG-Hz (geometric median): pitch is perceived logarithmically, so the average of
    # the voiced families must be taken in cents/log-Hz, not linear Hz (mir_eval interpolates melodies
    # in cents for the same reason). The linear midpoint is biased toward the higher family.
    with warnings.catch_warnings():                              # all-NaN slices -> NaN (handled below)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pitch_hz = np.exp2(np.nanmedian(np.where(Vs > 0, np.log2(Fs), np.nan), axis=0))
    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)

    # pitch_conf = (largest set of voiced families that PAIRWISE agree within 50c) / 3. Agreement must
    # be pairwise, NOT "within 50c of the median": the two cent-distances from any midpoint sum to the
    # families' separation, so "within 50c of the midpoint" would admit pairs up to ~100c apart (2x the
    # threshold). Normalizing by 3 (not nvoiced) makes conf >= 0.5 mean exactly ">= 2 families agree",
    # matching the promised label semantics (and correctly leaving a lone voiced family unconfident).
    with np.errstate(divide="ignore", invalid="ignore"):
        cluster = np.zeros_like(pitch_hz)
        for i in range(3):
            ci = sum((Vs[i] > 0) & (Vs[j] > 0) & (np.abs(cents(Fs[i], Fs[j])) < 50) for j in range(3))
            cluster = np.maximum(cluster, ci.astype(float))
    pitch_conf = cluster / 3.0

    return voicing_conf.astype(np.float32), pitch_hz.astype(np.float32), pitch_conf.astype(np.float32)


def build(name, data_dir, limit):
    items_fn, fmin, fmax = DATASETS[name]
    items = list(items_fn(Path(data_dir)))
    if limit:
        items = items[:limit]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_file = OUT_ROOT / f"{name}.npz"
    print(f"{name}: {len(items)} files -> {out_file}  (fmin={fmin}, fmax={fmax})", flush=True)

    # One estimator set per (fmin, fmax); reused across files (stateless w.r.t. the audio argument).
    algos = (
        get_algorithm("Praat")(SR, HOP, fmin, fmax),
        get_algorithm("REAPER")(SR, HOP, fmin, fmax),
        get_algorithm("Harvest")(SR, HOP, fmin, fmax),
        DEGGPitchAlgorithm(SR, HOP, fmin, fmax),
    )

    t0 = time.time()
    labels = {}                                 # stem -> (3, n) array [voicing_conf; pitch_hz; pitch_conf]
    vsum = nsum = tsum = fails = 0
    # tqdm gives the count / elapsed / ETA bar; set_postfix surfaces the running voiced% + fail count,
    # and bar.write keeps per-file error lines from corrupting the bar.
    bar = tqdm(items, desc=name, unit="file")
    for wav, egg_path, stem in bar:
        try:
            mic = _load_16k(wav)
            nfr = len(mic) // HOP
            if nfr < 1:
                fails += 1
                continue
            egg = _load_16k(egg_path)
            egg = egg - np.mean(egg)                          # DC removal (parity with the old _prep)
            egg = egg / (np.max(np.abs(egg)) + 1e-12)         # peak-norm -> |egg| < 1 for _validate_audio
            gate = silence_keep(mic, nfr)
            vconf, phz, pconf = consensus(algos, egg.astype(np.float32), gate)
        except Exception as e:
            fails += 1
            bar.write(f"  ! {stem}: {str(e)[:80]}")
            continue
        labels[stem] = np.stack([vconf, phz, pconf]).astype(np.float32)
        vsum += int((vconf >= 0.5).sum())
        tsum += int(((vconf >= 0.5) & (pconf >= 0.5)).sum())
        nsum += len(vconf)
        bar.set_postfix(voiced=f"{100 * vsum / max(nsum, 1):.0f}%", fails=fails)
    np.savez_compressed(out_file, **labels)     # one .npz per dataset, keyed by file stem
    el = time.time() - t0
    vpct = 100 * vsum / max(nsum, 1)
    tpct = 100 * tsum / max(nsum, 1)
    print(f"{name}: done {len(labels)}/{len(items)} ({fails} fails) in {el / 60:.1f}m | "
          f"voiced(conf>=.5) {vpct:.0f}%  pitch-confident {tpct:.0f}% of frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PTDB", choices=list(DATASETS))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #files (0 = all; for quick tests)")
    args = ap.parse_args()
    build(args.dataset, args.data_dir, args.limit)


if __name__ == "__main__":
    main()
