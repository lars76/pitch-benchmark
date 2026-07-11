"""Generate cross-estimator consensus labels for laryngograph speech corpora and write one
``<NAME>.npz`` per corpus into ``datasets/laryngograph/`` (read at benchmark runtime by the
LaryngographSpeechDataset subclasses there).

For each utterance: run three independent EGG f0 estimators (Praat, dEGG, Harvest) on the
laryngograph signal, sampled on the benchmark frame grid (16 kHz / hop 256). A mic-energy silence
gate removes EGG-bandpass silence artifacts. A frame is:
  - voiced (F1 positive) iff >=2 of the 3 estimators voice it;
  - pitch-confident iff >=2 estimators also AGREE on the f0 (<50 cents) -> store the consensus f0;
  - voiced-but-pitch-uncertain (>=2 voice, estimators disagree on value) -> f0=0, voiced=1
    (the benchmark's finite-frame rule drops f0=0 frames from RPA while F1 keeps them);
  - unvoiced -> f0=0, voiced=0.

The set was originally four trackers with Praat and REAPER grouped as a single correlation vote;
REAPER was dropped, leaving three estimators that each vote independently. pyreaper has a heap-overflow
bug that segfaults (SIGSEGV/SIGABRT, uncatchable by try/except) on ~1-6% of files at the wide 50/500
band the non-PTDB corpora use, which would kill a whole generation run. An A/B over all corpora
(against the RAPT-free dEGG+Harvest anchor) showed keeping REAPER beats dropping it by only ~0.2 RPA,
and that swapping in RAPT was strictly worse -- so Praat, dEGG, and Harvest now each vote independently.

The estimators are the SAME first-class trackers the benchmark uses, run on the EGG channel: Praat and
Harvest come from algorithms/ via get_algorithm; dEGG is algorithms.degg (EGG-only, so it is imported
directly and is deliberately not in the benchmark registry). The shared algorithm base class aligns
each estimator onto the eval grid (resampling.resample_to_grid), so the consensus is built on exactly
the grid the runtime uses -- there is no bespoke resampling here.

Each corpus's EGG is decoded from its ORIGINAL extracted download by the loader class's shared
``_read_original`` (the same reader the runtime uses for the speech), so the on-disk format lives in
exactly one place. Built for every EGG corpus (``BUILD_DATASETS`` below): PTDB, CMUArctic, AVID,
APLAWD, SVD, OSFGlottis, MOCHA, KEELE, FDA -- consensus is the default ground truth for all of them
(PTDB/KEELE/FDA also ship an author f0 available via label_source="reference"). URMP is music (no EGG).

GENERATION-TIME ONLY. Needs the praat/harvest backends (scipy/numpy/torch/torchaudio are core); the
``--data-dir`` is the EXTRACTED root for that corpus (see the README dataset tree):

    uv sync --extra praat --extra harvest
    uv run python scripts/build_consensus_labels.py --dataset PTDB      --data-dir "../datasets/SPEECH DATA"
    uv run python scripts/build_consensus_labels.py --dataset CMUArctic --data-dir "../datasets/cmu_arctic_egg"
    uv run python scripts/build_consensus_labels.py --dataset SVD       --data-dir "../datasets/svd/healthy"
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
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
from datasets import get_pitch_dataset, list_pitch_datasets
from datasets.base import frame_rms
from datasets.laryngograph import LaryngographSpeechDataset
from metrics import cents

SR, HOP = 16000, 256
OUT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "laryngograph"


# Every EGG corpus gets a consensus npz (the default ground truth for all of them). PTDB/KEELE/FDA
# additionally ship an author f0 (label_source="reference"), but consensus is still their default, so
# they are built here too. Derived from the registry as exactly the LaryngographSpeechDataset
# subclasses, so a newly-registered EGG corpus is buildable (and a valid --dataset choice) the moment
# it is registered -- no second hand-maintained list to forget. URMP (music, no EGG) is excluded for
# free (not a laryngograph subclass). Each corpus decodes its ORIGINAL download through its loader
# class's shared _iter_originals / _read_original -- the same reader the runtime uses -- so the on-disk
# format lives in one place, and the pitch band comes from the loader class via _band() (no drift).
BUILD_DATASETS = sorted(
    name for name in list_pitch_datasets()
    if issubclass(get_pitch_dataset(name), LaryngographSpeechDataset)
)


def _band(name: str) -> tuple[float, float]:
    """The (fmin, fmax) the corpus is scored in, read from its runtime loader class so the consensus
    is estimated in the same band and can never drift from what the benchmark actually uses."""
    cls = get_pitch_dataset(name)
    return float(cls.fmin), float(cls.fmax)


def _resample16(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample a native-rate mono array to 16 kHz with the SAME torchaudio resampler the runtime
    loader uses (via PitchDataset._prepare_audio), so a corpus's consensus label grid and its
    benchmark-time speech grid coincide frame for frame."""
    t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t.numpy()


def _item_stream(name: str, data_dir: str):
    """Yield ``(stem, load)`` where ``load()`` returns ``(mic16, egg16)`` for one item, decoded from
    the ORIGINAL download through the loader class's shared reader (the SAME reader the runtime uses).
    ``load`` is lazy so ``--resume`` can skip a finished stem without decoding it."""
    cls = get_pitch_dataset(name)
    for loc, stem in cls._iter_originals(Path(data_dir)):
        def load(loc=loc, cls=cls, stem=stem):
            speech, egg, sr = cls._read_original(loc)
            if egg is None:
                raise ValueError(f"{name}: no EGG channel for {stem}")
            return _resample16(speech, sr), _resample16(egg, sr)
        yield stem, load


def silence_keep(mic16k: np.ndarray, nfr: int, thr: float = 0.05) -> np.ndarray:
    """True where the mic frame is loud enough (>= thr of the per-file peak RMS). Uses the shared
    per-frame RMS primitive (datasets.base.frame_rms) with CENTERED windows, matching both the grid
    contract and the runtime energy_voicing_gate, so the consensus labels are gated the same way
    the author path is. NOTE: npz files generated before the centered-gate fix (and before the
    REAPER timestamp fix) carry the old conventions -- regenerate to pick both up."""
    a = torch.from_numpy(np.ascontiguousarray(mic16k, dtype=np.float32))
    rms = frame_rms(a, HOP, nfr, center=True).numpy()
    return rms / (rms.max() + 1e-12) >= thr


def _estimator_track(algo, egg16: np.ndarray, threshold: float):
    """Run one estimator on the 16 kHz EGG at a fixed operating point; return (pitch_hz, voiced_bool).
    voiced == pitch > 0 (every estimator marks voicing by f0 > 0). For Praat, threshold 0.0 makes
    voicing all-True so pitch survives wherever to_pitch itself voiced it (its strength threshold would
    otherwise cut voicing); Harvest/dEGG use 0.5 (they carry a binary confidence)."""
    pitch, _voicing, _notes = algo.extract_pitch(egg16, thresholds=[threshold], compute_notes=False)[0]
    pitch = np.asarray(pitch, dtype=float)
    return pitch, pitch > 0


def consensus(algos, egg16: np.ndarray, gate: np.ndarray):
    """Three per-frame confidence/value arrays over the 3 independent EGG estimators (Praat, dEGG,
    Harvest), all gated by mic-energy silence:
      voicing_conf = (# estimators that voice the frame) / 3                     in {0, 1/3, 2/3, 1}
      pitch_hz     = geometric (log-Hz) median of the voiced estimators' f0 (0 if none) -- always kept
      pitch_conf   = (largest set of voiced estimators PAIRWISE within 50c) / 3  in {0, 1/3, 2/3, 1}
    """
    praat, harvest, degg = algos
    pr, pv = _estimator_track(praat, egg16, 0.0)
    ha, hv = _estimator_track(harvest, egg16, 0.5)
    dg, dv = _estimator_track(degg, egg16, 0.5)
    L = min(len(pr), len(dg), len(ha), len(gate))
    pr, pv, dg, dv, ha, hv, gg = (x[:L] for x in (pr, pv, dg, dv, ha, hv, gate))

    # Three independent estimators, each gated by mic-energy silence (gg). praat_v requires Praat
    # voiced, so where its f0 is used pr > 0 and log2(pr) is well-defined.
    praat_v = pv & gg
    Vs = np.stack([praat_v, dv & gg, hv & gg]).astype(float)      # (3, L): Praat, dEGG, Harvest
    Fs = np.stack([pr, dg, ha])                                   # (3, L): Praat, dEGG, Harvest

    nvoiced = Vs.sum(0)                                           # 0..3
    voicing_conf = nvoiced / 3.0

    # Consensus f0 in LOG-Hz (geometric median): pitch is perceived logarithmically, so the average of
    # the voiced estimators must be taken in cents/log-Hz, not linear Hz (mir_eval interpolates melodies
    # in cents for the same reason). The linear midpoint is biased toward the higher estimator.
    with warnings.catch_warnings():                              # all-NaN slices -> NaN (handled below)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pitch_hz = np.exp2(np.nanmedian(np.where(Vs > 0, np.log2(Fs), np.nan), axis=0))
    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)

    # pitch_conf = (largest set of voiced estimators that PAIRWISE agree within 50c) / 3. Agreement
    # must be pairwise, NOT "within 50c of the median": the two cent-distances from any midpoint sum to
    # the estimators' separation, so "within 50c of the midpoint" would admit pairs up to ~100c apart
    # (2x the threshold). Normalizing by 3 (not nvoiced) makes conf >= 0.5 mean exactly ">= 2 estimators
    # agree", matching the promised label semantics (and correctly leaving a lone voiced estimator
    # unconfident).
    with np.errstate(divide="ignore", invalid="ignore"):
        cluster = np.zeros_like(pitch_hz)
        for i in range(3):
            ci = sum((Vs[i] > 0) & (Vs[j] > 0) & (np.abs(cents(Fs[i], Fs[j])) < 50) for j in range(3))
            cluster = np.maximum(cluster, ci.astype(float))
    pitch_conf = cluster / 3.0

    return voicing_conf.astype(np.float32), pitch_hz.astype(np.float32), pitch_conf.astype(np.float32)


def _save_atomic(labels: dict, out_file: Path, compress: bool = True):
    """Write the label dict to a temp file, then atomically rename over out_file. Passing an open
    handle (not a path) stops np.savez appending a second .npz; os.replace is atomic on POSIX, so a
    crash mid-write can never leave a truncated/corrupt .npz -- the previous checkpoint survives.
    ``compress=False`` skips gzip for intermediate checkpoints: recompressing the whole growing dict
    on every checkpoint is O(N^2) total work, so checkpoints are written uncompressed (np.load reads
    either format, so --resume is unaffected) and only the final flush is compressed."""
    tmp = out_file.with_name(out_file.name + ".tmp")
    save = np.savez_compressed if compress else np.savez
    with open(tmp, "wb") as f:
        save(f, **labels)
    os.replace(tmp, out_file)


CHECKPOINT_EVERY = 200  # flush the growing label dict every N newly-processed files


def build(name, data_dir, limit, resume=False):
    fmin, fmax = _band(name)
    items = list(_item_stream(name, data_dir))       # [(stem, load), ...]
    if limit:
        items = items[:limit]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_file = OUT_ROOT / f"{name}.npz"
    print(f"{name}: {len(items)} files -> {out_file}  (fmin={fmin}, fmax={fmax})", flush=True)

    # One estimator set per (fmin, fmax); reused across files (stateless w.r.t. the audio argument).
    algos = (
        get_algorithm("Praat")(SR, HOP, fmin, fmax),
        get_algorithm("Harvest")(SR, HOP, fmin, fmax),
        DEGGPitchAlgorithm(SR, HOP, fmin, fmax),
    )

    t0 = time.time()
    labels = {}                                 # stem -> (3, n) array [voicing_conf; pitch_hz; pitch_conf]
    ckpt_file = out_file.with_name(out_file.name + ".partial")   # checkpoints land HERE, never on out_file
    # Checkpoints go to a sidecar (.partial), so an interrupted or misdirected run can never replace a
    # good committed .npz with a partial/empty one: out_file is touched only by the final promote below,
    # and only when at least one label was produced. --resume continues from whichever is newer-complete
    # (a finished out_file, else a leftover .partial). The DEFAULT (no --resume) starts fresh and, on a
    # full successful pass, overwrites out_file so a regeneration never silently keeps stale labels.
    src = out_file if out_file.exists() else ckpt_file if ckpt_file.exists() else None
    if resume and src is not None:
        with np.load(src) as z:
            labels = {k: z[k] for k in z.files}
        print(f"  resuming: {len(labels)} stems already present, skipping those", flush=True)
    vsum = nsum = tsum = fails = since_save = 0
    # tqdm gives the count / elapsed / ETA bar; set_postfix surfaces the running voiced% + fail count,
    # and bar.write keeps per-file error lines from corrupting the bar.
    bar = tqdm(items, desc=name, unit="file")
    for stem, load in bar:
        if stem in labels:                       # already done (resume) -> skip
            continue
        try:
            mic, egg = load()
            nfr = len(mic) // HOP
            if nfr < 1:
                fails += 1
                continue
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
        since_save += 1
        if since_save >= CHECKPOINT_EVERY:       # periodic atomic flush to the SIDECAR -> crash loses <=N files
            _save_atomic(labels, ckpt_file, compress=False)  # uncompressed: avoids O(N^2) recompression
            since_save = 0
        bar.set_postfix(voiced=f"{100 * vsum / max(nsum, 1):.0f}%", fails=fails)
    if not labels:                               # empty/wrong --data-dir or all-failed: never clobber
        ckpt_file.unlink(missing_ok=True)
        print(f"{name}: produced 0 labels; leaving any existing {out_file.name} untouched.", flush=True)
        return
    _save_atomic(labels, out_file)               # promote: one committed .npz per dataset, keyed by file stem
    ckpt_file.unlink(missing_ok=True)            # drop the sidecar now that out_file is complete
    el = time.time() - t0
    vpct = 100 * vsum / max(nsum, 1)
    tpct = 100 * tsum / max(nsum, 1)
    print(f"{name}: done {len(labels)}/{len(items)} ({fails} fails) in {el / 60:.1f}m | "
          f"voiced(conf>=.5) {vpct:.0f}%  pitch-confident {tpct:.0f}% of frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PTDB", choices=BUILD_DATASETS,
                    help="laryngograph corpus to build consensus labels for")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #files (0 = all; for quick tests)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing .npz checkpoint (skip already-done stems); "
                         "default overwrites so a regeneration never keeps stale labels")
    args = ap.parse_args()
    build(args.dataset, args.data_dir, args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
