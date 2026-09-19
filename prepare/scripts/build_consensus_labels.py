import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from constants import CONSENSUS, HOP_SIZE, RMS_FLOOR, SAMPLE_RATE, VOICED_THRESHOLD
from corpora import get_pitch_dataset, list_pitch_datasets
from corpora.base import frame_rms
from corpora.laryngograph import LaryngographSpeechDataset
from grid import cents, is_voiced
from scripts.trackers import track

SILENCE_THRESHOLD = 0.05
CONSENSUS_BAND_HZ = (50.0, 600.0)
CONSENSUS_AGREE_CENTS = 50.0
CHECKPOINT_EVERY = 200

BUILD_DATASETS = sorted(
    name for name in list_pitch_datasets()
    if issubclass(get_pitch_dataset(name), LaryngographSpeechDataset)
)


def _band(name):
    cls = get_pitch_dataset(name)
    lo, hi = CONSENSUS_BAND_HZ
    if not (lo < float(cls.fmin) and float(cls.fmax) < hi):
        raise ValueError(
            f"{name} is scored in [{cls.fmin}, {cls.fmax}] but the reference band is [{lo}, {hi}]; "
            f"the consensus cannot express a range wider than it is estimated in")
    return lo, hi


def _resample16(y, sr):
    t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    if sr != SAMPLE_RATE:
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
    return t.numpy()


def _item_stream(name, data_dir):
    cls = get_pitch_dataset(name)
    for loc, stem in cls._iter_originals(Path(data_dir)):
        def load(loc=loc, cls=cls, stem=stem):
            speech, egg, sr = cls._read_original(loc)
            if egg is None:
                raise ValueError(f"{name}: no EGG channel for {stem}")
            return _resample16(speech, sr), _resample16(egg, sr)
        yield stem, load


def silence_keep(mic16k, nfr, mode, k, quantile, thr=SILENCE_THRESHOLD):
    a = torch.from_numpy(np.ascontiguousarray(mic16k, dtype=np.float32))
    rms = frame_rms(a, HOP_SIZE, nfr).numpy()
    if mode == "floor":
        return rms >= k * np.quantile(rms, quantile)
    if mode != "peak":
        raise ValueError(f"silence_keep: unknown gate mode {mode!r} (expected 'peak' or 'floor')")
    return rms / (rms.max() + RMS_FLOOR) >= thr


def _estimator_track(name, egg16, band, threshold):
    pitch, _voicing = track(name, egg16, SAMPLE_RATE, HOP_SIZE, *band, threshold)
    pitch = np.asarray(pitch, dtype=float)
    return pitch, pitch > 0


def consensus(band, egg16, gate):
    pr, pv = _estimator_track("Praat", egg16, band, 0.0)
    ha, hv = _estimator_track("Harvest", egg16, band, VOICED_THRESHOLD)
    dg, dv = _estimator_track("DEGG", egg16, band, VOICED_THRESHOLD)
    L = min(len(pr), len(dg), len(ha), len(gate))
    pr, pv, dg, dv, ha, hv, gg = (x[:L] for x in (pr, pv, dg, dv, ha, hv, gate))

    praat_v = pv & gg
    Vs = np.stack([praat_v, dv & gg, hv & gg]).astype(float)
    Fs = np.stack([pr, dg, ha])

    nvoiced = Vs.sum(0)
    voicing_conf = nvoiced / 3.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pitch_hz = np.exp2(np.nanmedian(np.where(Vs > 0, np.log2(Fs), np.nan), axis=0))
    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        agree = {(i, j): (Vs[i] > 0) & (Vs[j] > 0)
                 & (np.abs(cents(Fs[i], Fs[j])) < CONSENSUS_AGREE_CENTS)
                 for i, j in ((0, 1), (0, 2), (1, 2))}
    all_three = agree[(0, 1)] & agree[(0, 2)] & agree[(1, 2)]
    any_pair = agree[(0, 1)] | agree[(0, 2)] | agree[(1, 2)]
    cluster = np.where(all_three, 3.0, np.where(any_pair, 2.0, np.minimum(nvoiced, 1.0)))
    pitch_conf = cluster / 3.0

    return voicing_conf.astype(np.float32), pitch_hz.astype(np.float32), pitch_conf.astype(np.float32)


def _save_atomic(labels, out_file, compress=True):
    tmp = out_file.with_name(out_file.name + ".tmp")
    save = np.savez_compressed if compress else np.savez
    with open(tmp, "wb") as f:
        save(f, **labels)
    os.replace(tmp, out_file)


def build(name, data_dir, limit, out_root, resume=False):
    fmin, fmax = _band(name)
    cls = get_pitch_dataset(name)
    gate_mode, gate_k = str(cls.GATE), float(cls.GATE_FLOOR_K)
    gate_q = float(cls.GATE_FLOOR_QUANTILE)
    items = list(_item_stream(name, data_dir))
    if not items:
        raise SystemExit(
            f"{name}: no items found under {data_dir!r}. A registered EGG corpus is never empty, so "
            f"this is a wrong --data-dir, not an empty corpus. Refusing to continue: with --resume "
            f"the pass would promote the RESUMED CHECKPOINT as if it were a finished corpus, "
            f"committing a truncated {name}.npz that the runtime loader only warns about.")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if limit:
        items = items[:limit]
        out_file = out_root / f"{name}.limit{limit}.npz"
    else:
        out_file = out_root / f"{name}.npz"
    print(f"{name}: {len(items)} files -> {out_file}  (fmin={fmin}, fmax={fmax}, gate={gate_mode}"
          f"{f', k={gate_k}' if gate_mode == 'floor' else ''})", flush=True)
    if limit:
        print(f"  --limit {limit}: this is NOT the canonical {name}.npz and is not read at "
              f"benchmark time; re-run without --limit to build the real labels.", flush=True)

    t0 = time.time()
    labels = {}
    ckpt_file = out_file.with_name(out_file.name + ".partial")
    src = out_file if out_file.exists() else ckpt_file if ckpt_file.exists() else None
    if resume and src is not None:
        with np.load(src) as z:
            labels = {k: z[k] for k in z.files}
        print(f"  resuming: {len(labels)} stems already present, skipping those", flush=True)
    vsum = nsum = tsum = fails = since_save = 0
    bar = tqdm(items, desc=name, unit="file")
    for stem, load in bar:
        if stem in labels:
            continue
        try:
            mic, egg = load()
            nfr = len(mic) // HOP_SIZE
            if nfr < 1:
                fails += 1
                continue
            egg = egg - np.mean(egg)
            egg = egg / (np.max(np.abs(egg)) + RMS_FLOOR)
            gate = silence_keep(mic, nfr, mode=gate_mode, k=gate_k, quantile=gate_q)
            vconf, phz, pconf = consensus((fmin, fmax), egg.astype(np.float32), gate)
        except Exception as e:
            fails += 1
            bar.write(f"  ! {stem}: {str(e)[:80]}")
            continue
        labels[stem] = np.stack([vconf, phz, pconf]).astype(np.float32)
        vsum += int(is_voiced(vconf).sum())
        tsum += int((is_voiced(vconf) & (pconf >= VOICED_THRESHOLD)).sum())
        nsum += len(vconf)
        since_save += 1
        if since_save >= CHECKPOINT_EVERY:
            _save_atomic(labels, ckpt_file, compress=False)
            since_save = 0
        bar.set_postfix(voiced=f"{100 * vsum / max(nsum, 1):.0f}%", fails=fails)
    if not labels:
        ckpt_file.unlink(missing_ok=True)
        print(f"{name}: produced 0 labels; leaving any existing {out_file.name} untouched.", flush=True)
        return
    _save_atomic(labels, out_file)
    ckpt_file.unlink(missing_ok=True)
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
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #files for a smoke test (0 = all). A limited run writes "
                         "<NAME>.limit<N>.npz, never the canonical <NAME>.npz")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing .npz checkpoint (skip already-done stems); "
                         "default overwrites so a regeneration never keeps stale labels")
    args = ap.parse_args()
    build(args.dataset, args.data_dir, args.limit, CONSENSUS, resume=args.resume)


if __name__ == "__main__":
    main()
