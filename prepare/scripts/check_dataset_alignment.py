import argparse

import numpy as np

from constants import HOP_SIZE, SAMPLE_RATE, VOICED_THRESHOLD
from corpora import get_pitch_dataset
from grid import CENTS_REF_HZ, cents
from scripts.trackers import available, track

REFERENCES = ("Praat", "DIO", "SWIPE")

OFFSET_ATTR = "F0_LABEL_OFFSET_SECONDS"


def cents_at(pred_cents, pred_voiced, shift_frames):
    n = len(pred_cents)
    i = np.arange(n) + shift_frames
    lo = np.floor(i).astype(int)
    hi = lo + 1
    w = i - lo
    ok = (lo >= 0) & (hi < n)
    out = np.full(n, np.nan)
    if not ok.any():
        return out
    li, hj, ww = lo[ok], hi[ok], w[ok]
    both = pred_voiced[li] & pred_voiced[hj]
    vals = np.where(both, pred_cents[li] * (1 - ww) + pred_cents[hj] * ww, np.nan)
    out[ok] = vals
    return out


def sweep_file(label_f0, label_voiced, pred_f0, pred_voiced, hop_s, deltas_ms):
    lab_c = cents(np.where(label_f0 > 0, label_f0, np.nan), CENTS_REF_HZ)
    pre_c = cents(np.where(pred_f0 > 0, pred_f0, np.nan), CENTS_REF_HZ)
    lv = label_voiced & np.isfinite(lab_c)
    pv = pred_voiced & np.isfinite(pre_c)
    out = []
    for d_ms in deltas_ms:
        shifted = cents_at(pre_c, pv, (d_ms / 1000.0) / hop_s)
        m = lv & np.isfinite(shifted)
        out.append(np.abs(lab_c[m] - shifted[m]) if m.any() else np.empty(0))
    return out


def parabolic(deltas, errs):
    k = int(np.nanargmin(errs))
    if k == 0 or k == len(errs) - 1:
        return float(deltas[k]), 0.0
    y0, y1, y2 = errs[k - 1], errs[k], errs[k + 1]
    denom = y0 - 2 * y1 + y2
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float(deltas[k]), 0.0
    frac = 0.5 * (y0 - y2) / denom
    step = deltas[1] - deltas[0]
    return float(deltas[k] + frac * step), float(denom)


def measure(name, root, *, n_files, max_seconds, sr, hop, span_ms, raw):
    cls = get_pitch_dataset(name)
    if raw and hasattr(cls, OFFSET_ATTR):
        setattr(cls, OFFSET_ATTR, 0.0)
    ds = cls(root_dir=root, sample_rate=sr, hop_size=hop)
    idxs = sorted({round(i) for i in np.linspace(0, len(ds) - 1, min(n_files, len(ds)))})
    deltas = np.arange(-span_ms, span_ms + 1, 1.0)
    hop_s = hop / sr
    per_ref = {}
    for ref in REFERENCES:
        if not available(ref):
            continue
        pooled = [[] for _ in deltas]
        for k in idxs:
            s = ds[k]
            audio = s["audio"].numpy()
            if max_seconds:
                audio = audio[: int(max_seconds * sr)]
            f0l = np.asarray(s["pitch"]).reshape(-1)[: len(audio) // hop]
            vl = (f0l >= ds.fmin) & (f0l <= ds.fmax)
            conf = s.get("pitch_conf")
            if conf is not None:
                vl = vl & (np.asarray(conf).reshape(-1)[: len(f0l)] >= VOICED_THRESHOLD)
            f0p, vp = track(ref, audio, sr, hop, ds.fmin, ds.fmax, 0.5)
            f0p, vp = np.asarray(f0p, float), np.asarray(vp, float) > 0
            n = min(len(f0l), len(f0p))
            for j, e in enumerate(sweep_file(f0l[:n], vl[:n], f0p[:n], vp[:n], hop_s, deltas)):
                pooled[j].append(e)
        errs = np.array([np.median(np.concatenate(p)) if any(len(x) for x in p) else np.nan
                         for p in pooled])
        if np.isfinite(errs).sum() < 5:
            continue
        d, curv = parabolic(deltas, errs)
        per_ref[ref] = (d, float(np.nanmin(errs)), curv)
    return per_ref


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, metavar="NAME=DIR")
    p.add_argument("--n-files", type=int, default=20)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--span-ms", type=float, default=25.0)
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--hop", type=int, default=HOP_SIZE)
    p.add_argument("--raw", action="store_true",
                   help="zero the loader's applied correction: recovers the ORIGINAL offset "
                        "instead of the residual")
    args = p.parse_args()

    print(f"references: {', '.join(REFERENCES)}   {args.n_files} files x {args.max_seconds:g}s   "
          f"sweep +-{args.span_ms:g} ms @ 1 ms   mode={'RAW' if args.raw else 'residual'}\n")
    print(f"{'corpus':<16s} " + " ".join(f"{r:>8s}" for r in REFERENCES)
          + f" {'consensus':>10s} {'min_err':>9s} {'curv':>7s}  reading")
    for spec in args.data:
        name, _, root = spec.partition("=")
        try:
            per_ref = measure(name, root, n_files=args.n_files, max_seconds=args.max_seconds,
                              sr=args.sample_rate, hop=args.hop, span_ms=args.span_ms,
                              raw=args.raw)
        except Exception as e:
            print(f"{name:<16s} SKIP ({type(e).__name__}: {str(e)[:60]})")
            continue
        if not per_ref:
            print(f"{name:<16s} (no reference produced a usable sweep)")
            continue
        ds_ = [v[0] for v in per_ref.values()]
        cons = float(np.median(ds_))
        err = float(np.median([v[1] for v in per_ref.values()]))
        curv = float(np.median([v[2] for v in per_ref.values()]))
        spread = max(ds_) - min(ds_)
        if err > 20:
            read = "UNUSABLE: labels do not describe the audio (min err too high)"
        elif curv < 0.02:
            read = "FLAT sweep: argmin not trustworthy"
        elif spread > 4:
            read = f"references disagree ({spread:.1f} ms spread) -- treat as unreliable"
        elif abs(cons) < 1.0:
            read = "aligned (within noise of 0)"
        else:
            read = f"labels stamped {cons:+.1f} ms early -> add {cons / 1000:.4f} to label_times"
        cells = " ".join(f"{per_ref[r][0]:+8.2f}" if r in per_ref else f"{'--':>8s}"
                         for r in REFERENCES)
        print(f"{name:<16s} {cells} {cons:>+10.2f} {err:>8.1f}c {curv:>7.3f}  {read}")


if __name__ == "__main__":
    main()
