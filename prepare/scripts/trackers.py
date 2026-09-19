import importlib.util

import numpy as np

from grid import frame_times, is_voiced, resample_to_grid

CHUNK_SECONDS, CHUNK_OVERLAP_SECONDS = 20.0, 1.0


def _praat(audio, sample_rate, hop_size, fmin, fmax, _threshold):
    import parselmouth
    pitch = parselmouth.Sound(audio, sample_rate).to_pitch(
        time_step=hop_size / sample_rate, pitch_floor=fmin, pitch_ceiling=fmax)
    return (pitch.xs(), pitch.selected_array["frequency"],
            pitch.selected_array["strength"])


def _harvest(audio, sample_rate, hop_size, fmin, fmax, _threshold):
    import pyworld
    f0, t = pyworld.harvest(audio.astype(np.float64), sample_rate, f0_floor=fmin, f0_ceil=fmax,
                            frame_period=hop_size / sample_rate * 1000.0)
    return t, f0, (f0 >= fmin).astype(np.float32)


def _dio(audio, sample_rate, hop_size, fmin, fmax, threshold):
    import pyworld
    x = audio.astype(np.float64)
    f0, t = pyworld.dio(x, sample_rate, f0_floor=fmin, f0_ceil=fmax,
                        frame_period=hop_size / sample_rate * 1000.0,
                        allowed_range=0.02 + threshold * (0.2 - 0.02))
    return t, pyworld.stonemask(x, f0, t, sample_rate), (f0 >= fmin).astype(np.float32)


def _swipe(audio, sample_rate, hop_size, fmin, fmax, threshold):
    from pysptk import sptk
    f0 = sptk.swipe(audio, sample_rate, hop_size, min=fmin, max=fmax,
                    threshold=0.2 + threshold * (0.5 - 0.2), otype="f0")
    return (frame_times(len(f0), hop_size, sample_rate), f0,
            (f0 >= fmin).astype(np.float32))


def _degg(audio, sample_rate, hop_size, fmin, fmax, _threshold):
    from scipy.signal import butter, filtfilt, find_peaks
    times = frame_times(len(audio) // hop_size, hop_size, sample_rate)
    x = np.asarray(audio, dtype=float)
    x = x - np.mean(x)
    nyq = 0.5 * sample_rate
    b, a = butter(2, [max(1.0, 0.5 * fmin) / nyq, min(0.95, 3.0 * fmax / nyq)], btype="band")
    d = np.diff(filtfilt(b, a, x))
    if np.mean(d ** 3) < 0:
        d = -d
    pos = d[d > 0]
    if pos.size == 0:
        return times, np.zeros(len(times)), np.zeros(len(times), dtype=np.float32)
    ref = np.percentile(pos, 90)
    dist = max(1, int(sample_rate / fmax))
    gci, _ = find_peaks(d, distance=dist, height=0.30 * ref, prominence=0.20 * ref)
    if len(gci) >= 4:
        dist = max(dist, int(0.62 * np.median(np.diff(gci))))
        gci, _ = find_peaks(d, distance=dist, height=0.30 * ref, prominence=0.20 * ref)
    gci = gci[(gci > 0) & (gci < len(d) - 1)]
    y0, y1, y2 = d[gci - 1], d[gci], d[gci + 1]
    denom = y0 - 2 * y1 + y2
    gci = gci + np.clip(np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0), -0.5, 0.5)

    f0 = np.zeros(len(times))
    voiced = np.zeros(len(times), dtype=bool)
    if len(gci) >= 2:
        cyc = sample_rate / np.diff(gci)
        ok = (cyc >= fmin) & (cyc <= fmax)
        interval = np.searchsorted(gci, times * sample_rate) - 1
        inb = (interval >= 0) & (interval < len(cyc))
        sel = interval[inb]
        f0[inb] = np.where(ok[sel], cyc[sel], 0.0)
        voiced[inb] = ok[sel]
    return times, f0, voiced.astype(np.float32)


TRACKERS = {
    "Praat":   ("parselmouth", _praat,   "linear",  CHUNK_SECONDS),
    "Harvest": ("pyworld",     _harvest, "linear",  CHUNK_SECONDS),
    "DEGG":    ("scipy",       _degg,    "linear",  None),
    "DIO":     ("pyworld",     _dio,     "nearest", CHUNK_SECONDS),
    "SWIPE":   ("pysptk",      _swipe,   "nearest", None),
}


def available(name):
    return importlib.util.find_spec(TRACKERS[name][0]) is not None


def _windowed(raw, audio, sample_rate, hop_size, fmin, fmax, threshold, chunk_seconds):
    n = len(audio)
    chunk = (None if chunk_seconds is None
             else max(hop_size, int(chunk_seconds * sample_rate) // hop_size * hop_size))
    if chunk is None or n <= chunk:
        return raw(audio, sample_rate, hop_size, fmin, fmax, threshold)
    overlap = max(hop_size, int(CHUNK_OVERLAP_SECONDS * sample_rate) // hop_size * hop_size)
    out, start = [], 0
    while start < n:
        end = min(start + chunk, n)
        cs, ce = max(0, start - overlap), min(n, end + overlap)
        t, p, q = raw(audio[cs:ce], sample_rate, hop_size, fmin, fmax, threshold)
        t = np.asarray(t, dtype=float) + cs / sample_rate
        keep = (t * sample_rate >= start - 0.5) & (t * sample_rate < (np.inf if end >= n
                                                                     else end - 0.5))
        out.append((t[keep], np.asarray(p)[keep], np.asarray(q)[keep]))
        start = end
    return tuple(np.concatenate(part) for part in zip(*out))


def track(name, audio, sample_rate, hop_size, fmin, fmax, threshold):
    _module, raw, voicing_kind, chunk_seconds = TRACKERS[name]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    times, pitch, conf = _windowed(raw, audio, sample_rate, hop_size, fmin, fmax,
                                   threshold, chunk_seconds)
    pitch = np.asarray(pitch, dtype=np.float64).copy()
    conf = np.asarray(conf, dtype=np.float64).copy()
    unusable = ~(np.isfinite(pitch) & np.isfinite(conf) & (pitch > 0) & (conf > 0))
    pitch[unusable], conf[unusable] = 0.0, 0.0
    pitch[~unusable] = np.clip(pitch[~unusable], fmin, fmax)
    pitch, conf = resample_to_grid(pitch, np.clip(conf, 0.0, 1.0), times,
                                   frame_times(len(audio) // hop_size, hop_size, sample_rate),
                                   voicing_kind)
    voiced = conf >= threshold if voicing_kind == "linear" else is_voiced(conf)
    return np.where(voiced, pitch, 0.0), voiced
