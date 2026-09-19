import json
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from constants import (
    DIRECT_HALF_MS,
    ENVELOPE_WIN_S,
    FADE_MS,
    MEDLEYDB_TRACKS,
    NOISE_ALPHA_RANGE,
    NOISE_FLOOR_TAIL_FRAC,
    NOISE_FMIN_HZ,
    POOL_AUDIO_NAME,
    POOL_INDEX_NAME,
    POOL_SUBTYPE,
    RMS_FLOOR,
    ROOM_DRR_SIGMA_DB,
    ROOM_DRR_TARGET_DB,
    ROOM_MAX_SECONDS,
    ROOM_MIN_SUPPORT_S,
    ROOM_T20_SNR_MARGIN_DB,
    ROOM_T60_LOG_SIGMA,
    ROOM_T60_TARGET_S,
    SAMPLE_RATE,
    SCENE_ROOMS,
    SCENE_SOURCES,
    T20_END_DB,
    TRUNCATE_ABOVE_FLOOR_DB,
    TRUNCATED_SNR_BIAS_DB,
)
from stages import MIN_CLIP_POWER, direct_path_index, signal_rms

LOG_RATIO_FLOOR = 1e-30
MIN_DECAY_SLOPE = -1e-9
RIR_MIN_RMS = 1e-6


def peak_normalized(audio):
    audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    return audio / (peak if peak > 0.0 else 1.0)


def load_audio(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def decode(path, sample_rate, channel="mix"):
    y, sr = load_audio(path)
    if channel == "left":
        if y.shape[0] < 2:
            return None
        y = y[0:1]
    else:
        y = y.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        y = torchaudio.functional.resample(y, sr, sample_rate)
    return y.squeeze(0).numpy().astype(np.float32)


def spans_levels(x, min_levels):
    if not min_levels:
        return True
    step = np.diff(np.unique(np.asarray(x, dtype=np.float32)))
    return len(step) > 0 and signal_rms(x) / step.min() >= min_levels


def normalize_power(x):
    power = float(np.mean(x ** 2))
    return x / np.sqrt(power) if power > 0 else None


def _admit(path, min_levels):
    audio = decode(path, SAMPLE_RATE)
    power = float(np.mean(audio ** 2))
    if power > MIN_CLIP_POWER and spans_levels(audio, min_levels):
        return audio / np.sqrt(power)
    return None


def select_grouped_files(root, glob, group_re, per_group, select_seed):
    pattern = re.compile(group_re)
    files_by_group = {}
    for path in sorted(root.glob(glob)):
        rel = path.relative_to(root).as_posix()
        m = pattern.search(rel)
        if m is None:
            raise ValueError(f"{group_re!r} does not match {rel!r}")
        files_by_group.setdefault(m.group(1), []).append(path)
    if not files_by_group:
        raise FileNotFoundError(f"no {glob} files under {root}")
    rng = np.random.default_rng(select_seed)
    out = []
    for gid in sorted(files_by_group):
        files = sorted(files_by_group[gid])
        picks = sorted(rng.choice(len(files), min(per_group, len(files)), replace=False))
        out.append((gid, [files[i] for i in picks]))
    return out


def grouped_clips(entry, root):
    for _gid, files in select_grouped_files(root, entry["glob"], entry["group_re"],
                                            entry["per_group"], entry["select_seed"]):
        for path in files:
            audio = _admit(path, entry.get("min_levels"))
            if audio is not None:
                yield audio, {}


def grouped_pool(entry, root):
    by_group, group_ids = [], []
    for gid, files in select_grouped_files(root, entry["glob"], entry["group_re"],
                                           entry["per_group"], entry["select_seed"]):
        clips = [a for a in (_admit(p, entry.get("min_levels")) for p in files) if a is not None]
        if clips:
            by_group.append(clips)
            group_ids.append(gid)
    return by_group, group_ids


def accompaniment(path, streams):
    import av

    with av.open(path) as container:
        picked = [container.streams.audio[i] for i in streams]
        rate = picked[0].rate
        to_stereo = {s.index: av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo", rate=rate) for s in picked}
        parts = {s.index: [] for s in picked}
        for packet in container.demux(picked):
            for frame in packet.decode():
                for out in to_stereo[packet.stream.index].resample(frame):
                    parts[packet.stream.index].append(out.to_ndarray())
        lengths = []
        for s in picked:
            for out in to_stereo[s.index].resample(None):
                parts[s.index].append(out.to_ndarray())
            if s.duration is None:
                raise SystemExit(f"{path}: stream {s.index} has no duration, so the "
                                 f"decoder's trailing padding cannot be told from the music")
            lengths.append(min(int(Fraction(s.duration) * s.time_base * rate),
                               sum(p.shape[1] for p in parts[s.index])))
        n = min(lengths)
        mixed = sum(np.concatenate(parts[s.index], axis=1)[:, :n] for s in picked)

    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(mixed), format="fltp",
                                       layout="stereo")
    frame.rate = rate
    to_mono = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = [out.to_ndarray()[0] for out in to_mono.resample(frame)]
    chunks += [out.to_ndarray()[0] for out in to_mono.resample(None)]
    return np.concatenate(chunks)


def stem_clips(entry, root):
    suffix = entry["glob"].rsplit("*", 1)[-1]
    for path in sorted(root.glob(entry["glob"])):
        if path.name[:-len(suffix)] not in MEDLEYDB_TRACKS:
            x = accompaniment(str(path), entry["streams"])
            y = normalize_power(x) if spans_levels(x, entry.get("min_levels")) else None
            if y is not None:
                yield y, {}


def _segments_of(n, sample_rate, seg_seconds):
    seg = int(seg_seconds * sample_rate)
    out, start = [], 0
    while start + seg <= n:
        out.append((start, start + seg))
        start += seg
    if n - start >= seg // 2:
        out.append((start, n))
    return out


def segment_clips(entry, root):
    wav_dir = root / entry["subdir"] if entry.get("subdir") else root
    files = sorted(wav_dir.rglob(entry["glob"]))
    if not files:
        raise FileNotFoundError(f"no {entry['glob']} files under {wav_dir}")
    segments = []
    for path in files:
        audio = decode(path, SAMPLE_RATE, entry["channel"])
        if audio is None:
            continue
        for start, end in _segments_of(len(audio), SAMPLE_RATE, entry["seg_seconds"]):
            segments.append(audio[start:end])
    return segments


def colored_noise(length, rng, alpha, sample_rate=SAMPLE_RATE, fmin_hz=NOISE_FMIN_HZ):
    f = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    fmin = max(fmin_hz, sample_rate / length)
    scale = np.maximum(f, fmin) ** (-alpha / 2.0)
    spec = np.fft.rfft(rng.standard_normal(length)) * scale
    spec[0] = 0.0
    n = np.fft.irfft(spec, n=length).astype(np.float32)
    return n / (signal_rms(n) + RMS_FLOOR)


def draw_colored(length, rng):
    alpha = float(rng.uniform(*NOISE_ALPHA_RANGE))
    return colored_noise(length, rng, alpha), {"alpha": round(alpha, 3)}


def load_rir_file(path, sample_rate):
    h, sr = load_audio(path)
    h = h[0:1]
    if sr != sample_rate:
        h = torchaudio.functional.resample(h, sr, sample_rate)
    return h.squeeze(0).numpy().astype(np.float32)


def _rir_files_openair(entry, root):
    rng = np.random.default_rng(entry["select_seed"])
    out = []
    for env in sorted(p for p in root.iterdir() if p.is_dir()):
        if env.name in entry["exclude"]:
            continue
        files = []
        for fmt in entry["formats"]:
            files = sorted((env / fmt).rglob("*.wav"))
            if files:
                break
        if len(files) > entry["per_env"]:
            keep = sorted(rng.choice(len(files), entry["per_env"], replace=False))
            files = [files[i] for i in keep]
        out.extend(files)
    return out


def rir_files(entry, root):
    if entry["kind"] == "openair":
        return _rir_files_openair(entry, root)
    return sorted((root / entry["subdir"] if entry.get("subdir") else root).glob(entry["glob"]))


def rir_clips(entry, root):
    for path in rir_files(entry, root):
        h = load_rir_file(path, SAMPLE_RATE)
        if float(np.mean(h ** 2)) > 0:
            yield h, {"p": direct_path_index(h)}


def rir_snr_db(h, sample_rate):
    e = np.asarray(h, np.float64) ** 2
    n = len(e)
    tail = e[int(NOISE_FLOOR_TAIL_FRAC * n):]
    floor = float(np.mean(tail)) if len(tail) else 0.0
    win = max(1, int(ENVELOPE_WIN_S * sample_rate))
    env = np.convolve(e, np.ones(win) / win, mode="same")
    peak = float(env.max()) if n else 0.0
    if floor <= 0.0 or peak <= 0.0:
        return float("inf")
    return 10.0 * np.log10(peak / floor)


def rir_t60(h, sample_rate, snr_db=None):
    e = np.asarray(h, np.float64) ** 2
    sch = np.cumsum(e[::-1])[::-1]
    total = float(sch[0]) if len(sch) else 0.0
    if total <= 0.0:
        return float("nan")
    if snr_db is None:
        snr_db = rir_snr_db(h, sample_rate) + TRUNCATED_SNR_BIAS_DB
    if snr_db < T20_END_DB + ROOM_T20_SNR_MARGIN_DB:
        return float("nan")
    sch = 10.0 * np.log10(np.maximum(sch / total, LOG_RATIO_FLOOR))
    i5 = int(np.argmax(sch <= -5.0))
    i25 = int(np.argmax(sch <= -T20_END_DB))
    if i25 <= i5 + 3:
        return float("nan")
    t = np.arange(i5, i25) / float(sample_rate)
    slope = float(np.polyfit(t, sch[i5:i25], 1)[0])
    return float("nan") if slope >= MIN_DECAY_SLOPE else -60.0 / slope


def rir_drr(h, sample_rate):
    h = np.asarray(h, np.float64)
    idx = direct_path_index(h)
    half = round(DIRECT_HALF_MS * 1e-3 * sample_rate)
    lo, hi = max(idx - half, 0), min(idx + half + 1, len(h))
    direct = float(np.sum(h[lo:hi] ** 2))
    rev = float(np.sum(h ** 2)) - direct
    return 10.0 * np.log10(direct / max(rev, LOG_RATIO_FLOOR))


def truncate_noise_floor(h, sample_rate):
    h = np.asarray(h, np.float64)
    n = len(h)
    tail = h[int(NOISE_FLOOR_TAIL_FRAC * n):] ** 2
    floor = float(np.mean(tail)) if len(tail) else 0.0
    if floor <= 0:
        return h
    win = max(1, int(ENVELOPE_WIN_S * sample_rate))
    env = np.convolve(h ** 2, np.ones(win) / win, mode="same")
    above = np.nonzero(env > floor * 10 ** (TRUNCATE_ABOVE_FLOOR_DB / 10))[0]
    if not len(above):
        return h
    cut = min(n, int(above[-1]) + win)
    if cut >= n:
        return h
    out = h[:cut].copy()
    fade = min(cut, int(FADE_MS * 1e-3 * sample_rate))
    if fade > 1:
        out[cut - fade:] *= np.linspace(1.0, 0.0, fade)
    return out


def rir_group(rir_id):
    parts = str(rir_id).split("/")
    if parts[0] == "openair":
        return "/".join(parts[:2])
    stem = parts[-1].rsplit(".", 1)[0]
    stem = re.sub(r"_imp\d+$", "", stem)
    stem = re.sub(r"_(far|near)_angl[a-z]$", "", stem)
    return f"{parts[0]}/{stem}"


def rir_weights(rirs, ids, sample_rate, snr_db=None):
    w = np.ones(len(rirs), np.float64)
    t60_w = np.full(len(rirs), np.nan)
    for i, h in enumerate(rirs):
        t60 = rir_t60(h, sample_rate, None if snr_db is None else float(snr_db[i]))
        drr = rir_drr(h, sample_rate)
        if np.isfinite(t60) and t60 > 0.0:
            t60_w[i] = np.exp(-0.5 * ((np.log(t60) - np.log(ROOM_T60_TARGET_S))
                                      / ROOM_T60_LOG_SIGMA) ** 2)
        if np.isfinite(drr):
            w[i] *= np.exp(-0.5 * ((drr - ROOM_DRR_TARGET_DB) / ROOM_DRR_SIGMA_DB) ** 2)
    measurable = np.isfinite(t60_w)
    w *= np.where(measurable, t60_w, t60_w[measurable].mean() if measurable.any() else 1.0)
    groups = [rir_group(i) for i in ids]
    per_group = Counter(groups)
    w = np.array([w[i] / per_group[groups[i]] for i in range(len(rirs))], np.float64)
    total = float(w.sum())
    if total <= 0.0:
        return np.full(len(rirs), 1.0 / len(rirs))
    return w / total


class SegmentPool:

    def __init__(self, segments):
        self.cache = []
        for s in segments:
            s = np.asarray(s, dtype=np.float32)
            power = float(np.mean(s ** 2))
            if power > MIN_CLIP_POWER:
                self.cache.append(s / np.sqrt(power))
        if not self.cache:
            raise RuntimeError("SegmentPool: no usable segments")

    def __call__(self, length, rng):
        noise = self.cache[int(rng.integers(len(self.cache)))]
        n = len(noise)
        if n >= length:
            start = int(rng.integers(0, n - length + 1))
            return noise[start : start + length], {}
        reps = (length + n - 1) // n
        return np.tile(noise, reps)[:length], {}


class GroupedPool:
    def __init__(self, clips_by_group, group_ids):
        if len(clips_by_group) != len(group_ids) or not group_ids:
            raise RuntimeError("GroupedPool: empty or mismatched group lists")
        for gid, clips in zip(group_ids, clips_by_group):
            if not clips:
                raise RuntimeError(f"GroupedPool: group {gid} has no clips")
        self.clips = [list(c) for c in clips_by_group]
        self.group_ids = list(group_ids)

    def __call__(self, length, rng):
        g = int(rng.integers(len(self.clips)))
        clips = self.clips[g]
        order = rng.permutation(len(clips))
        cat = np.concatenate([clips[i] for i in order])
        start = int(rng.integers(0, max(len(cat) - length, 0) + 1))
        if len(cat) < length:
            reps = (length + len(cat) - 1) // len(cat)
            cat = np.tile(cat, reps)
        return cat[start:start + length], {"group": self.group_ids[g]}


@dataclass(frozen=True)
class SceneSource:
    name: str
    generator: object
    point: bool
    label_safe: bool
    weight: float


def load_scene(raw_root):
    sources = []
    for entry in SCENE_SOURCES:
        kind = entry["kind"]
        if kind == "colored":
            generator = draw_colored
        elif kind == "segments":
            clips = segment_clips(entry, raw_root / entry["root"])
            if not clips:
                raise ValueError(f"{entry['name']}: no admitted clips")
            generator = SegmentPool(clips)
        elif kind == "grouped":
            groups, ids = grouped_pool(entry, raw_root / entry["root"])
            if not ids:
                raise ValueError(f"{entry['name']}: no admitted groups")
            generator = GroupedPool(groups, ids)
        else:
            raise ValueError(f"Unknown scene kind: {kind}")
        sources.append(SceneSource(entry["name"], generator, point=entry["point"],
                                   label_safe=entry["label_safe"],
                                   weight=entry.get("weight", 1.0)))
    return tuple(sources)


CLIP_KINDS = {"grouped": grouped_clips, "stems": stem_clips, "rirs": rir_clips}


def build_pool(entry, pools_root, raw_root):
    root = Path(raw_root) / entry["root"]
    out_dir = Path(pools_root) / entry["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets, extras = [0], []
    with sf.SoundFile(str(out_dir / POOL_AUDIO_NAME), "w", samplerate=SAMPLE_RATE,
                      channels=1, format="FLAC", subtype=POOL_SUBTYPE) as out:
        for audio, meta in CLIP_KINDS[entry["kind"]](entry, root):
            audio = peak_normalized(audio)
            out.write(audio)
            offsets.append(offsets[-1] + audio.size)
            extras.append(meta)
    if not extras:
        raise SystemExit(f"{entry['name']}: nothing admitted under {root}")
    index = {"offsets": offsets}
    index.update({key: [m.get(key) for m in extras]
                  for key in sorted({k for m in extras for k in m})})
    (out_dir / POOL_INDEX_NAME).write_text(json.dumps(index))
    print(f"{entry['name']}: {len(extras)} clips, {offsets[-1]} samples", flush=True)


class Rooms:
    def __init__(self, rirs, ids, sample_rate=SAMPLE_RATE, snr_db=None):
        if not rirs:
            raise RuntimeError("rooms: no impulse responses")
        self.rirs = rirs
        self.ids = list(ids)
        self.sample_rate = int(sample_rate)
        self.snr_db = None if snr_db is None else np.asarray(snr_db, np.float64)
        self.weights = rir_weights(rirs, self.ids, self.sample_rate, self.snr_db)

    def __len__(self):
        return len(self.rirs)

    def draw(self, rng):
        i = int(rng.choice(len(self.rirs), p=self.weights))
        return self.rirs[i], self.ids[i]


def load_rooms(raw_root):
    max_len = int(ROOM_MAX_SECONDS * SAMPLE_RATE)
    half = round(DIRECT_HALF_MS * 1e-3 * SAMPLE_RATE)
    min_support = int(ROOM_MIN_SUPPORT_S * SAMPLE_RATE)
    rirs, ids, snrs = [], [], []
    for entry in SCENE_ROOMS:
        root = raw_root / entry["root"]
        files = rir_files(entry, root)
        if not files:
            raise FileNotFoundError(f"no RIRs under {root}")
        for path in files:
            h = load_rir_file(path, SAMPLE_RATE)
            if len(h) < 4 * half or signal_rms(h) <= RIR_MIN_RMS:
                continue
            p = direct_path_index(h)
            if float(np.sum(h[p + half:] ** 2)) <= 0:
                continue
            supported = truncate_noise_floor(h.astype(np.float64), SAMPLE_RATE)[:max_len]
            if len(supported) - p < min_support:
                continue
            rirs.append(supported.astype(np.float32))
            ids.append(f"{entry['name']}/{path.relative_to(root).as_posix()}")
            snrs.append(rir_snr_db(h.astype(np.float64), SAMPLE_RATE))
    if not rirs:
        raise RuntimeError("room admission left nothing")
    print(f"rooms: {len(rirs)} impulse responses", flush=True)
    return Rooms(rirs, ids, SAMPLE_RATE, np.array(snrs, np.float64))
