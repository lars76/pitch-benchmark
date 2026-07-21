"""Composable audio augmentation for the benchmark.

Structure:
  - NoiseSource: produces a unit-power noise segment  (length, rng) -> np.ndarray
  - transforms: callables                              (audio, sr, periodicity, hop, rng) -> audio
  - Augment(base, pipeline, seed): a transparent Dataset wrapper that applies a pipeline
    (a list of transforms) to `audio`, deterministically per (seed, idx) -> worker-safe.
  - Truncate(base, max_seconds): trims audio + labels (the sampling concern; cap uses the
    built-in torch.utils.data.Subset).
  - Condition subclasses: each .build(sr) -> a pipeline (list of transforms); CANONICAL is
    the fixed axis, BY_NAME maps a condition's derived .name to its object.

The real-noise conditions (chime, demand) are [AddNoise(SampleBank(non-pitched segments))], so the
voiced-aware-SNR logic lives once (in AddNoise) and is shared with the colored-noise paths. The
non-pitched segment banks are built by _nonpitched_{chime,demand}_segments (signal gate: _is_nonpitched).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.signal import butter, fftconvolve, sosfiltfilt
from torch.utils.data import Dataset, Subset

from metrics import (
    is_voiced,  # single definition of "voiced" (periodicity >= VOICED_THRESHOLD)
)

_EPS = 1e-9
_POWER_THRESHOLD = 1e-8


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _match_rms(y, audio):
    """Scale y to the RMS level of audio (the level-match convention shared by the transforms)."""
    return (y / (_rms(y) + _EPS) * _rms(audio)).astype(np.float32)


def _voiced_power(audio, periodicity, hop):
    """Signal power over voiced samples (falls back to whole-signal RMS^2)."""
    if periodicity is not None:
        per = is_voiced(np.asarray(periodicity).reshape(-1))   # periodicity is a [0,1] confidence
        if np.any(per):
            mask = np.repeat(per, hop)[: len(audio)]
            if mask.any():
                return float(np.mean(audio[: len(mask)][mask] ** 2) + 1e-12)
    return _rms(audio) ** 2


# --------------------------------------------------------------------------- #
# Non-pitched gate: real-noise banks (CHiMe, DEMAND) must not inject a competing f0. We determine
# "non-pitched" from the SIGNAL, not from dataset labels: CHiMe's cmfv tags are unreliable/inverted vs
# measured periodicity, and DEMAND has no such tags and is contaminated by steady tonal sources
# (engine/appliance/HVAC/birdsong). The in-band normalized-autocorrelation peak fraction below is
# validated (white/pink ~0.00, speech ~0.43); segments with vfrac < GATE_VFRAC are kept.
# --------------------------------------------------------------------------- #
GATE_FMIN, GATE_FMAX = 65.0, 500.0     # plausible pitch band for "competing f0" content
GATE_FRAME, GATE_HOP = 1024, 256
GATE_PEAK = 0.6                        # per-frame periodicity peak counted as "voiced"
GATE_VFRAC = 0.15                      # keep a segment if < this fraction of frames are voiced-in-band
BANK_CAP = 200                         # max segments per gated bank (bounds startup scan cost)
GATE_SEG_SECONDS = 4.0                 # segment length when slicing long recordings (DEMAND)


def _periodicity_vfrac(x, sample_rate):
    """Fraction of frames with a strong in-band periodicity peak (competing-pitch content)."""
    x = np.asarray(x, dtype=float)
    lo, hi = int(sample_rate / GATE_FMAX), int(sample_rate / GATE_FMIN)
    n = (len(x) - GATE_FRAME) // GATE_HOP + 1
    if n < 1 or hi <= lo:
        return 0.0
    w = np.hanning(GATE_FRAME)
    voiced = 0
    for i in range(n):
        fr = (x[i * GATE_HOP:i * GATE_HOP + GATE_FRAME] - 0.0)
        fr = (fr - fr.mean()) * w
        F = np.fft.rfft(fr, 2048)
        ac = np.fft.irfft(F * np.conj(F))[:GATE_FRAME]
        peak = float(np.max(ac[lo:hi + 1])) / (ac[0] + 1e-12)
        voiced += peak > GATE_PEAK
    return voiced / n


def _is_nonpitched(seg, sample_rate):
    """Signal gate for real-noise banks: loud enough, and no strong in-band periodicity (so the
    segment injects noise without a competing f0). The single definition shared by both banks.
    Segments shorter than one gate frame are rejected outright: too short to measure periodicity
    honestly, and SampleBank would tile them into a repeating (pitched) pattern."""
    return (
        len(seg) >= GATE_FRAME
        and _rms(seg) > 1e-6
        and _periodicity_vfrac(seg, sample_rate) < GATE_VFRAC
    )


def _load_mono(path, sample_rate):
    """Load an audio file as a mono float32 array resampled to sample_rate."""
    y, sr = torchaudio.load(str(path))
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        # functional (not transforms.Resample) so we don't build a throwaway resampler module per
        # file; matches datasets/base._prepare_audio and the consensus generator.
        y = torchaudio.functional.resample(y, sr, sample_rate)
    return y.squeeze().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Noise sources: (length, rng) -> unit-power float32 array
# --------------------------------------------------------------------------- #
class ColoredNoise:
    """White (alpha=0), pink (1) or brown (2) noise, FFT-shaped to unit power.

    Timmer-Koenig 1/f^alpha shaping (the half-spectrum is scaled by f^(-alpha/2)), with two
    conventions that keep it honest for a pitch benchmark (cf. the `colorednoise` package):
      - the DC bin is zeroed, so the noise has no spurious mean/offset and DC is excluded from the
        power normalization (a raw 1/f^(alpha/2) at f->0 would inject a large constant offset);
      - the 1/f divergence is bounded by a low-frequency floor (flat below `fmin_hz`), so pink/brown
        do not dump most of their power into sub-audible rumble below the pitch range, which would
        make the audible-band SNR far milder than the nominal SNR the condition asks for.
    """

    def __init__(self, alpha: float, sample_rate: int = 16000, fmin_hz: float = 20.0):
        self.alpha = float(alpha)
        self.sample_rate = int(sample_rate)
        self.fmin_hz = float(fmin_hz)

    def __call__(self, length, rng):
        f = np.fft.rfftfreq(length, d=1.0 / self.sample_rate)     # Hz
        fmin = max(self.fmin_hz, self.sample_rate / length)       # cannot resolve below one bin
        scale = np.maximum(f, fmin) ** (-self.alpha / 2.0)        # flat below fmin -> bounded 1/f
        spec = np.fft.rfft(rng.standard_normal(length)) * scale
        spec[0] = 0.0                                             # zero DC: no offset, excluded from power
        n = np.fft.irfft(spec, n=length).astype(np.float32)
        return n / (_rms(n) + _EPS)


class SampleBank:
    """Random unit-power segment from a cached bank of pre-loaded noise arrays (the signal-gated
    non-pitched CHiMe/DEMAND segments). Segments arrive already mono + resampled via _load_mono."""

    def __init__(self, segments):
        self.cache = []
        for s in segments:
            s = np.asarray(s, dtype=np.float32)
            power = float(np.mean(s ** 2))
            if power > _POWER_THRESHOLD:
                self.cache.append(s / np.sqrt(power))
        if not self.cache:
            raise RuntimeError("SampleBank: no usable noise segments")

    def __call__(self, length, rng):
        noise = self.cache[int(rng.integers(len(self.cache)))]
        n = len(noise)
        if n >= length:
            start = int(rng.integers(0, n - length + 1))
            return noise[start : start + length]
        # Short bank clip: tile to cover the length. The tile seam is a mild discontinuity (a weak
        # click at the repeat period, well below the pitch range); acceptable for a noise distractor.
        reps = (length + n - 1) // n
        return np.tile(noise, reps)[:length]


# --------------------------------------------------------------------------- #
# Transforms: (audio, sr, periodicity, hop, rng) -> audio
# --------------------------------------------------------------------------- #
class AddNoise:
    """Add a noise source at a fixed voiced-aware SNR (dB)."""

    def __init__(self, source, snr_db=0.0):
        self.source = source
        self.snr_db = snr_db

    def __call__(self, audio, sr, periodicity, hop, rng):
        noise = self.source(len(audio), rng)
        noise_power = float(np.mean(noise ** 2))
        if noise_power < _POWER_THRESHOLD:
            return audio
        sig_power = _voiced_power(audio, periodicity, hop)
        # SNR enters as a multiplier of the signal power, so a silent / near-silent clip gets ~zero
        # added noise (silent-in -> silent-out), the same behavior as torchaudio.add_noise and
        # audiomentations. The noise denominator is guarded above (_POWER_THRESHOLD); there is no
        # signal-power floor, by design (a floor would fabricate a fixed noise level on silent clips).
        scale = np.sqrt(sig_power / (10 ** (self.snr_db / 10) * noise_power + _EPS))
        return (audio + scale * noise).astype(np.float32)


class Gain:
    """Static level change (dB). Attenuation flows through untouched (the eval-side
    peak-normalize triggers only above 1.0), the absolute-level robustness condition,
    catching trackers whose voicing depends on absolute signal level."""

    def __init__(self, db):
        self.scale = 10.0 ** (db / 20.0)

    def __call__(self, audio, sr, periodicity, hop, rng):
        return (audio * self.scale).astype(np.float32)


class Fade:
    """Linear-in-dB gain ramp across the clip (db_start -> db_end): the AGC/fade-out class.
    A DISTINCT failure mode from static Gain: it catches trackers whose voicing decision is
    relative to a GLOBAL clip statistic (per-frame level invariance does not help against a
    within-clip ramp)."""

    def __init__(self, db_start=0.0, db_end=-40.0):
        self.db_start = float(db_start)
        self.db_end = float(db_end)

    def __call__(self, audio, sr, periodicity, hop, rng):
        g = 10.0 ** (np.linspace(self.db_start, self.db_end, len(audio)) / 20.0)
        return (audio * g).astype(np.float32)


class Reverb:
    """Synthetic exponential-decay RIR convolution, level-matched to the dry signal.

    ``rir[0] = 1`` puts the direct path at t=0 so onsets stay aligned with the frame labels (the
    standard alignment-preserving RIR convention). The decay tail rings the clip's own pitch a few
    hundred ms past voicing offsets, so this condition is only borderline label-preserving there."""

    def __init__(self, rt60: float = 0.4):
        self.rt60 = rt60

    def __call__(self, audio, sr, periodicity, hop, rng):
        n = max(1, int(self.rt60 * sr))
        t = np.arange(n) / sr
        rir = (rng.standard_normal(n) * np.exp(-6.9 * t / self.rt60)).astype(np.float32)
        rir[0] = 1.0
        y = fftconvolve(audio, rir)[: len(audio)].astype(np.float32)
        return _match_rms(y, audio)


class BandLimit:
    """Band-pass (e.g. telephone 300-3400 Hz): kills low fundamentals + the top band."""

    def __init__(self, low: float, high: float):
        self.low, self.high = low, high

    def __call__(self, audio, sr, periodicity, hop, rng):
        nyq = sr / 2.0
        lo = min(max(self.low, 1.0), nyq - 1.0) / nyq
        hi = min(self.high, nyq - 1.0) / nyq
        if hi <= lo:  # band collapses at very low sample rates -> leave audio unchanged
            return audio.astype(np.float32)
        # Zero-phase (sosfiltfilt) so the band-limited audio stays time-aligned with the frame labels;
        # a causal lfilter would add frequency-dependent group delay (~1-2 ms, up to ~0.1 frame) on
        # this condition alone. SOS form for numerical stability at the doubled effective order.
        sos = butter(4, [lo, hi], btype="band", output="sos")
        y = sosfiltfilt(sos, audio).astype(np.float32)
        return _match_rms(y, audio)


class RoomReverb:
    """Physics-based room reverberation via a pyroomacoustics image-source RIR (shoebox room).

    Distinct in character from the synthetic exponential ``Reverb`` (discrete early reflections +
    colored diffuse decay; the two only moderately correlate). The direct-path peak is aligned to
    index 0 so onsets stay aligned with the frame labels. The RIR is deterministic (fixed geometry)
    and built once per instance and sample rate, then cached; pyroomacoustics is imported lazily so it
    is only needed when this condition is actually used."""

    def __init__(self, room_dim=(6.0, 5.0, 3.0), rt60: float = 0.5):
        self.room_dim = tuple(float(d) for d in room_dim)
        self.rt60 = float(rt60)
        self._rir = None
        self._rir_sr = None

    def _make_rir(self, sr):
        import pyroomacoustics as pra

        e_abs, max_order = pra.inverse_sabine(self.rt60, self.room_dim)
        room = pra.ShoeBox(self.room_dim, fs=sr, materials=pra.Material(e_abs),
                           max_order=min(max_order, 12))
        room.add_source([d * 0.25 for d in self.room_dim])
        room.add_microphone([self.room_dim[0] * 0.7, self.room_dim[1] * 0.7, 1.5])
        room.compute_rir()
        rir = np.asarray(room.rir[0][0], dtype=np.float32)
        rir = rir[int(np.argmax(np.abs(rir))):]                  # align direct-path peak to index 0
        return rir / (np.max(np.abs(rir)) + _EPS)

    def __call__(self, audio, sr, periodicity, hop, rng):
        if self._rir is None or self._rir_sr != sr:
            self._rir = self._make_rir(sr)
            self._rir_sr = sr
        y = fftconvolve(audio, self._rir)[: len(audio)].astype(np.float32)
        return _match_rms(y, audio)


# --------------------------------------------------------------------------- #
# Per-item seeding (THE seeding idiom for generated randomness)
# --------------------------------------------------------------------------- #
def item_seed(seed: int, *ids: int) -> int:
    """A stable uint32 seed that is a pure function of (seed, *ids).

    THE seeding idiom for anything that generates per-item randomness (augmentation transforms,
    SpeechSynth word choice, OOD noise clips): derive one private seed per item instead of drawing
    from the global RNG stream, so the item is independent of access order, cache state, worker
    processes, and any other code that consumes randomness. The uint32 derivation is part of the
    contract: changing it re-rolls every stream built on it and invalidates benchmark results."""
    return int(
        np.random.SeedSequence([int(seed), *(int(i) for i in ids)])
        .generate_state(1, dtype=np.uint32)[0]
    )


def item_rng(seed: int, *ids: int) -> np.random.Generator:
    """A private numpy Generator per item, seeded by item_seed(seed, *ids). See item_seed."""
    return np.random.default_rng(item_seed(seed, *ids))


# --------------------------------------------------------------------------- #
# Dataset wrappers
# --------------------------------------------------------------------------- #
EVAL_ATTRS = ("sample_rate", "hop_size", "fmin", "fmax")   # attrs the eval loop reads off a dataset
FRAME_KEYS = ("pitch", "periodicity", "pitch_conf")        # frame-aligned label keys a wrapper must slice


def _copy_eval_attrs(dst, src):
    """Forward the eval-loop attrs (grid + pitch range) from a base dataset to a wrapper."""
    for attr in EVAL_ATTRS:
        setattr(dst, attr, getattr(src, attr))


class Augment(Dataset):
    """Apply a pipeline (list of transforms) to a base dataset's audio, deterministically.

    Two seeding modes, selected by `epoch`:
      - epoch is None (default, EVAL): the per-item stream is item_rng(seed, idx), FROZEN across
        accesses, so two benchmark runs at the same seed produce bit-identical degraded audio. This
        is the reproducibility contract the benchmark depends on; leave epoch=None for eval.
      - epoch is an int (TRAINING): the stream is item_rng(seed, epoch, idx), so every epoch re-rolls
        every transform's randomness (fresh noise / SNR / condition choice each pass) while staying a
        PURE function of (seed, epoch, idx), reproducible and resumable (set_epoch(N) after a
        checkpoint continues the identical stream), independent of shuffle order, worker count and
        cache state. Call set_epoch(e) once per epoch before iterating.

    Adding the epoch key changes the derived seed, so the two modes are intentionally distinct
    streams; the eval (None) path derives from exactly (seed, idx), the stream every eval
    result is built on, so it must never change. An empty pipeline is a pass-through (the
    'clean' condition)."""

    def __init__(self, base_dataset, pipeline, seed: int = 0, epoch: int | None = None):
        self.base_dataset = base_dataset
        self.pipeline = list(pipeline)
        self.seed = int(seed)
        self.epoch = None if epoch is None else int(epoch)
        _copy_eval_attrs(self, base_dataset)

    def set_epoch(self, epoch: int) -> None:
        """Switch to training seeding for `epoch` (see class docstring). Call once per epoch, before
        the DataLoader iterator is created, so re-forked workers pick up the new epoch. With
        persistent_workers=True a worker holds a stale copy and never sees this; use
        persistent_workers=False for training (the decode cache is off there anyway)."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base_dataset)

    def get_group(self, idx):
        """Delegate the leakage-safe cluster id to the wrapped dataset (1:1 index)."""
        return self.base_dataset.get_group(idx)

    def __getitem__(self, idx):
        if not self.pipeline:
            return self.base_dataset[idx]
        sample = dict(self.base_dataset[idx])  # shallow copy: don't mutate a cached base dict

        audio = sample["audio"]
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        x = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)

        # None -> eval: item_rng(seed, idx), the frozen stream eval results are built on.
        # int  -> training: fold the epoch in so each epoch re-rolls, still pure(seed, epoch, idx).
        rng = (
            item_rng(self.seed, idx)
            if self.epoch is None
            else item_rng(self.seed, self.epoch, idx)
        )

        per = sample.get("periodicity")
        per_np = per.detach().cpu().numpy() if per is not None else None
        for transform in self.pipeline:
            x = transform(x, self.sample_rate, per_np, self.hop_size, rng)
        # Peak-normalize (not hard-clip) so adding noise can't inject clipping distortion
        # on top of the intended condition, which would confound the robustness deltas.
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.0:
            x = x / peak

        sample["audio"] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return sample


class _Subset(Subset):
    """torch.utils.data.Subset that also forwards get_group, REMAPPING the subset-local index back to
    the base index (subset does not preserve indices, unlike the 1:1 Augment/Truncate wrappers)."""

    def get_group(self, idx):
        return self.dataset.get_group(self.indices[idx])


def subset(base_dataset, indices):
    """torch.utils.data.Subset (the cap) but forwarding the attrs + get_group the eval loop needs."""
    sub = _Subset(base_dataset, list(indices))
    _copy_eval_attrs(sub, base_dataset)
    return sub


class Truncate(Dataset):
    """Trim each clip to the first `max_seconds` of audio starting at its first voiced frame,
    trimming every frame-aligned label (FRAME_KEYS) to keep len == samples // hop_size, and
    shifting/clipping the time-based `notes` to the kept window."""

    def __init__(self, base_dataset, max_seconds: float):
        self.base_dataset = base_dataset
        self.max_seconds = max_seconds
        _copy_eval_attrs(self, base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def get_group(self, idx):
        """Delegate the leakage-safe cluster id to the wrapped dataset (1:1 index)."""
        return self.base_dataset.get_group(idx)

    def __getitem__(self, idx):
        sample = dict(self.base_dataset[idx])  # shallow copy: don't mutate a cached base dict
        audio = sample["audio"]
        if audio.dim() > 1:
            audio = audio.squeeze(0)

        n_frames = int(self.max_seconds * self.sample_rate / self.hop_size)
        start_f = 0
        periodicity = sample.get("periodicity")
        if periodicity is not None:
            voiced = torch.nonzero(is_voiced(periodicity.reshape(-1)))   # periodicity is a [0,1] confidence
            if voiced.numel():
                start_f = int(voiced[0])
        s0 = start_f * self.hop_size

        sample["audio"] = audio[s0 : s0 + n_frames * self.hop_size].clone()
        for key in FRAME_KEYS:
            if sample.get(key) is not None:
                sample[key] = sample[key][start_f : start_f + n_frames].clone()
        if sample.get("notes") is not None:
            # Notes are time-based (seconds), not frame-aligned: keep the ones overlapping the kept
            # window, re-zeroed to its clock and clamped to its bounds.
            t0 = s0 / self.sample_rate
            t1 = (s0 + n_frames * self.hop_size) / self.sample_rate
            sample["notes"] = [
                {**n, "start": max(n["start"] - t0, 0.0), "end": min(n["end"], t1) - t0}
                for n in sample["notes"]
                if n["end"] > t0 and n["start"] < t1
            ]
        return sample


# --------------------------------------------------------------------------- #
# Condition registry: name -> a builder returning the transform pipeline.
# Mirrors datasets/_PITCH_REGISTRY (one explicit dict, no auto-registration). The default benchmark
# conditions (the benchmark default; see evaluate.py) are the minimal-but-expressive set: an additive-noise family at a common
# SNR, reported per-source (white, pink, and the real-noise provenances chime + demand), plus
# telephone (low-cut) and reverb/room (convolutional). The real-noise banks are SIGNAL-gated to
# non-pitched segments (the gate above) so they stay label-preserving.
# --------------------------------------------------------------------------- #
def _collect_nonpitched(groups, sample_rate, cap, per_group=None):
    """Keep the non-pitched segments (signal gate) from an iterable of candidate groups: at most
    `per_group` kept from each group, stopping at `cap` total. The gate-and-cap loop shared by both
    real-noise banks. `per_group` bounds any single recording's share so the bank spans categories
    (DEMAND, one long recording per environment); pass None to draw freely from a single group (CHiMe,
    where each candidate is already a distinct short chunk)."""
    kept = []
    for group in groups:
        g = 0
        for seg in group:
            if _is_nonpitched(seg, sample_rate):
                kept.append(seg)
                g += 1
                if (per_group and g >= per_group) or len(kept) >= cap:
                    break
        if len(kept) >= cap:
            break
    return kept


def _nonpitched_chime_segments(chime_dir, sample_rate, cap=BANK_CAP):
    """Non-pitched CHiMe segments for the noise bank. Each chunk file is one candidate; scans chunks
    in sorted order until `cap` pass the signal gate."""
    chunks = Path(chime_dir) / "chunks"
    if not chunks.exists():
        raise ValueError(f"CHiME chunks directory not found: {chunks}")
    # CHiME-Home ships every chunk at both 16 kHz and 48 kHz; discovery is pinned to the 16 kHz
    # copies so ANY benchmark sample rate works (_load_mono resamples after load). rglob keeps
    # nested chunk layouts discoverable.
    suffix = ".16kHz.wav"

    def candidates():
        for wav in sorted(chunks.rglob(f"*{suffix}")):
            try:
                yield _load_mono(wav, sample_rate)
            except Exception:
                continue

    segs = _collect_nonpitched([candidates()], sample_rate, cap)   # one group: no per-recording cap
    if not segs:
        raise RuntimeError(f"No non-pitched CHiMe segments found under {chunks}")
    return segs


def _nonpitched_demand_segments(demand_dir, sample_rate, cap=BANK_CAP):
    """Non-pitched DEMAND segments for the noise bank. One channel per environment, sliced into fixed
    windows; a per-environment cap keeps the bank spanning categories rather than filling from one
    recording."""
    root = Path(demand_dir)
    if not root.exists():
        raise ValueError(f"DEMAND directory not found: {root}")
    seg_len = int(GATE_SEG_SECONDS * sample_rate)

    def env_slices(y):
        for s in range(0, len(y) - seg_len + 1, seg_len):
            yield y[s:s + seg_len]

    def groups():
        for env in sorted(p for p in root.iterdir() if p.is_dir()):
            wav = env / "ch01.wav"
            if not wav.exists():
                ws = sorted(env.glob("*.wav"))
                if not ws:
                    continue
                wav = ws[0]
            try:
                y = _load_mono(wav, sample_rate)
            except Exception:
                continue
            yield env_slices(y)   # one group per environment

    segs = _collect_nonpitched(groups(), sample_rate, cap, per_group=max(4, cap // 8))
    if not segs:
        raise RuntimeError(f"No non-pitched DEMAND segments found under {root}")
    return segs


class Codec:
    """G.711-style telephone codec, label-preserving: narrowband 8 kHz round-trip (linear-phase
    resample_poly, no group delay, honoring the same zero-phase contract as BandLimit) plus an
    8-bit mu-law companding round-trip (quantization distortion). Deterministic (rng unused).
    Complements 'telephone' (pure band-limit): codec adds the NONLINEAR quantization artifacts of
    real phone/VoIP chains while keeping f0 (<4 kHz) intact."""

    def __init__(self, mu: float = 255.0):
        self.mu = mu

    def __call__(self, audio, sr, periodicity, hop, rng):
        from scipy.signal import resample_poly

        x = np.asarray(audio, dtype=np.float64)
        n = len(x)
        if sr > 8000:
            x = resample_poly(resample_poly(x, 8000, sr), sr, 8000)
            x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
        x = np.clip(x, -1.0, 1.0)
        comp = np.sign(x) * np.log1p(self.mu * np.abs(x)) / np.log1p(self.mu)
        q = np.round((comp + 1.0) * 127.5) / 127.5 - 1.0        # 256 levels
        x = np.sign(q) * ((1.0 + self.mu) ** np.abs(q) - 1.0) / self.mu
        return x.astype(np.float32)


# --------------------------------------------------------------------------- #
# Conditions: typed degradations. A condition is fully specified by its parameters (FIELDS, not
# baked into a lambda or a name), and knows how to `build` its transform pipeline. `name` is a
# DERIVED string -- the object is the source of truth; the string exists only to cross boundaries
# an object cannot (cell filenames, subprocess argv, the CLI, cross-run comparability), and
# BY_NAME maps it back. The recorded conditions carry their machine-specific corpus as a field,
# so there is no separate location map -- a recorded condition is unrunnable without its corpus.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """The base condition is `clean`: no transform."""
    def build(self, sample_rate):
        return []

    @property
    def name(self):
        return "clean"


@dataclass(frozen=True)
class Noise(Condition):
    """Additive colored noise at a voiced-aware SNR. alpha 0 = white, 1 = pink."""
    alpha: float = 1.0
    snr_db: float = 0.0

    def build(self, sample_rate):
        return [AddNoise(ColoredNoise(self.alpha, sample_rate), snr_db=self.snr_db)]

    @property
    def name(self):
        base = "white" if self.alpha == 0.0 else "pink"
        return base if self.snr_db == 0.0 else f"{base}_snr{self.snr_db:+g}"


@dataclass(frozen=True)
class Recorded(Condition):
    """Additive real-world noise sampled from a recorded corpus (chime / demand) at 0 dB. The
    corpus lives at a machine-specific `corpus_dir`, so this condition cannot run without it --
    the dependency the type makes unwriteable."""
    which: str = "chime"
    corpus_dir: str | None = None

    def build(self, sample_rate):
        if not self.corpus_dir:
            raise ValueError(f"the {self.which} condition needs its noise corpus (corpus_dir)")
        return [AddNoise(SampleBank(_SCAN[self.which](self.corpus_dir, sample_rate)))]

    @property
    def name(self):
        return self.which


@dataclass(frozen=True)
class Telephone(Condition):
    """Narrowband band-limit, the pure passband of a phone line (no quantization -- see Codec)."""
    low: float = 300.0
    high: float = 3400.0

    def build(self, sample_rate):
        return [BandLimit(self.low, self.high)]

    @property
    def name(self):
        return "telephone"


@dataclass(frozen=True)
class Coded(Condition):
    """G.711-style narrowband + mu-law quantization (the nonlinear artifacts a phone chain adds)."""
    mu: float = 255.0

    def build(self, sample_rate):
        return [Codec(self.mu)]

    @property
    def name(self):
        return "codec"


@dataclass(frozen=True)
class Reverberation(Condition):
    """Synthetic exponential reverb (discrete early reflections + colored diffuse decay)."""
    rt60: float = 0.4

    def build(self, sample_rate):
        return [Reverb(self.rt60)]

    @property
    def name(self):
        return "reverb"


@dataclass(frozen=True)
class Room(Condition):
    """Physics-based room reverb via a pyroomacoustics image-source RIR."""
    rt60: float = 0.5
    room_dim: tuple = (6.0, 5.0, 3.0)

    def build(self, sample_rate):
        import importlib.util
        if importlib.util.find_spec("pyroomacoustics") is None:
            raise ValueError("the room condition requires pyroomacoustics (run `uv sync`)")
        return [RoomReverb(self.room_dim, self.rt60)]

    @property
    def name(self):
        return "room"


@dataclass(frozen=True)
class Gained(Condition):
    """Absolute level attenuation."""
    db: float = -40.0

    def build(self, sample_rate):
        return [Gain(self.db)]

    @property
    def name(self):
        return "gain" if self.db == -40.0 else f"gain{self.db:g}"


@dataclass(frozen=True)
class Faded(Condition):
    """A linear level ramp across the clip."""
    db_start: float = 0.0
    db_end: float = -40.0

    def build(self, sample_rate):
        return [Fade(self.db_start, self.db_end)]

    @property
    def name(self):
        return "fade"


# The recorded-corpus scanners, indexed by the recorded condition's `which`.
_SCAN = {"chime": _nonpitched_chime_segments, "demand": _nonpitched_demand_segments}

# THE fixed axis: the canonical 13 conditions, each fully specified by its explicit parameters.
# Recorded ones carry no corpus here -- the run binds it (Recorded's copy with corpus_dir set).
CANONICAL = (
    Condition(),
    Noise(alpha=0.0), Noise(alpha=1.0),                      # white, pink
    Recorded("chime"), Recorded("demand"),
    Telephone(), Coded(), Reverberation(), Room(),
    Gained(db=-40.0), Faded(),
    Noise(alpha=1.0, snr_db=10.0), Noise(alpha=1.0, snr_db=-5.0),   # pink_snr+10, pink_snr-5
)
BY_NAME = {c.name: c for c in CANONICAL}
assert len(BY_NAME) == len(CANONICAL), "condition names must be unique"
