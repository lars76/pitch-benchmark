import contextlib
import functools
import os
from abc import ABC, abstractmethod

import numpy as np

from resampling import resample_to_grid
from score import is_voiced

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "numba"))

PCM16_MAX = 32767.0


def salience_band_mask(cents_mapping, fmin, fmax):
    hz = 10.0 * 2.0 ** (np.asarray(cents_mapping, dtype=np.float64) / 1200.0)
    return (hz >= fmin) & (hz <= fmax)


def threshold_to_param(threshold, param_range):
    lo, hi = param_range
    return lo + float(threshold) * (hi - lo)


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    import torch
    import torchaudio

    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return torchaudio.functional.resample(x, orig_sr, target_sr).numpy()


_TF_CONFIGURED = False


class TensorFlowModelMixin:

    def _init_tensorflow(self):
        import tensorflow as tf

        global _TF_CONFIGURED
        if not _TF_CONFIGURED:
            tf.get_logger().setLevel("ERROR")
            with contextlib.suppress(Exception):
                tf.config.set_visible_devices([], "GPU")
            _TF_CONFIGURED = True


class PitchAlgorithm(ABC):

    CHUNK_SECONDS = 20.0
    CHUNK_OVERLAP_SECONDS = 1.0

    def __init__(self, sample_rate, hop_size, fmin, fmax):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

        if self.CHUNK_SECONDS is None:
            self.chunk_samples = None
        else:
            self.chunk_samples = max(
                hop_size, int(self.CHUNK_SECONDS * sample_rate) // hop_size * hop_size
            )
        self.chunk_overlap_samples = max(
            hop_size, int(self.CHUNK_OVERLAP_SECONDS * sample_rate) // hop_size * hop_size
        )

    def _extract_windowed(self, audio, raw_fn):
        n = len(audio)
        if self.chunk_samples is None or n <= self.chunk_samples:
            return raw_fn(audio)

        all_t, all_p, all_q = [], [], []
        start = 0
        while start < n:
            end = min(start + self.chunk_samples, n)
            cs = max(0, start - self.chunk_overlap_samples)
            ce = min(n, end + self.chunk_overlap_samples)
            t, p, q = raw_fn(audio[cs:ce])
            t = np.asarray(t, dtype=float) + cs / self.sample_rate
            ts = t * self.sample_rate
            lo = start - 0.5
            hi = np.inf if end >= n else end - 0.5
            keep = (ts >= lo) & (ts < hi)
            all_t.append(t[keep])
            all_p.append(np.asarray(p)[keep])
            all_q.append(np.asarray(q)[keep])
            start = end
        return np.concatenate(all_t), np.concatenate(all_p), np.concatenate(all_q)

    def native_frames(self, audio):
        audio = self._validate_audio(audio)
        if isinstance(self, ContinuousPitchAlgorithm):
            raw_fn = self._extract_raw_pitch_and_periodicity
        else:
            raw_fn = functools.partial(self._extract_pitch_with_threshold,
                                       threshold=self._get_default_threshold())
        times, pitch, periodicity = self._extract_windowed(audio, raw_fn)
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(pitch, dtype=np.float64),
            np.asarray(periodicity, dtype=np.float64),
        )

    def _validate_audio(self, audio):
        if audio.size == 0:
            raise ValueError("Empty audio input")
        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1-D (mono), got shape {audio.shape}")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        if np.any(np.abs(audio) > 1.0):
            raise ValueError("Audio must be normalized to [-1.0, 1.0]")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _compute_target_times(self, audio_length):
        n_hops = audio_length // self.hop_size
        return np.arange(n_hops) * (self.hop_size / self.sample_rate)

    def _sanity_check(self, pitch, periodicity):
        pitch = np.asarray(pitch, dtype=np.float64).copy()
        periodicity = np.asarray(periodicity, dtype=np.float64).copy()
        unusable = ~(np.isfinite(pitch) & np.isfinite(periodicity))
        pitch[unusable] = 0.0
        periodicity[unusable] = 0.0

        voiced = (periodicity > 0) & (pitch > 0)
        pitch[~voiced] = 0.0
        periodicity[~voiced] = 0.0
        pitch[voiced] = np.clip(pitch[voiced], self.fmin, self.fmax)

        periodicity = np.clip(periodicity, 0.0, 1.0)
        return pitch, periodicity

    def extract_pitch(self, audio, thresholds=None):
        if thresholds is None:
            thresholds = [self._get_default_threshold()]
        return self._extract_all(audio, list(thresholds))

    @abstractmethod
    def _extract_all(self, audio, thresholds):
        pass

    def _get_default_threshold(self):
        return 0.5

    @classmethod
    def get_name(cls):
        return cls.__name__.replace("PitchAlgorithm", "")


class ContinuousPitchAlgorithm(PitchAlgorithm):
    # A tracker whose periodicity is a binary voicing flag sets this to "nearest", so the
    # flag stays 0 or 1 on the frame grid instead of being interpolated between stamps.
    voicing_kind = "linear"

    @abstractmethod
    def _extract_raw_pitch_and_periodicity(self, audio):
        pass

    def extract_continuous_periodicity(self, audio):
        audio = self._validate_audio(audio)
        times, pitch, periodicity = self._extract_windowed(
            audio, self._extract_raw_pitch_and_periodicity
        )
        pitch, periodicity = self._sanity_check(pitch, periodicity)
        target_times = self._compute_target_times(len(audio))
        aligned_pitch, aligned_periodicity = resample_to_grid(
            pitch, periodicity, times, target_times, voicing_kind=self.voicing_kind
        )
        return aligned_pitch, aligned_periodicity

    def _extract_all(self, audio, thresholds):
        pitch, confidence = self.extract_continuous_periodicity(audio)
        results = []
        for threshold in thresholds:
            voicing = (confidence >= threshold).astype(bool)
            pitch_t = np.where(voicing, pitch, 0.0)
            results.append((pitch_t, voicing))
        return results


class ThresholdPitchAlgorithm(PitchAlgorithm):
    @abstractmethod
    def _extract_pitch_with_threshold(self, audio, threshold):
        pass

    def _extract_all(self, audio, thresholds):
        audio = self._validate_audio(audio)
        results = []
        target_times = self._compute_target_times(len(audio))
        for threshold in thresholds:
            times, pitch, periodicity = self._extract_windowed(
                audio, lambda a, t=threshold: self._extract_pitch_with_threshold(a, t)
            )
            pitch, periodicity = self._sanity_check(pitch, periodicity)
            aligned_pitch, aligned_periodicity = resample_to_grid(
                pitch, periodicity, times, target_times, voicing_kind="nearest"
            )
            voicing = is_voiced(aligned_periodicity)
            aligned_pitch = np.where(voicing, aligned_pitch, 0.0)
            results.append((aligned_pitch, voicing))
        return results
