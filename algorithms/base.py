import os
from abc import ABC, abstractmethod

import numpy as np

from metrics import is_voiced
from resampling import resample_to_grid

# Route the few ops MPS lacks to the CPU instead of erroring. Set here (before any torch import or
# inference) so it is in effect for every device-aware tracker; harmless on cpu/cuda machines.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested="auto"):
    """Map a device request to a concrete ``torch.device``, shared by every torch-based tracker.

    ``auto`` picks the fastest available: cuda -> mps -> cpu. An explicit ``cuda`` or ``mps`` that is
    not available falls back down that same ladder; ``cpu`` is honored as-is. torch is imported
    lazily so importing this module (and the torch-free trackers like DIO/Harvest) does not pull torch.
    """
    import torch

    if isinstance(requested, torch.device):
        return requested
    req = str(requested or "auto").lower()
    if req in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    # auto, or a requested accelerator that is unavailable -> best available
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_TF_CONFIGURED = False  # guards the one-time, process-wide TensorFlow setup below


class TensorFlowModelMixin:
    """Shared TensorFlow setup + teardown for the TF-backed trackers (CREPE, SPICE). tensorflow is
    imported lazily inside the methods so importing this module never pulls TF for the other
    trackers."""

    def _init_tensorflow(self, device: str) -> str:
        """Run the one-time process-wide TF config (quiet logging + GPU memory growth) and return the
        tf.device string ('/GPU:0' or '/CPU:0') for the requested benchmark device."""
        import tensorflow as tf

        global _TF_CONFIGURED
        if not _TF_CONFIGURED:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            tf.get_logger().setLevel("ERROR")
            try:
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
            _TF_CONFIGURED = True
        on_gpu = device == "cuda" and len(tf.config.list_physical_devices("GPU")) > 0
        return "/GPU:0" if on_gpu else "/CPU:0"

    def cleanup(self):
        """Release the loaded model + TF session (called at teardown and from __del__). Imports live
        inside the guard so an already-cleaned instance (e.g. at interpreter shutdown) touches nothing."""
        if getattr(self, "model", None) is not None:
            import gc

            import tensorflow as tf

            del self.model
            self.model = None
            tf.keras.backend.clear_session()
            gc.collect()

    def __del__(self):
        self.cleanup()


class PitchAlgorithm(ABC):
    """Base pitch tracker.

    Prediction contract: `extract_pitch(audio, thresholds, compute_notes)` returns a LIST of
    `(pitch, voicing, notes)`, one tuple per threshold:
      - pitch:   np.ndarray float, shape (F,), Hz; 0.0 on unvoiced frames; clamped to [fmin, fmax]
                 on voiced frames; NaN mapped to 0.
      - voicing: np.ndarray bool, shape (F,). Already thresholded: the binary voiced/unvoiced
                 decision for this entry's threshold. (The ground-truth `periodicity` is instead
                 a [0,1] confidence; the metric compares this bool against periodicity >= 0.5.)
      - notes:   Optional[List[{start, end, midi_pitch}]] (None if compute_notes=False).

    All arrays are length F = len(audio) // hop_size on the shared eval grid: frame m is the audio
    CENTERED at sample m*hop_size (time m*hop_size / sample_rate). Each backend's native frame times
    are resampled onto this grid (resampling.resample_to_grid), the SAME grid the ground truth uses
   , so predictions and labels line up frame-for-frame.

    The `thresholds` list in [0,1] selects the voicing operating point. Backends respond in one of
    three ways (see ContinuousPitchAlgorithm / ThresholdPitchAlgorithm):
      - confidence-based: threshold a real [0,1] confidence (run once); smooth precision/recall sweep;
      - parameter-based:  the threshold maps to an internal voicing parameter (re-run per threshold);
      - fixed operating point: no confidence and no parameter (Harvest); the sweep is DEGENERATE
        (all thresholds give the same voicing); the algorithm reports a single point.
    """

    # Audio longer than CHUNK_SECONDS is processed in overlapping windows so peak memory is
    # O(window) instead of O(file length); every backend here processes the whole signal at
    # once, so without this the activation/peak footprint grows with clip length and long files
    # OOM. CHUNK_OVERLAP_SECONDS of context is fed on each side and then trimmed, so the result
    # is seamless (must exceed any backend's analysis window; ~1 s >> the ~1024-sample frames).
    # Files shorter than CHUNK_SECONDS take the exact non-windowed path. Set CHUNK_SECONDS=None
    # to disable. Subclasses may override either constant.
    CHUNK_SECONDS = 20.0
    CHUNK_OVERLAP_SECONDS = 1.0

    # Device semantics, one source of truth (replaces `"device" in __init__.co_varnames` sniffing):
    #   None   -> cpu-only tracker (DSP / own-runtime); __init__ takes no `device` kwarg.
    #   "torch"-> honors auto/mps/cuda via resolve_device.
    #   "tf"   -> TensorFlow; uses the GPU only for a literal "cuda" request with a GPU present.
    # `device_backend is not None` == "accepts a device kwarg" (used by algorithms.build_algorithm).
    device_backend = None

    @classmethod
    def resolve_effective_device(cls, requested="cpu") -> str:
        """The device this tracker actually runs on for `requested`: the cache key + metadata stamp.

        Dispatches on `device_backend`, so device semantics live in one place and a new tracker only
        sets the attribute. torch trackers resolve against real availability; TF trackers use the GPU
        only for a literal "cuda" request when an NVIDIA GPU is present (torch.cuda availability is a
        cheap proxy that avoids importing TF here; a CPU-only TF build on a CUDA box is the residual edge).
        """
        if cls.device_backend == "torch":
            return resolve_device(requested).type
        if cls.device_backend == "tf":
            import torch
            return "cuda" if (str(requested) == "cuda" and torch.cuda.is_available()) else "cpu"
        return "cpu"

    def __init__(self, sample_rate: int, hop_size: int, fmin: float, fmax: float):
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

        # Window length is hop-aligned so chunk edges land on the global frame grid.
        if self.CHUNK_SECONDS is None:
            self.chunk_samples = None
        else:
            self.chunk_samples = max(
                hop_size, int(self.CHUNK_SECONDS * sample_rate) // hop_size * hop_size
            )
        # Hop-align the overlap margin so every chunk starts on the global frame grid. A non-multiple
        # (1 s = 16000 samples = 62.5 hops at hop 256) starts each post-first chunk half a hop off the
        # grid, shifting all its native frames ~8 ms off-grid and dropping the frame on the seam. Floor
        # to a whole number of hops, but keep >= hop so the context margin never vanishes.
        self.chunk_overlap_samples = max(
            hop_size, int(self.CHUNK_OVERLAP_SECONDS * sample_rate) // hop_size * hop_size
        )

    def _extract_windowed(self, audio: np.ndarray, raw_fn):
        """Run ``raw_fn(segment) -> (times, pitch, periodicity)`` over the whole signal.

        Short signals call ``raw_fn`` once (byte-identical to the non-windowed path). Long
        signals are split into hop-aligned windows, each computed with CHUNK_OVERLAP_SECONDS of
        context on either side; only the frames whose timestamps fall in the window's own
        [start, end) span are kept, so boundary/centering artifacts are trimmed and the
        concatenated timestamps stay strictly increasing (required by the downstream interp).
        ``times`` are in seconds from the segment start, so each window is offset by its start.
        """
        n = len(audio)
        if self.chunk_samples is None or n <= self.chunk_samples:
            return raw_fn(audio)

        sr, win, margin = self.sample_rate, self.chunk_samples, self.chunk_overlap_samples
        all_t, all_p, all_q = [], [], []
        start = 0
        while start < n:
            end = min(start + win, n)
            cs, ce = max(0, start - margin), min(n, end + margin)
            t, p, q = raw_fn(audio[cs:ce])
            t = np.asarray(t, dtype=float) + cs / sr
            # Assign each native frame to exactly one chunk by its sample position, with the boundary
            # at the half-sample midpoint: every on-grid tracker lands a frame exactly on start/end, so
            # a plain t >= start/sr comparison is at the mercy of float rounding (the frame can fall
            # into both windows or neither). The half-sample offset claims it once. Last window keeps
            # its whole tail.
            ts = t * sr
            lo = start - 0.5
            hi = np.inf if end >= n else end - 0.5
            keep = (ts >= lo) & (ts < hi)
            all_t.append(t[keep])
            all_p.append(np.asarray(p)[keep])
            all_q.append(np.asarray(q)[keep])
            start = end
        return np.concatenate(all_t), np.concatenate(all_p), np.concatenate(all_q)

    @property
    def supports_continuous_periodicity(self) -> bool:
        return isinstance(self, ContinuousPitchAlgorithm)

    def native_frames(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The tracker's raw (times, pitch, periodicity) through the SAME path production uses
        (_extract_windowed) but BEFORE grid resampling: the quantity whose timestamp truthfulness
        the eval grid depends on. Parameter-based trackers run at their default threshold.
        Diagnostic API shared by the timestamp calibration and the
        dataset label-offset sweep (scripts/check_dataset_alignment.py)."""
        audio = self._validate_audio(audio)
        if self.supports_continuous_periodicity:
            raw_fn = self._extract_raw_pitch_and_periodicity
        else:
            thr = self._get_default_threshold()
            raw_fn = lambda a: self._extract_pitch_with_threshold(a, thr)  # noqa: E731
        times, pitch, periodicity = self._extract_windowed(audio, raw_fn)
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(pitch, dtype=np.float64),
            np.asarray(periodicity, dtype=np.float64),
        )

    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """Validate and return the audio as the canonical contiguous float32 every tracker expects.
        The datasets already deliver float32 (PitchDataset contract); normalizing here makes the
        input type uniform for any caller so the wrappers never special-case dtype (pysptk casts to
        C double itself; the neural trackers wrap the float32 in a tensor)."""
        if audio.size == 0:
            raise ValueError("Empty audio input")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        if np.any(np.abs(audio) > 1.0):
            raise ValueError("Audio must be normalized to [-1.0, 1.0]")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _compute_target_times(self, audio_length: int) -> np.ndarray:
        n_hops = audio_length // self.hop_size
        return np.arange(n_hops) * (self.hop_size / self.sample_rate)

    def _sanity_check(
        self, pitch: np.ndarray, periodicity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        periodicity = np.nan_to_num(periodicity, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)

        # A frame with no pitch (<=0, incl. NaN mapped to 0 above) is unvoiced regardless of the
        # backend's confidence; otherwise np.clip(0, fmin, fmax) would fabricate a voiced fmin (e.g.
        # pyin reports f0=NaN with voiced_prob>0 on unvoiced frames). Zero its confidence too so nothing
        # downstream (resample_to_grid's f0>0 gap-mask, the 0.0 sweep point, consensus voiced=pitch>0)
        # can treat it as a real detection.
        voiced = (periodicity > 0) & (pitch > 0)
        pitch[~voiced] = 0.0
        periodicity[~voiced] = 0.0
        pitch[voiced] = np.clip(pitch[voiced], self.fmin, self.fmax)

        periodicity = np.clip(periodicity, 0.0, 1.0)
        return pitch, periodicity

    def notes_from_pitch_contour(
        self,
        pitch_contour: np.ndarray,
        voicing_contour: np.ndarray,
        audio: np.ndarray | None = None,
        lam_per_s: float = 500.0,
        min_note_duration: float = 0.05,
        onset_gate: float = 2.5,
        rise_gate: float = 0.6,
    ) -> list[dict[str, float]]:
        """Contour -> notes via an exact penalised piecewise-constant fit (L2 changepoint DP)
        plus, when `audio` is given, two mandatory-boundary gates for same-pitch re-articulations
        the pitch contour cannot see: spectral-flux peaks above mean + onset_gate*std, and frame-RMS
        rises >= 1/rise_gate within two frames (legato re-bows / re-tongues / glottal resets).

        Replaces the previous greedy running-median segmenter: the DP is globally optimal for its
        objective and keeps CONTINUOUS pitch per note (midi_pitch = median of the segment, not
        rounded), so downstream scoring against continuous-Hz references is not quantised.
        lam_per_s is the split penalty per second of frames (scales with frame rate);
        min_note_duration drops shorter notes. Boundary evidence is derived from `audio` on the
        tracker's own (sample_rate, hop_size) grid; with audio=None only the DP runs.
        """
        frame_period = self.hop_size / self.sample_rate
        lam = lam_per_s * frame_period
        n = len(pitch_contour)
        voiced = (
            (np.asarray(voicing_contour) > 0)
            & (pitch_contour > 0)
            & (pitch_contour >= self.fmin)
            & (pitch_contour <= self.fmax)
        )
        midi = np.where(voiced, 69 + 12 * np.log2(np.clip(pitch_contour, 1e-6, None) / 440.0), np.nan)

        onsets = np.zeros(n, dtype=bool)
        if audio is not None:
            w = 1024
            xp = np.pad(np.asarray(audio, dtype=np.float64).reshape(-1), (w // 2, w // 2))
            frames = np.lib.stride_tricks.sliding_window_view(xp, w)[:: self.hop_size][:n]
            if onset_gate is not None and len(frames) > 2:
                lm = np.log(np.abs(np.fft.rfft(frames * np.hanning(w)[None, :], axis=1)) + 1e-8)
                fl = np.concatenate([[0.0], np.maximum(0.0, np.diff(lm, axis=0)).sum(axis=1)])
                loc = np.r_[False, (fl[1:-1] > fl[:-2]) & (fl[1:-1] >= fl[2:]), False]
                onsets[: len(fl)] |= loc & (fl > fl.mean() + onset_gate * fl.std())
            if rise_gate is not None and len(frames) > 2:
                rms = np.sqrt((frames**2).mean(axis=1))
                rise = np.zeros(len(rms), dtype=bool)
                rise[2:] = rms[:-2] <= rise_gate * rms[2:]
                rise[1:] &= ~rise[:-1]
                onsets[: len(rms)] |= rise

        def _pwc_change_points(x: np.ndarray) -> list[tuple[int, int]]:
            m = len(x)
            s1 = np.concatenate([[0.0], np.cumsum(x)])
            s2 = np.concatenate([[0.0], np.cumsum(x * x)])
            best = np.full(m + 1, np.inf)
            best[0] = 0.0
            back = np.zeros(m + 1, dtype=int)
            for j in range(1, m + 1):
                i = np.arange(0, j)
                sse = (s2[j] - s2[i]) - (s1[j] - s1[i]) ** 2 / (j - i)
                cand = best[i] + sse + lam
                k = int(np.argmin(cand))
                best[j], back[j] = cand[k], k
            segs, j = [], m
            while j > 0:
                segs.append((back[j], j))
                j = back[j]
            return segs[::-1]

        notes = []
        i = 0
        while i < n:
            if voiced[i]:
                j = i
                while j < n and voiced[j]:
                    j += 1
                bounds = [i] + [t for t in range(i + 2, j - 1) if onsets[t]] + [j]
                for a0, b0 in zip(bounds[:-1], bounds[1:]):
                    x = midi[a0:b0]
                    if len(x) == 0:
                        continue
                    for a, b in _pwc_change_points(x):
                        if (b - a) * frame_period >= min_note_duration:
                            notes.append(
                                {
                                    "start": (a0 + a) * frame_period,
                                    "end": (a0 + b - 1) * frame_period + frame_period,
                                    "midi_pitch": float(np.median(x[a:b])),
                                }
                            )
                i = j
            else:
                i += 1
        return notes

    def extract_pitch(
        self,
        audio: np.ndarray,
        thresholds: float | list[float] | None = None,
        compute_notes: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray, list[dict[str, float]] | None]]:
        """Always returns a list of (pitch, voicing, notes), one entry per threshold.

        `compute_notes=False` skips note segmentation (the benchmark discards notes, so the
        hot path passes False; other callers keep the default to get notes).
        """
        if thresholds is None:
            thresholds = [self._get_default_threshold()]
        elif isinstance(thresholds, (int, float)):
            thresholds = [float(thresholds)]
        else:
            thresholds = list(thresholds)

        if self.supports_continuous_periodicity:
            return self._extract_continuous_multiple_thresholds(
                audio, thresholds, compute_notes
            )
        return self._extract_threshold_multiple_thresholds(
            audio, thresholds, compute_notes
        )

    @abstractmethod
    def _get_default_threshold(self) -> float:
        pass

    @classmethod
    def get_name(cls) -> str:
        return getattr(cls, "_name", cls.__name__.replace("PitchAlgorithm", ""))


class ContinuousPitchAlgorithm(PitchAlgorithm):
    @abstractmethod
    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _get_default_threshold(self) -> float:
        return 0.5

    def extract_continuous_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        audio = self._validate_audio(audio)
        times, pitch, periodicity = self._extract_windowed(
            audio, self._extract_raw_pitch_and_periodicity
        )
        pitch, periodicity = self._sanity_check(pitch, periodicity)
        target_times = self._compute_target_times(len(audio))
        # Shared resampler (resampling.resample_to_grid): gap-aware cents interpolation for pitch (no
        # fabrication through the unvoiced 0 sentinel) and LINEAR resampling for the continuous
        # confidence, thresholded per operating point in _extract_continuous_multiple_thresholds.
        # Exact no-op for on-grid trackers (native times == grid).
        aligned_pitch, aligned_periodicity = resample_to_grid(
            pitch, periodicity, times, target_times, voicing_kind="linear"
        )
        return aligned_pitch, aligned_periodicity

    def _extract_continuous_multiple_thresholds(
        self, audio: np.ndarray, thresholds: list[float], compute_notes: bool = True
    ) -> list[tuple[np.ndarray, np.ndarray, list[dict[str, float]] | None]]:
        pitch, confidence = self.extract_continuous_periodicity(audio)
        results = []
        for threshold in thresholds:
            voicing = (confidence >= threshold).astype(bool)
            pitch_t = np.where(voicing, pitch, 0.0)   # pred contract: pitch == 0 on unvoiced frames
            notes = self.notes_from_pitch_contour(pitch_t, voicing, audio=audio) if compute_notes else None
            results.append((pitch_t, voicing, notes))
        return results


class ThresholdPitchAlgorithm(PitchAlgorithm):
    @abstractmethod
    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _get_default_threshold(self) -> float:
        return 0.5

    def _extract_threshold_multiple_thresholds(
        self, audio: np.ndarray, thresholds: list[float], compute_notes: bool = True
    ) -> list[tuple[np.ndarray, np.ndarray, list[dict[str, float]] | None]]:
        audio = self._validate_audio(audio)
        results = []
        target_times = self._compute_target_times(len(audio))
        for threshold in thresholds:
            times, pitch, periodicity = self._extract_windowed(
                audio, lambda a, t=threshold: self._extract_pitch_with_threshold(a, t)
            )
            pitch, periodicity = self._sanity_check(pitch, periodicity)
            # Shared resampler (resampling.resample_to_grid): gap-aware cents interpolation for pitch
            # (no fabrication through the unvoiced 0 sentinel) and NEAREST for the binary periodicity
            # (linear-then-threshold would dilate voiced regions by ~1 frame at every boundary for
            # off-grid trackers like YAAPT). Voicing via metrics.is_voiced (>= 0.5, not > 0).
            aligned_pitch, aligned_periodicity = resample_to_grid(
                pitch, periodicity, times, target_times, voicing_kind="nearest"
            )
            voicing = is_voiced(aligned_periodicity)
            aligned_pitch = np.where(voicing, aligned_pitch, 0.0)   # pred contract: pitch == 0 on unvoiced
            notes = self.notes_from_pitch_contour(aligned_pitch, voicing, audio=audio) if compute_notes else None
            results.append((aligned_pitch, voicing, notes))
        return results
