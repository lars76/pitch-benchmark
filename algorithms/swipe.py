
import numpy as np
from pysptk import sptk

from .base import ThresholdPitchAlgorithm


class SWIPEPitchAlgorithm(ThresholdPitchAlgorithm):
    # Run SWIPE UNCHUNKED (one pysptk.swipe call per file), overriding the base 20 s windowing.
    # pysptk's SPTK C swipe has a heap bug (large unchecked matrix allocations, S = candidates x
    # frames; see its source): feeding it MANY same-size ~20 s chunks fragments the heap until a
    # later large alloc corrupts it -> SIGSEGV after ~40 calls (reproducible with pure pysptk, no
    # benchmark code). One allocation per file avoids that pattern: 0 crashes across the whole
    # MDB-stem-synth corpus incl. its 514 s files, with peak RSS plateauing at ~3.5 GB (a
    # fragmentation high-water-mark, not an unbounded leak). Unchunked is also the MOST correct
    # result -- SWIPE has a weak global dependency (ERB loudness normalization), so any chunking
    # perturbs a few boundary frames; no overlap removes it. Other trackers keep the base windowing.
    CHUNK_SECONDS = None

    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # SWIPE's threshold knob: map threshold [0,1] -> [0.2, 0.5].
        norm_threshold = np.clip(0.2 + threshold * (0.5 - 0.2), 0.2, 0.5)

        f0 = sptk.swipe(
            audio,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            threshold=norm_threshold,
            otype="f0",
        )

        # pysptk.swipe estimates f0 at frames spaced hop_size apart, frame m at sample m*hop_size,
        # so the timestamp is m*hop_size / sample_rate.
        n_frames = len(f0)
        times = np.arange(n_frames) * self.hop_size / self.sample_rate

        return times, f0, (f0 >= self.fmin).astype(np.float32)

    def _get_default_threshold(self) -> float:
        return 0.4
