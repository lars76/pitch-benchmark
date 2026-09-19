
import numpy as np
from pysptk import sptk

from .base import ThresholdPitchAlgorithm, threshold_to_param


class SWIPEPitchAlgorithm(ThresholdPitchAlgorithm):
    CHUNK_SECONDS = None

    def _extract_pitch_with_threshold(self, audio, threshold):
        norm_threshold = threshold_to_param(threshold, (0.2, 0.5))

        f0 = sptk.swipe(
            audio,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            threshold=norm_threshold,
            otype="f0",
        )

        n_frames = len(f0)
        times = np.arange(n_frames) * self.hop_size / self.sample_rate

        return times, f0, (f0 >= self.fmin).astype(np.float32)

    def _get_default_threshold(self):
        return 0.45
