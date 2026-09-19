
import numpy as np
from pysptk import sptk

from .base import PCM16_MAX, ThresholdPitchAlgorithm, threshold_to_param


class RAPTPitchAlgorithm(ThresholdPitchAlgorithm):
    NCCF_HALF_WINDOW_S = 0.00375

    def _extract_pitch_with_threshold(self, audio, threshold):
        audio_scaled = np.clip(audio * PCM16_MAX, -PCM16_MAX - 1, PCM16_MAX)
        norm_threshold = threshold_to_param(threshold, (-0.6, 0.7))

        f0 = sptk.rapt(
            audio_scaled,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            voice_bias=norm_threshold,
            otype="f0",
        )

        idx = np.arange(len(f0))
        voiced = f0 > 0
        half_period = (np.interp(idx, idx[voiced], 0.5 / f0[voiced]) if voiced.any()
                       else np.full(len(f0), 0.5 / np.sqrt(self.fmin * self.fmax)))
        times = np.maximum.accumulate(
            idx * (self.hop_size / self.sample_rate) + self.NCCF_HALF_WINDOW_S + half_period)

        return times, f0, (f0 >= self.fmin).astype(np.float32)

    def _get_default_threshold(self):
        return 0.325
