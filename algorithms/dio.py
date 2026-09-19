
import numpy as np
import pyworld as pw

from .base import ThresholdPitchAlgorithm, threshold_to_param


class DIOPitchAlgorithm(ThresholdPitchAlgorithm):
    def _extract_pitch_with_threshold(self, audio, threshold):
        audio64 = audio.astype(np.float64)
        frame_period = self.hop_size / self.sample_rate * 1000.0
        norm_threshold = threshold_to_param(threshold, (0.02, 0.2))

        f0, t = pw.dio(
            audio64,
            self.sample_rate,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=frame_period,
            allowed_range=norm_threshold,
        )
        f0 = pw.stonemask(audio64, f0, t, self.sample_rate)
        return t, f0, (f0 >= self.fmin).astype(np.float32)

    def _get_default_threshold(self):
        return 0.475
