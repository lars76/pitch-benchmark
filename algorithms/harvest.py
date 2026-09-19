
import numpy as np
import pyworld as pw

from .base import ContinuousPitchAlgorithm


class HarvestPitchAlgorithm(ContinuousPitchAlgorithm):
    voicing_kind = "nearest"

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio64 = audio.astype(np.float64)
        frame_period = self.hop_size / self.sample_rate * 1000.0
        f0, t = pw.harvest(
            audio64,
            self.sample_rate,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=frame_period,
        )
        return t, f0, (f0 >= self.fmin).astype(np.float32)
