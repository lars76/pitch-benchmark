from typing import Tuple

import numpy as np
import pyworld as pw

from .base import ContinuousPitchAlgorithm


class HarvestPitchAlgorithm(ContinuousPitchAlgorithm):
    # Harvest has no configurable threshold parameter. All smoothing constants
    # are hardcoded in WORLD's FixF0Contour. We expose a binary confidence
    # (0.0 unvoiced / 1.0 voiced) so the benchmark runs the algorithm once
    # and applies threshold comparisons cheaply in post-processing.
    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        audio64 = audio.astype(np.float64)
        frame_period = self.hop_size / self.sample_rate * 1000.0  # ms
        f0, t = pw.harvest(
            audio64,
            self.sample_rate,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=frame_period,
        )
        return t, f0, (f0 >= self.fmin).astype(np.float32)
