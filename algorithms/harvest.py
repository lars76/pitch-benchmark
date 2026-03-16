from typing import Tuple

import numpy as np
import pyworld as pw

from .base import ThresholdPitchAlgorithm


class HarvestPitchAlgorithm(ThresholdPitchAlgorithm):
    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
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

    def _get_default_threshold(self) -> float:
        return 0.5
