from typing import Tuple

import numpy as np
import pyworld as pw

from .base import ThresholdPitchAlgorithm


class DIOPitchAlgorithm(ThresholdPitchAlgorithm):
    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        audio64 = audio.astype(np.float64)
        frame_period = self.hop_size / self.sample_rate * 1000.0  # ms
        # DIO expects a special range.
        # Map threshold from [0,1] to [0.02,0.2]
        norm_threshold = 0.02 + threshold * (0.2 - 0.02)

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

    def _get_default_threshold(self) -> float:
        return 0.444
