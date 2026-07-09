
import numpy as np
from swift_f0 import SwiftF0

from .base import ContinuousPitchAlgorithm


class SwiftF0PitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.detector = SwiftF0()

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # SwiftF0 reports each frame's true time: frame m at (m*256 + 127.5)/16000 s, where 127.5 =
        # (1024-1)/2 - 384(pad) is the analysis-window center in original-audio samples (see
        # swift_f0.core._calculate_timestamps). Pass the timestamps through unchanged and let the
        # eval-side resampler place them on the grid.
        result = self.detector.detect_from_array(audio, self.sample_rate)
        return np.asarray(result.timestamps, dtype=float), result.pitch_hz, result.confidence

    def _get_default_threshold(self) -> float:
        return 0.887
