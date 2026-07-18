
import numpy as np
from pysptk import sptk

from .base import ThresholdPitchAlgorithm


class RAPTPitchAlgorithm(ThresholdPitchAlgorithm):
    # RAPT computes frame m's f0 from a forward-looking NCCF starting at sample m*hop: the
    # timestamp calibration (tests/test_time_calibration.py) measures the reported content a
    # constant 6.1 ms AFTER m*hop, invariant across sample rates (16 k / 22.05 k), hops (128-512)
    # and fmin (50-100), consistent with the centroid of RAPT's ~7.5 ms correlation window plus
    # the f0-period lag. Stamped per the calibration policy in TIMING.md (measured constant;
    # chirp +6.1 ms and step probe +8.0 ms agree in sign and magnitude).
    NCCF_CONTENT_DELAY = 0.0061  # seconds after the frame index time m*hop/sr

    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        audio_scaled = np.clip(audio * 32767, -32768, 32767)
        # RAPT's voice_bias knob: map threshold [0,1] -> [-0.6, 0.7].
        norm_threshold = -0.6 + threshold * (0.7 - (-0.6))

        f0 = sptk.rapt(
            audio_scaled,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            voice_bias=norm_threshold,
            otype="f0",
        )

        # Frames are spaced hop_size apart from sample 0; stamp the measured content time
        # m*hop/sr + NCCF_CONTENT_DELAY (see class comment).
        n_frames = len(f0)
        times = np.arange(n_frames) * self.hop_size / self.sample_rate + self.NCCF_CONTENT_DELAY

        return times, f0, (f0 >= self.fmin).astype(np.float32)

    def _get_default_threshold(self) -> float:
        return 0.525
