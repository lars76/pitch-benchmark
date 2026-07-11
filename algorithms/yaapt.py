
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np

from .base import ThresholdPitchAlgorithm


class YAAPTPitchAlgorithm(ThresholdPitchAlgorithm):
    """YAAPT pitch detection algorithm implementation.

    Timestamps: pYAAPT's `frames_pos` marks the CENTER of each frame_length window, but the
    timestamp calibration (tests/test_time_calibration.py) shows the reported pitch content sits
    at a fixed delay after the frame START, independent of frame_length (sweeping frame_length
    35 -> 45 ms moves the apparent offset by exactly -frame_length/2: -8.65 -> -13.48 ms), i.e.
    YAAPT's NCCF/spectral evidence is anchored to the front of the analysis window, not its
    center. We therefore stamp frame_start + NCCF_CONTENT_DELAY (measured 8.9 ms at 16 kHz and
    10.3 ms at 22.05 kHz; the 9.6 ms mean leaves <=0.8 ms residual at both rates). Applied per
    the calibration policy in TIMING.md (chirp -8.65 / step -10.0 agree in sign and magnitude)."""

    NCCF_CONTENT_DELAY = 0.0096  # seconds after the analysis-frame start

    def __init__(
        self,
        frame_length: float = 35.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.frame_length_samples = int((frame_length / 1000.0) * self.sample_rate)
        self.frame_space_ms = (self.hop_size / self.sample_rate) * 1000.0

        # Configure YAAPT parameters
        self.yaapt_params = {
            "frame_length": frame_length,  # frame length in ms
            "frame_space": self.frame_space_ms,  # frame spacing in ms
            "f0_min": self.fmin,  # minimum pitch
            "f0_max": self.fmax,  # maximum pitch
            "nccf_thresh1": 0.25,  # lower NCCF threshold
            "nccf_thresh2": 0.9,  # upper NCCF threshold
            "nccf_maxcands": 4,  # maximum number of candidates
            "shc_maxpeaks": 4,  # maximum number of SHC peaks
            "shc_pwidth": 50,  # SHC window width
            "shc_thresh1": 5,  # SHC threshold 1
            "shc_thresh2": 1.25,  # SHC threshold 2
            "f0_double": 150,  # pitch doubling threshold
            "f0_half": 150,  # pitch halving threshold
            "merit_boost": 0.20,  # merit boost
            "merit_pivot": 0.99,  # merit pivot
            "merit_extra": 0.4,  # merit extra
            "median_value": 7,  # median filter order
            "dp_w1": 0.15,  # DP weight for voiced-voiced transitions
            "dp_w2": 0.5,  # DP weight for voiced-unvoiced transitions
            "dp_w3": 0.1,  # DP weight for unvoiced-unvoiced transitions
            "dp_w4": 0.9,  # DP weight for local costs
            "spec_pitch_min_std": 0.05,  # minimum spectral pitch std dev
        }

    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create signal object for pYAAPT
        signal = basic.SignalObj(audio, self.sample_rate)

        # Update NLFER threshold based on input threshold
        self.yaapt_params["nlfer_thresh1"] = threshold

        # Extract pitch using YAAPT
        pitch = pYAAPT.yaapt(signal, **self.yaapt_params)

        # Get pitch values and voicing decisions
        pitch_values = pitch.samp_values

        # frames_pos marks window centers; stamp start + measured content delay (class docstring).
        starts = np.asarray(pitch.frames_pos, dtype=float) - self.frame_length_samples // 2
        times = starts / self.sample_rate + self.NCCF_CONTENT_DELAY

        return (
            times,
            pitch_values,
            (pitch_values >= self.fmin).astype(np.float32),
        )

    def _get_default_threshold(self) -> float:
        return 0.675
