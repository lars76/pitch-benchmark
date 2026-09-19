import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np

from .base import ThresholdPitchAlgorithm

FRAME_LENGTH_MS = 35.0
FIXED_PARAMS = {
    "nccf_thresh1": 0.25,
    "nccf_thresh2": 0.9,
    "nccf_maxcands": 4,
    "shc_maxpeaks": 4,
    "shc_pwidth": 50,
    "shc_thresh1": 5,
    "shc_thresh2": 1.25,
    "f0_double": 150,
    "f0_half": 150,
    "merit_boost": 0.20,
    "merit_pivot": 0.99,
    "merit_extra": 0.4,
    "median_value": 7,
    "dp_w1": 0.15,
    "dp_w2": 0.5,
    "dp_w3": 0.1,
    "dp_w4": 0.9,
    "spec_pitch_min_std": 0.05,
}


class YAAPTPitchAlgorithm(ThresholdPitchAlgorithm):

    NCCF_CONTENT_DELAY_S = 0.0096

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_length_samples = int((FRAME_LENGTH_MS / 1000.0) * self.sample_rate)
        self.yaapt_params = {
            **FIXED_PARAMS,
            "frame_length": FRAME_LENGTH_MS,
            "frame_space": (self.hop_size / self.sample_rate) * 1000.0,
            "f0_min": self.fmin,
            "f0_max": self.fmax,
        }

    def _extract_pitch_with_threshold(self, audio, threshold):
        signal = basic.SignalObj(audio, self.sample_rate)
        pitch = pYAAPT.yaapt(signal, nlfer_thresh1=threshold, **self.yaapt_params)
        pitch_values = pitch.samp_values

        starts = np.asarray(pitch.frames_pos, dtype=float) - self.frame_length_samples // 2
        times = starts / self.sample_rate + self.NCCF_CONTENT_DELAY_S

        return (
            times,
            pitch_values,
            (pitch_values >= self.fmin).astype(np.float32),
        )

    def _get_default_threshold(self):
        return 0.825
