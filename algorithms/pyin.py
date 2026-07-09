
import librosa
import numpy as np

from .base import ContinuousPitchAlgorithm


class pYINPitchAlgorithm(ContinuousPitchAlgorithm):
    def _extract_raw_pitch_and_periodicity(
        self, audio
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pitch, _, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_size,
            center=True,
        )
        times = librosa.times_like(pitch, sr=self.sample_rate, hop_length=self.hop_size)
        return times, pitch, voiced_probs

    def _get_default_threshold(self) -> float:
        return 0.1
