
import parselmouth

from .base import ContinuousPitchAlgorithm


class PraatPitchAlgorithm(ContinuousPitchAlgorithm):
    def _extract_raw_pitch_and_periodicity(self, audio):
        sound = parselmouth.Sound(audio, self.sample_rate)
        pitch_obj = sound.to_pitch(
            time_step=self.hop_size / self.sample_rate,
            pitch_floor=self.fmin,
            pitch_ceiling=self.fmax,
        )
        return (
            pitch_obj.xs(),
            pitch_obj.selected_array["frequency"],
            pitch_obj.selected_array["strength"],
        )

    def _get_default_threshold(self):
        return 0.6
