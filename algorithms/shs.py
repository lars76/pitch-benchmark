import parselmouth

from .base import ContinuousPitchAlgorithm


class SHSPitchAlgorithm(ContinuousPitchAlgorithm):

    def _extract_raw_pitch_and_periodicity(self, audio):
        pitch_obj = parselmouth.Sound(audio, self.sample_rate).to_pitch_shs(
            time_step=self.hop_size / self.sample_rate,
            minimum_pitch=self.fmin,
            maximum_frequency_component=max(self.fmax, self.sample_rate / 2),
            ceiling=self.fmax,
        )
        return (
            pitch_obj.xs(),
            pitch_obj.selected_array["frequency"],
            pitch_obj.selected_array["strength"],
        )
