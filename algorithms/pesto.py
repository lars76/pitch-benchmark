import pesto
import torch
from pesto.utils import reduce_activations

from resampling import frame_times

from .base import ContinuousPitchAlgorithm

MODEL_NAME = "mir-1k_g7"
REDUCTION = "alwa"
MIDI_NOTES = 128


class PESTOPitchAlgorithm(ContinuousPitchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = pesto.load_model(MODEL_NAME, step_size=1000.0 * self.hop_size / self.sample_rate,
                                      sampling_rate=self.sample_rate)
        bins = MIDI_NOTES * self.model.bins_per_semitone
        hz = 440.0 * 2.0 ** ((torch.arange(bins) / self.model.bins_per_semitone - 69.0) / 12.0)
        self.band = (hz >= self.fmin) & (hz <= self.fmax)

    def _extract_raw_pitch_and_periodicity(self, audio):
        with torch.inference_mode():
            _, confidence, _, activations = self.model(
                torch.from_numpy(audio), sr=self.sample_rate, convert_to_freq=False, return_activations=True)
            semitones = reduce_activations(activations * self.band, reduction=REDUCTION)
        pitch = 440.0 * 2.0 ** ((semitones.numpy() - 69.0) / 12.0)
        times = frame_times(len(pitch), self.hop_size, self.sample_rate)
        return times, pitch, confidence.numpy()
