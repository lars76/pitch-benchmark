import numpy as np
from crepe.core import build_and_load_model, to_viterbi_cents
from numpy.lib.stride_tricks import as_strided

from resampling import frame_times, model_hop_length

from .base import ContinuousPitchAlgorithm, TensorFlowModelMixin, resample_audio, salience_band_mask

MODEL_SAMPLE_RATE = 16000
MODEL_WINDOW = 1024
FRAME_STD_FLOOR = 1e-8
N_CLASS = 360
CENTS_MAPPING = np.linspace(0, 7180, N_CLASS) + 1997.3794084376191


class CREPEPitchAlgorithm(TensorFlowModelMixin, ContinuousPitchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_tensorflow()
        self.model = build_and_load_model('full')

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio = resample_audio(audio.astype(np.float32), self.sample_rate, MODEL_SAMPLE_RATE)

        audio = np.pad(audio, MODEL_WINDOW // 2, mode="constant", constant_values=0)

        hop_length = model_hop_length(self.hop_size, self.sample_rate, MODEL_SAMPLE_RATE)
        n_frames = 1 + int((len(audio) - MODEL_WINDOW) / hop_length)
        frames = as_strided(
            audio,
            shape=(MODEL_WINDOW, n_frames),
            strides=(audio.itemsize, hop_length * audio.itemsize),
        )
        frames = frames.transpose().copy()

        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], FRAME_STD_FLOOR, None)

        activation = self.model.predict(frames, verbose=0)
        activation = np.where(salience_band_mask(CENTS_MAPPING, self.fmin, self.fmax),
                              activation, 0.0)

        confidence = activation.max(axis=1)
        frequency = 10 * 2 ** (to_viterbi_cents(activation) / 1200)
        frequency[np.isnan(frequency)] = 0

        time = frame_times(confidence.shape[0], hop_length, MODEL_SAMPLE_RATE)

        return time, frequency, confidence

    def _get_default_threshold(self):
        return 0.6
