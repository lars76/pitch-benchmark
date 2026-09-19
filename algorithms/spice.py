
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from resampling import frame_times

from .base import ContinuousPitchAlgorithm, TensorFlowModelMixin, resample_audio

MODEL_SAMPLE_RATE = 16000
PT_OFFSET, PT_SLOPE = 25.58, 63.07
MODEL_FMIN_HZ = 10.0
BINS_PER_OCTAVE = 12.0


class SPICEPitchAlgorithm(TensorFlowModelMixin, ContinuousPitchAlgorithm):

    MODEL_HOP = 512

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_tensorflow()
        self.model = hub.load("https://tfhub.dev/google/spice/2")

    def _extract_raw_pitch_and_periodicity(self, audio):
        processed_audio = resample_audio(audio.astype(np.float32), self.sample_rate, MODEL_SAMPLE_RATE)

        model_output = self.model.signatures["serving_default"](
            tf.constant(processed_audio, dtype=tf.float32))
        pitch_outputs = model_output["pitch"].numpy()
        uncertainty_outputs = model_output["uncertainty"].numpy()

        confidence_outputs = 1.0 - uncertainty_outputs

        cqt_bin = pitch_outputs * PT_SLOPE + PT_OFFSET
        frequency_outputs = np.nan_to_num(MODEL_FMIN_HZ * (2.0 ** (cqt_bin / BINS_PER_OCTAVE)),
                                          nan=0.0, posinf=0.0, neginf=0.0)

        time_outputs = frame_times(len(pitch_outputs), self.MODEL_HOP, MODEL_SAMPLE_RATE)

        return time_outputs, frequency_outputs, confidence_outputs

    def _get_default_threshold(self):
        return 0.675
