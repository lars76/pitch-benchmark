
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from resampling import frame_times, resample_audio

from .base import ContinuousPitchAlgorithm, TensorFlowModelMixin


class SPICEPitchAlgorithm(TensorFlowModelMixin, ContinuousPitchAlgorithm):
    """
    SPICE (Self-supervised Pitch Estimation) implementation using TensorFlow Hub.

    SPICE is a self-supervised pitch estimation model that provides robust
    pitch detection for monophonic audio signals.
    """

    _name = "SPICE"
    device_backend = "tf"

    def __init__(
        self,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_srate = 16000  # SPICE expects 16kHz
        # SPICE's fixed frame stride at 16 kHz (32 ms): frame m at sample m*512, and the model
        # returns len(audio)//512 + 1 frames (verified empirically across input lengths).
        self.MODEL_HOP = 512
        self.tf_device = self._init_tensorflow(device)

        # Load the SPICE model from TensorFlow Hub
        with tf.device(self.tf_device):
            self.model = hub.load("https://tfhub.dev/google/spice/2")

        # Pitch-conversion constants from https://tfhub.dev/google/spice/2
        self.PT_OFFSET = 25.58
        self.PT_SLOPE = 63.07
        self.FMIN = 10.0
        self.BINS_PER_OCTAVE = 12.0

    def _spice_output_to_hz(self, pitch_output: np.ndarray) -> np.ndarray:
        """
        Convert SPICE pitch output (0-1 range) to frequency in Hz.

        Args:
            pitch_output: SPICE pitch predictions in range [0, 1]

        Returns:
            Frequency values in Hz
        """
        # Convert using SPICE's specific constants
        cqt_bin = pitch_output * self.PT_SLOPE + self.PT_OFFSET
        frequency = self.FMIN * (2.0 ** (cqt_bin / self.BINS_PER_OCTAVE))

        frequency = np.nan_to_num(frequency, nan=0.0, posinf=0.0, neginf=0.0)

        return frequency

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for SPICE model.

        Args:
            audio: Input audio array

        Returns:
            Preprocessed audio ready for SPICE
        """
        # Convert to mono if stereo
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        # Convert to float32
        audio = audio.astype(np.float32)

        # Resample to the model's 16 kHz rate if necessary
        audio = resample_audio(audio, self.sample_rate, self.model_srate)

        # Ensure audio is normalized to [-1, 1] range
        audio_max = np.max(np.abs(audio))
        if audio_max > 1.0:
            audio = audio / audio_max

        return audio

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch and periodicity from audio using SPICE model.

        Args:
            audio: Input audio array

        Returns:
            Tuple of (times, frequencies, confidences)
        """
        processed_audio = self._preprocess_audio(audio)

        with tf.device(self.tf_device):
            # SPICE expects a constant tensor
            audio_tensor = tf.constant(processed_audio, dtype=tf.float32)

            model_output = self.model.signatures["serving_default"](audio_tensor)

            pitch_outputs = model_output["pitch"].numpy()
            uncertainty_outputs = model_output["uncertainty"].numpy()

        # Convert uncertainty to confidence
        confidence_outputs = 1.0 - uncertainty_outputs

        # Convert SPICE pitch outputs to Hz
        frequency_outputs = self._spice_output_to_hz(pitch_outputs)

        # SPICE returns no timestamps but has a fixed 512-sample hop at 16 kHz: frame m sits at
        # m*512 (verified: n_frames == len(audio)//512 + 1 for any input length). Stamp from that
        # true stride; the previous duration/(n_frames-1) heuristic coincides with it only when the
        # clip length is an exact hop multiple and stretches the timeline otherwise.
        time_outputs = frame_times(len(pitch_outputs), self.MODEL_HOP, self.model_srate)

        return time_outputs, frequency_outputs, confidence_outputs

    def _get_default_threshold(self) -> float:
        return 0.825
