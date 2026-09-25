import os
import tempfile

import numpy as np
import soundfile as sf
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import Model, run_inference
from basic_pitch.note_creation import model_frames_to_time

from .base import ContinuousPitchAlgorithm

MIDI_START = 21


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


class BasicPitchPitchAlgorithm(ContinuousPitchAlgorithm):

    CONTENT_DELAY_S = 0.01075

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
        self.model = Model(model_path)
        if os.environ.get("OMP_NUM_THREADS") == "1":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.model.model = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    def _extract_raw_pitch_and_periodicity(self, audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            sf.write(temp_path, audio, self.sample_rate, subtype="FLOAT")
            note_activations = run_inference(temp_path, self.model)["note"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        frequencies = midi_to_hz(np.arange(MIDI_START, MIDI_START + note_activations.shape[1]))
        masked_activations = note_activations * ((frequencies >= self.fmin) & (frequencies <= self.fmax))
        max_indices = np.argmax(masked_activations, axis=1)
        max_confidences = masked_activations[np.arange(len(max_indices)), max_indices]
        pitch_estimates = frequencies[max_indices]

        times = model_frames_to_time(note_activations.shape[0]) + self.CONTENT_DELAY_S

        return times, pitch_estimates, max_confidences

    def _get_default_threshold(self):
        return 0.25
