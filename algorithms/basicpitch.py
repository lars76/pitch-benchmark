
import librosa
import numpy as np
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import Model, predict
from basic_pitch.note_creation import model_frames_to_time

from .base import ContinuousPitchAlgorithm


class BasicPitchPitchAlgorithm(ContinuousPitchAlgorithm):
    # Mean measured delay of the reported pitch content relative to the library's
    # model_frames_to_time stamps; see the comment at the `times` computation below.
    CONTENT_DELAY = 0.01075  # seconds

    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: float = 0.058,
        melodia_trick: bool = True,
        **kwargs,
    ):
        """Initialize Basic Pitch algorithm.

        Basic Pitch is a polyphonic automatic music transcription model that can detect
        multiple simultaneous pitches. This wrapper extracts the most prominent pitch
        per frame to fit the monophonic PitchAlgorithm interface.

        Args:
            onset_threshold: Threshold for onset detection (0.0-1.0, default: 0.5)
            frame_threshold: Threshold for frame-level note detection (0.0-1.0, default: 0.3)
            minimum_note_length: Minimum note length in seconds (default: 0.058)
            melodia_trick: Use melodia trick for better monophonic performance (default: True)
        """
        super().__init__(**kwargs)

        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length
        self.melodia_trick = melodia_trick

        # Load the Basic Pitch model once during initialization
        onnx_model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
        self.model = Model(onnx_model_path)

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import os
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        import soundfile as sf

        # Basic Pitch takes a file path, so write the audio to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            sf.write(temp_path, audio, self.sample_rate)

            # Suppress Basic Pitch's stdout chatter during predict.
            with redirect_stdout(StringIO()):
                model_output, _, _ = predict(
                    temp_path,
                    self.model,
                    onset_threshold=self.onset_threshold,
                    frame_threshold=self.frame_threshold,
                    minimum_note_length=self.minimum_note_length,
                    minimum_frequency=self.fmin,
                    maximum_frequency=self.fmax,
                    melodia_trick=self.melodia_trick,
                )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # Note activation matrix: shape (time, pitch).
        note_activations = model_output["note"]
        n_frames = note_activations.shape[0]

        # Basic Pitch uses 88 piano keys (A0 to C8), starting from MIDI note 21
        num_pitches = note_activations.shape[1]
        midi_start = 21  # A0
        midi_numbers = np.arange(midi_start, midi_start + num_pitches)
        frequencies = librosa.midi_to_hz(midi_numbers)

        # Zero out pitches outside [fmin, fmax].
        valid_freq_mask = (frequencies >= self.fmin) & (frequencies <= self.fmax)
        masked_activations = note_activations * valid_freq_mask[np.newaxis, :]

        # Most prominent pitch per frame.
        max_indices = np.argmax(masked_activations, axis=1)
        max_confidences = masked_activations[np.arange(len(max_indices)), max_indices]
        pitch_estimates = frequencies[max_indices]

        # Basic Pitch's posteriorgram runs at AUDIO_SAMPLE_RATE/FFT_HOP (~86 fps); the library's
        # model_frames_to_time maps frame m to ~m*FFT_HOP/22050 with a per-inference-window seam
        # correction. The timestamp calibration measures the
        # reported content a further CONTENT_DELAY after those times: +10.8 ms at 16 kHz and
        # +10.7 ms at 22.05 kHz (rate-invariant; step probe agrees in sign). The error is not a
        # single geometric constant: it ramps ~+5..+15 ms across each 142-frame inference window
        # (basic_pitch windows clips internally at ~1.64 s), so this corrects the MEAN stamp error
        # and the remaining within-window jitter stays part of BasicPitch's score. Applied per the
        # calibration rule: a constant measured offset is corrected in the wrapper, so
        # every tracker is compared on one shared grid.
        times = model_frames_to_time(n_frames) + self.CONTENT_DELAY

        return times, pitch_estimates, max_confidences

    def _get_default_threshold(self) -> float:
        return 0.45
