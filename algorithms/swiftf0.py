from swift_f0 import SAMPLE_RATE, SwiftF0

from .base import ContinuousPitchAlgorithm, resample_audio


class SwiftF0PitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = SwiftF0()

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio = resample_audio(audio, self.sample_rate, SAMPLE_RATE)
        result = self.detector.detect(audio, SAMPLE_RATE, self.fmin, self.fmax)
        return result.timestamps, result.pitch_hz, result.confidence
