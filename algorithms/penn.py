import numpy as np
import penn
import torch

from .base import ContinuousPitchAlgorithm


class PENNPitchAlgorithm(ContinuousPitchAlgorithm):
    CHUNK_SECONDS = None

    MODEL_LAG_SAMPLES = 88

    BATCH_SIZE = 256
    CENTER = "half-hop"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hopsize_seconds = float(self.hop_size) / self.sample_rate

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

        with torch.inference_mode():
            pitch_parts, periodicity_parts = [], []
            for frames in penn.preprocess(
                audio_tensor,
                self.sample_rate,
                self.hopsize_seconds,
                self.BATCH_SIZE,
                self.CENTER,
            ):
                logits = penn.infer(frames).detach()
                _, p, per = penn.postprocess(logits, self.fmin, self.fmax)
                pitch_parts.append(p)
                periodicity_parts.append(per)
            pitch = torch.cat(pitch_parts, 1)
            periodicity = torch.cat(periodicity_parts, 1)

        hop8k = self.hopsize_seconds * penn.SAMPLE_RATE
        padding = int((penn.WINDOW_SIZE - hop8k) / 2)
        window_center = ((penn.WINDOW_SIZE - 1) / 2 - padding) / penn.SAMPLE_RATE
        time_offset = window_center - self.MODEL_LAG_SAMPLES / penn.SAMPLE_RATE
        times = (np.arange(pitch.shape[1]) * self.hopsize_seconds) + time_offset

        return (
            times,
            pitch.squeeze(0).cpu().numpy(),
            periodicity.squeeze(0).cpu().numpy(),
        )

    def _get_default_threshold(self):
        return 0.15
