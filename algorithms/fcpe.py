import contextlib
import io

import numpy as np
import torch
from torchfcpe import spawn_bundled_infer_model

from resampling import frame_times

from .base import ContinuousPitchAlgorithm, resample_audio

DECODER = "local_argmax"


class FCPEPitchAlgorithm(ContinuousPitchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = spawn_bundled_infer_model(device="cpu")
        self.model_sample_rate = self.model.get_model_sr()
        self.model_hop_size = self.model.get_hop_size()
        hz = 10.0 * 2.0 ** (self.model.model.cent_table / 1200.0)
        self.band = (hz >= self.fmin) & (hz <= self.fmax)

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio = resample_audio(audio, self.sample_rate, self.model_sample_rate)
        with torch.inference_mode():
            mel = self.model.wav2mel(torch.from_numpy(np.ascontiguousarray(audio))[None, :, None],
                                     self.model_sample_rate)
            latent = self.model.model(mel) * self.band
            cents = self.model.model.latent2cents_local_decoder(latent, mask=False)
            pitch = self.model.model.cent_to_f0(cents)[0, :, 0].numpy()
            confidence = latent.amax(dim=-1)[0].numpy()
        return frame_times(len(pitch), self.model_hop_size, self.model_sample_rate), pitch, confidence
