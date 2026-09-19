import torch
import torchcrepe

from resampling import frame_times, model_hop_length

from .base import ContinuousPitchAlgorithm


class TorchCREPEPitchAlgorithm(ContinuousPitchAlgorithm):
    CHUNK_SECONDS = None

    MODEL = "full"
    BATCH_SIZE = 256

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            self.sample_rate,
            self.hop_size,
            self.fmin,
            self.fmax,
            model=self.MODEL,
            return_periodicity=True,
            decoder=torchcrepe.decode.viterbi,
            device="cpu",
            batch_size=self.BATCH_SIZE,
            pad=True,
        )
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()

        hop16k = model_hop_length(self.hop_size, self.sample_rate, torchcrepe.SAMPLE_RATE)
        times = frame_times(len(pitch), hop16k, torchcrepe.SAMPLE_RATE)

        return times, pitch, periodicity

    def _get_default_threshold(self):
        return 0.525
