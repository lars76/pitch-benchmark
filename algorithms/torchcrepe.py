from collections.abc import Callable

import numpy as np
import torch
import torchcrepe

from resampling import frame_times, model_hop_length

from .base import ContinuousPitchAlgorithm, resolve_device


class TorchCREPEPitchAlgorithm(ContinuousPitchAlgorithm):
    # Frame-batched internally (batch_size), so peak is bounded by the batch, not the file
    # length -> the base-class audio windowing would only add extra decode seams. Opt out.
    CHUNK_SECONDS = None
    device_backend = "torch"

    def __init__(
        self,
        decoder: Callable = torchcrepe.decode.viterbi,
        model: str = "full",
        device: str = "auto",
        batch_size: int = 256,
        **kwargs,
    ):
        """Initialize TorchCREPE pitch detector.

        Args:
            decoder: Strategy for converting network output to pitch ('weighted_argmax', 'argmax' or 'viterbi')
            model: Model capacity ('tiny', or 'full')
            device: Computation device ("cpu" or "cuda")
            batch_size: Frames per model forward pass; bounds peak memory independently of clip
                length. NOTE: torchcrepe runs the Viterbi decode *per batch*, so batch_size shifts
                results slightly at batch boundaries (a few Hz on a handful of frames). 256 keeps
                peak ~9 GB for model='full' (upstream default 2048 needs ~19 GB); drop batch_size
                or use model='tiny' for a tighter bound.
        """
        super().__init__(**kwargs)
        self.decoder = decoder
        self.model = model
        self.batch_size = batch_size

        self.device = resolve_device(device)

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio).to(self.device).unsqueeze(0)

        # Run prediction
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            self.sample_rate,
            self.hop_size,
            self.fmin,
            self.fmax,
            model=self.model,
            return_periodicity=True,
            decoder=self.decoder,
            device=self.device,
            batch_size=self.batch_size,
            pad=True,  # Ensure consistent frame count
        )

        # Convert to numpy arrays and remove batch dimension
        pitch = pitch.squeeze().cpu().numpy()
        periodicity = periodicity.squeeze().cpu().numpy()

        # torchcrepe resamples to its 16 kHz and strides with the TRUNCATED hop
        # int(hop_size*16000/sample_rate) (torchcrepe.core.preprocess); pad=True centers frame t at
        # t*hop16k on that timeline. Stamp from that actual stride, identical to the nominal hop
        # at 16 kHz, but stamping the nominal hop at e.g. 22.05 kHz drifts ~0.4% per frame.
        hop16k = model_hop_length(self.hop_size, self.sample_rate, torchcrepe.SAMPLE_RATE)
        times = frame_times(len(pitch), hop16k, torchcrepe.SAMPLE_RATE)

        return times, pitch, periodicity

    def _get_default_threshold(self) -> float:
        return 0.613
