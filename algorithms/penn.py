
import numpy as np
import penn
import torch

from .base import ContinuousPitchAlgorithm, resolve_device


class PENNPitchAlgorithm(ContinuousPitchAlgorithm):
    # Frame-batched internally (batch_size), so peak is bounded by the batch, not the file
    # length -> the base-class audio windowing is unnecessary here (would just re-run it). Opt out.
    CHUNK_SECONDS = None
    device_backend = "torch"

    # FCNF0++'s reported pitch corresponds to signal this many 8 kHz samples BEFORE the analysis
    # window center (~11 ms), constant across sample rates and sweep rates; see the timestamp
    # comment in _extract_raw_pitch_and_periodicity and the timestamp calibration.
    MODEL_LAG_SAMPLES = 88

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 256,
        center: str = "half-hop",
        **kwargs,
    ):
        """Initialize PENN pitch detector.
        Args:
            device: Computation device ("cpu", "cuda", or specific GPU index)
            batch_size: Frames per model forward pass. Bounds peak memory independently of clip
                length; PENN decodes over the full sequence, so results are identical for any
                batch_size. 256 keeps peak ~4 GB; the upstream default of 2048 needs ~15 GB.
            center: Frame centering strategy ('half-window', 'half-hop', or 'zero')
        """
        super().__init__(**kwargs)

        # Convert hop_size from samples to seconds as required by PENN
        self.hopsize_seconds = float(self.hop_size) / self.sample_rate

        # cuda -> mps -> cpu for 'auto'. penn's public gpu-index API only knows cpu/cuda, but its
        # model follows frames.device, so we drive the device explicitly in _extract below.
        self.device = resolve_device(device)

        self.batch_size = batch_size
        self.center = center

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Ensure audio is float32 and in correct shape
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert to torch tensor and add batch dimension if needed
        audio_tensor = torch.from_numpy(audio).unsqueeze(0) if len(audio.shape) == 1 else torch.from_numpy(audio)

        # Replicate penn.from_audio's loop so inference runs on self.device (cpu/cuda/mps): penn.infer
        # moves the model to frames.device, so frames.to(self.device) selects the backend. preprocess/
        # infer/postprocess are penn's own building blocks (stable under our penn==0.0.14 pin).
        # inference_mode prevents the autograd graph (and its activations) from being built (~halves peak).
        with torch.inference_mode():
            pitch_parts, periodicity_parts = [], []
            for frames in penn.preprocess(
                audio_tensor,
                self.sample_rate,
                self.hopsize_seconds,
                self.batch_size,
                self.center,
            ):
                logits = penn.infer(frames.to(self.device)).detach()
                _, p, per = penn.postprocess(logits, self.fmin, self.fmax)
                pitch_parts.append(p)
                periodicity_parts.append(per)
            pitch = torch.cat(pitch_parts, 1)
            periodicity = torch.cat(periodicity_parts, 1)

        # Calculate time points
        n_frames = pitch.shape[1]

        # Frame timestamp = the time whose f0 the model actually reports, in two parts (measured
        # by the timestamp calibration; the previous hardcoded 40/penn.SAMPLE_RATE was half of
        # penn's DEFAULT 80-sample hop and wrong for every other hop):
        #  1. Window-center geometry per `center` mode: penn.preprocess pads the 8 kHz audio with
        #     int((WINDOW_SIZE - hop8k)/2) ('half-hop'), WINDOW_SIZE//2 ('zero') or nothing
        #     ('half-window') and strides windows of WINDOW_SIZE from the padded start, so frame
        #     m's window center sits at m*hop + ((WINDOW_SIZE-1)/2 - padding)/penn.SAMPLE_RATE.
        #  2. A constant FCNF0++ content lag: the reported pitch corresponds to signal
        #     MODEL_LAG_SAMPLES before the window center, measured -10.99 ms at 16 kHz and
        #     -11.05 ms at 22.05 kHz (rate-invariant across sweep rates, same sign on the step
        #     probe, so it is corrected here -- a constant measured offset is corrected in
        #     the wrapper, so every tracker is compared on one shared grid).
        hop8k = self.hopsize_seconds * penn.SAMPLE_RATE
        if self.center == "half-hop":
            padding = int((penn.WINDOW_SIZE - hop8k) / 2)
        elif self.center == "zero":
            padding = penn.WINDOW_SIZE // 2
        else:  # "half-window": no padding
            padding = 0
        window_center = ((penn.WINDOW_SIZE - 1) / 2 - padding) / penn.SAMPLE_RATE
        time_offset = window_center - self.MODEL_LAG_SAMPLES / penn.SAMPLE_RATE

        times = (np.arange(n_frames) * self.hopsize_seconds) + time_offset

        # Convert to numpy and remove batch dimension
        return (
            times,
            pitch.squeeze().cpu().numpy(),
            periodicity.squeeze().cpu().numpy(),
        )

    def _get_default_threshold(self) -> float:
        return 0.388
