import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


class CHiMeNoiseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        chime_home_dir: Union[str, Path],
        additional_noise_dirs: Optional[List[Union[str, Path]]] = None,
        chime_sample_rate: int = 16000,
        background_snr_range: Tuple[float, float] = (20.0, 40.0),
        voice_gain_range: Tuple[float, float] = (-6.0, 6.0),
        chime_noise_ratio_range: Tuple[float, float] = (0.0, 1.0),
        power_threshold: float = 1e-8,
        fallback_snr_db: float = -35.0,
        seed: int = 0,
        **kwargs,
    ):
        self.seed = int(seed)
        self.base_dataset = base_dataset
        self.background_snr_range = background_snr_range
        self.voice_gain_range = voice_gain_range
        self.chime_noise_ratio_range = chime_noise_ratio_range
        self.power_threshold = power_threshold
        self.fallback_snr_db = fallback_snr_db
        self.sample_rate = base_dataset.sample_rate
        self.hop_size = base_dataset.hop_size
        self.fmin = base_dataset.fmin
        self.fmax = base_dataset.fmax

        # Setup CHiME files
        chime_home_dir = Path(chime_home_dir)
        chunks_dir = chime_home_dir / "chunks"

        if not chunks_dir.exists():
            raise ValueError(f"CHiME chunks directory not found: {chunks_dir}")

        # Find matching files
        suffix = f".{chime_sample_rate // 1000}kHz.wav"
        chime_files = sorted(chunks_dir.rglob(f"*{suffix}"))   # sorted -> deterministic cache order

        if not chime_files:
            raise ValueError(f"No CHiME files found with suffix {suffix}")

        # Initialize resampler cache
        self.resamplers = {}
        self.noise_cache = []

        # Load CHiME noises
        self._load_noise_files(chime_files)

        # Load additional noises
        if additional_noise_dirs:
            for noise_dir in additional_noise_dirs:
                noise_dir = Path(noise_dir)
                if not noise_dir.exists():
                    raise ValueError(f"Noise directory not found: {noise_dir}")
                noise_files = sorted(noise_dir.rglob("*.wav"))
                if noise_files:
                    self._load_noise_files(noise_files)

        if not self.noise_cache:
            raise RuntimeError("No noise files available after loading")

        # Initialize noise selection
        self._noise_perm = list(range(len(self.noise_cache)))
        random.Random(self.seed).shuffle(self._noise_perm)

    def __len__(self):
        return len(self.base_dataset)

    def _load_noise_files(self, file_paths: List[Path]):
        """Load and cache noise files with proper resampling"""
        for path in file_paths:
            try:
                noise, orig_sr = torchaudio.load(str(path))
                if noise.shape[0] > 1:
                    noise = noise.mean(dim=0, keepdim=True)

                if orig_sr != self.sample_rate:
                    if orig_sr not in self.resamplers:
                        self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                            orig_freq=orig_sr, new_freq=self.sample_rate
                        )
                    noise = self.resamplers[orig_sr](noise)

                noise = noise.squeeze()
                power = noise.pow(2).mean()
                if power > self.power_threshold:
                    noise = noise / power.sqrt()
                    self.noise_cache.append(noise)
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")

    def _get_mixed_noise(
        self, target_length: int, idx: int, rng: random.Random, gen: torch.Generator
    ) -> torch.Tensor:
        """Get mixed CHiME + Gaussian noise (per-sample-deterministic via rng/gen)."""
        if target_length <= 0:
            return torch.empty(0)

        noise_idx = self._noise_perm[idx % len(self._noise_perm)]
        chime_noise = self._get_noise_segment(
            self.noise_cache[noise_idx], target_length, rng
        )

        gaussian_noise = torch.randn(target_length, generator=gen)
        if target_length > 1:
            power = gaussian_noise.pow(2).mean()
            if power > self.power_threshold:
                gaussian_noise = gaussian_noise / power.sqrt()

        chime_ratio = rng.uniform(*self.chime_noise_ratio_range)
        mixed_noise = (
            torch.sqrt(torch.tensor(chime_ratio)) * chime_noise
            + torch.sqrt(torch.tensor(1 - chime_ratio)) * gaussian_noise
        )
        return mixed_noise

    def _scale_noise_to_snr(
        self, noise: torch.Tensor, reference_signal: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Scale noise to achieve target SNR relative to reference signal power"""
        # Handle empty signals
        if reference_signal.numel() == 0 or noise.numel() == 0:
            return noise

        signal_power = reference_signal.pow(2).mean()
        if signal_power < self.power_threshold:
            # Use fallback if reference is too quiet
            signal_power = 10 ** (self.fallback_snr_db / 10)

        noise_power = noise.pow(2).mean()
        if noise_power < self.power_threshold:
            return noise  # Can't scale zero noise

        # Calculate required scaling factor
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-9))
        return noise * scale

    def __getitem__(self, idx) -> Dict:
        sample = self.base_dataset[idx]
        # Clone tensors to avoid modifying cached originals
        audio = sample["audio"].clone()
        periodicity = sample.get("periodicity")
        pitch = sample.get("pitch")

        # Preserve original if exists, else None
        if periodicity is not None:
            periodicity = periodicity.clone()
        if pitch is not None:
            pitch = pitch.clone()

        # Ensure audio is 1D (squeeze channel dimension)
        if audio.dim() > 1:
            audio = audio.squeeze(0)

        # Per-sample RNGs from SeedSequence(seed, idx): noise is a pure function of (seed, idx),
        # so it is reproducible and worker-safe.
        py_seed, torch_seed = (
            int(x) for x in np.random.SeedSequence([self.seed, int(idx)]).generate_state(2, dtype=np.uint32)
        )
        rng = random.Random(py_seed)
        gen = torch.Generator().manual_seed(torch_seed)

        # Apply random voice gain
        voice_gain_db = rng.uniform(*self.voice_gain_range)
        voice_scale = 10 ** (voice_gain_db / 20)
        audio = audio * voice_scale

        chunk_length = audio.shape[0]

        # Skip augmentation for too small chunks
        if chunk_length < self.hop_size:
            sample["audio"] = audio
            return sample

        # Apply background noise
        background_noise = self._get_mixed_noise(chunk_length, int(idx), rng, gen)
        if background_noise.numel() > 0:
            background_snr = rng.uniform(*self.background_snr_range)
            audio = self._apply_voice_aware_snr(
                audio, background_noise, background_snr, periodicity
            )

        # Prevent clipping after all processing
        audio = torch.clamp(audio, -1.0, 1.0)

        # Update sample with modified tensors
        sample["audio"] = audio
        if periodicity is not None:
            sample["periodicity"] = periodicity
        if pitch is not None:
            sample["pitch"] = pitch
        return sample

    def _get_noise_segment(
        self, noise: torch.Tensor, target_length: int, rng: random.Random
    ) -> torch.Tensor:
        """Extract noise segment without redundant normalization"""
        if target_length <= 0:
            return torch.empty(0)

        noise_len = noise.size(0)
        if noise_len >= target_length:
            start = rng.randint(0, noise_len - target_length)
            return noise[start : start + target_length]
        else:
            # Repeat noise to cover target length
            repeats = (target_length + noise_len - 1) // noise_len
            return noise.repeat(repeats)[:target_length]

    def _apply_voice_aware_snr(
        self,
        signal: torch.Tensor,
        noise: torch.Tensor,
        snr_db: float,
        periodicity: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply SNR normalization based on voiced segments"""
        if signal.numel() == 0:
            return signal

        # Handle silent segments
        if periodicity is not None and periodicity.sum() == 0:
            snr_db = self.fallback_snr_db

        # Compute signal power based on voiced regions
        if periodicity is not None and periodicity.sum() > 0:
            signal_power = self._compute_voiced_power(signal, periodicity)
        else:
            signal_power = signal.pow(2).mean()

        noise_power = noise.pow(2).mean()
        if noise_power < self.power_threshold:
            return signal

        # Calculate scaling factor
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-9))
        return (signal + scale * noise).clamp(-1, 1)

    def _compute_voiced_power(
        self,
        signal: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute signal power on voiced frames only."""
        if signal.numel() == 0 or periodicity.numel() == 0 or periodicity.sum() == 0:
            return signal.pow(2).mean() if signal.numel() > 0 else torch.tensor(0.0)

        periodicity_mask = (periodicity > 0).squeeze()

        # Create a sample-level mask
        sample_mask = torch.zeros_like(signal, dtype=torch.bool)
        for i in torch.where(periodicity_mask)[0]:
            start = i * self.hop_size
            end = min(start + self.hop_size, len(sample_mask))
            if start < len(sample_mask):
                sample_mask[start:end] = True

        voiced_samples = signal[sample_mask]
        if voiced_samples.numel() > 0:
            return voiced_samples.pow(2).mean()
        else:
            return signal.pow(2).mean()  # Fallback
