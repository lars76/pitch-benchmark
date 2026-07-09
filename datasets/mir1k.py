from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from .base import PitchDataset


class PitchDatasetMIR1K(PitchDataset):
    """
    Implementation of PitchDataset for the MIR-1K dataset.

    MIR-1K contains 1000 song clips where music accompaniment and singing voice
    are recorded at left and right channels respectively. The dataset includes
    manual pitch annotations in semitone.
    The pitch labels have a 40 ms frame size and 20 ms hop.

    Note: Pitch annotations are only available for the singing voice channel.
    The accompaniment channel has no ground truth pitch labels.

    Args:
        root_dir (str): Root directory of the MIR-1K dataset
        use_cache (bool, optional): Whether to cache loaded data. Defaults to True
        **kwargs: Additional arguments passed to PitchDataset
    """

    fmin = 65
    fmax = 2093
    # MIR-1K manual pitch labels: 40 ms analysis window, 20 ms hop. Used to resample labels at their
    # true times; see _load_sample for the frame-centring convention and its empirical verification.
    PITCH_LABEL_HOP_SECONDS = 0.02

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(use_cache=use_cache, **kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        # Find all valid wav-pitch pairs
        self.wav_pitch_pairs = self._find_wav_pitch_pairs()
        if not self.wav_pitch_pairs:
            raise ValueError(f"No valid wav-pitch pairs found in '{root_dir}'")

    def _find_wav_pitch_pairs(self) -> list[tuple[Path, Path]]:
        """Finds matching WAV and pitch label file pairs in the dataset."""
        pairs = []

        wav_dir = self.root_dir / "Wavfile"
        pitch_dir = self.root_dir / "PitchLabel"

        if not wav_dir.exists():
            raise FileNotFoundError(f"Wavfile directory not found: {wav_dir}")
        if not pitch_dir.exists():
            raise FileNotFoundError(f"PitchLabel directory not found: {pitch_dir}")

        for wav_path in wav_dir.glob("*.wav"):
            # Look for corresponding .pv pitch file
            pitch_path = pitch_dir / f"{wav_path.stem}.pv"
            if pitch_path.exists():
                pairs.append((wav_path, pitch_path))

        return pairs

    def get_group(self, idx: int) -> str:
        """
        Extract singer ID from filename.
        MIR-1K filenames are in format "SingerId_SongId_ClipId"
        """
        wav_path = self.wav_pitch_pairs[idx][0]
        parts = wav_path.stem.split("_")
        return parts[0] if len(parts) >= 1 else "unknown"

    def __len__(self) -> int:
        return len(self.wav_pitch_pairs)

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        wav_path, pitch_path = self.wav_pitch_pairs[idx]

        try:
            # Load stereo audio - pitch annotations are for singing voice (right channel) only
            waveform, sr = torchaudio.load(wav_path)

            # Always use singing voice channel (right channel, index 1) since that's what has pitch labels
            if waveform.shape[0] == 2:  # Stereo
                waveform = waveform[1]  # Right channel = singing voice
            else:  # Mono - use as is (assume it's singing voice)
                waveform = waveform.squeeze()

        except Exception as e:
            raise OSError(f"Error loading audio file {wav_path}: {e!s}") from e

        try:
            # Load pitch labels
            # MIR-1K pitch labels are in semitone format, space-separated
            pitch_semitone = np.loadtxt(pitch_path)

            # Convert semitone (= MIDI note number) to Hz. In MIR-1K, 0 represents unvoiced frames.
            pitch_hz = np.zeros_like(pitch_semitone)
            voiced_mask = pitch_semitone > 0
            if np.any(voiced_mask):
                pitch_hz[voiced_mask] = librosa.midi_to_hz(pitch_semitone[voiced_mask])

            pitch = torch.from_numpy(pitch_hz).float()
            periodicity = torch.from_numpy(voiced_mask.astype(np.float32))

        except Exception as e:
            raise OSError(f"Error loading pitch file {pitch_path}: {e!s}") from e

        # Resample labels at their true timestamps. MIR-1K labels come from 40 ms windows hopped by
        # 20 ms from t=0, so frame i spans [i*20, i*20+40] ms and is centered at (i+1)*20 ms. This is
        # confirmed by the label count: every file has n = floor(dur/20 ms) - 1 labels (last centre
        # at n*20 ms <= dur-20 ms), the fingerprint of centres at 20, 40, ..., dur-20 ms; the (i+0.5)
        # convention would instead predict n = floor(dur/20 ms). Using (i+0.5) put every label half a
        # hop (~10 ms) early on the eval grid.
        label_times = (np.arange(pitch.numel()) + 1.0) * self.PITCH_LABEL_HOP_SECONDS
        waveform, pitch, periodicity = self.process_sample(
            waveform, pitch, periodicity, sr, label_times=label_times
        )

        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": wav_path,
        }
