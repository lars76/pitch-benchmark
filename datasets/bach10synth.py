from pathlib import Path

import torch
import torchaudio

from .base import PitchDataset


class PitchDatasetBach10Synth(PitchDataset):
    """
    Implementation of PitchDataset for the monophonic (stem) tracks of the Bach10-mf0-synth dataset.

    This dataset contains individual solo stems from Bach10,
    along with perfect single F0 annotations.
    Annotations for stems are provided at a hop size of 128/44100 seconds (~2.9 ms).

    Args:
        root_dir (str): Root directory of the Bach10-mf0-synth dataset
        use_cache (bool, optional): Whether to cache loaded data. Defaults to True
        **kwargs: Additional arguments passed to PitchDataset
    """

    fmin = 65
    fmax = 2093

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(use_cache=use_cache, **kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        self.audio_dir = self.root_dir / "audio_stems"
        self.annot_dir = self.root_dir / "annotation_stems"

        if not self.audio_dir.exists() or not self.annot_dir.exists():
            raise FileNotFoundError(
                "Audio stems or annotation stems directory not found. "
                "Ensure 'audio_stems' and 'annotation_stems' exist in the root directory."
            )

        # Find all valid wav-annotation pairs for stems
        self.wav_f0_pairs = self._find_wav_f0_pairs()
        if not self.wav_f0_pairs:
            raise ValueError(
                f"No valid wav-annotation pairs found in '{root_dir}/audio_stems' and '{root_dir}/annotation_stems'."
            )

    def _find_wav_f0_pairs(self) -> list[tuple[Path, Path]]:
        """Find matching WAV and CSV annotation file pairs for the stem dataset."""
        pairs = []
        audio_files = self.audio_dir.glob("*.RESYN.wav")
        for wav_path in audio_files:
            csv_path = self.annot_dir / wav_path.name.replace(".wav", ".csv")
            if csv_path.exists():
                pairs.append((wav_path, csv_path))
        return sorted(pairs)

    def __len__(self) -> int:
        """Return the total number of samples (stems) in the dataset."""
        return len(self.wav_f0_pairs)

    def get_group(self, idx: int) -> str:
        """Return group identifier for sample (piece ID and title)"""
        file_path = self.wav_f0_pairs[idx][0]
        # 01_AchGottundHerr_bassoon.RESYN.wav -> "01_AchGottundHerr" (pieceID + title)
        parts = file_path.stem.split("_")
        return f"{parts[0]}_{parts[1]}"

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        """Load and process one sample (stem) from the dataset."""
        wav_path, csv_path = self.wav_f0_pairs[idx]

        try:
            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform.squeeze()
        except Exception as e:
            raise OSError(f"Error loading audio file {wav_path}: {e!s}") from e

        times, pitch, periodicity = self._load_csv_f0_annotation(csv_path)

        # Process the sample (resample labels at their true annotation timestamps)
        waveform, pitch, periodicity = self.process_sample(
            waveform, pitch, periodicity, sr, label_times=times
        )

        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": wav_path,
        }
