import re
from pathlib import Path

import torch
import torchaudio

from .base import PitchDataset


class PitchDatasetMDBStemSynth(PitchDataset):
    """
    Implementation of PitchDataset for the MDB-stem-synth dataset.

    The dataset contains resynthesized solo stems from MedleyDB with perfect F0 annotations.
    Annotations are provided at a hop size of 128/44100 seconds (~2.9 ms).

    Args:
        root_dir (str): Root directory of the MDB-stem-synth dataset
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
                "Audio stems or annotation stems directory not found"
            )

        # Find all valid wav-annotation pairs
        self.wav_f0_pairs = self._find_wav_f0_pairs()
        if not self.wav_f0_pairs:
            raise ValueError(f"No valid wav-annotation pairs found in '{root_dir}'")

    def _find_wav_f0_pairs(self) -> list[tuple[Path, Path]]:
        """Find matching WAV and CSV annotation file pairs in the dataset."""
        pairs = []
        for wav_path in self.audio_dir.glob("*.RESYN.wav"):
            csv_path = self.annot_dir / wav_path.name.replace(".wav", ".csv")
            if csv_path.exists():
                pairs.append((wav_path, csv_path))
        return sorted(pairs)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.wav_f0_pairs)

    def get_group(self, idx: int) -> str:
        """Return group identifier for sample = the source TRACK (Artist_Title).

        Filenames are `Artist_Title_STEM_NN.RESYN.wav`, so the previous `stem.split("_")[0]` returned
        only the first token, collapsing every distinct `MusicDelta_*` track (Beethoven, Rock, ...)
        into one 100-clip 'MusicDelta' group, which fabricates within-group correlation for the cluster
        bootstrap. The leakage-safe unit is the full track: drop `.RESYN.wav` then strip the trailing
        `_STEM_NN` stem index."""
        name = self.wav_f0_pairs[idx][0].name
        name = re.sub(r"\.RESYN\.wav$", "", name)
        return re.sub(r"_STEM_\d+$", "", name)

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        """Load and process one sample from the dataset."""
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
