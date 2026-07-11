"""URMP (University of Rochester Multi-modal Music Performance): 44 classical chamber pieces,
duets to quintets, with manually corrected per-track frame-level f0 and note transcriptions.

Each mixed piece is expanded into one item per instrument STEM (``AuSep_*.wav``); the sibling
``F0s_*.txt`` (whitespace ``time freq``, 10 ms hop, 0 = unvoiced) is the gold frame-level label and
``Notes_*.txt`` (``onset freq_hz duration``) the note transcription. Grouping is by piece so a
train/eval split cannot leak a piece across folds. This is gold-annotated music, so there is no
author/consensus distinction (cf. the laryngograph speech corpora)."""
from pathlib import Path

import torch
import torchaudio

from .base import PitchDataset


class PitchDatasetURMP(PitchDataset):
    fmin = 32.70    # C1: chamber ensembles include cello/bass register
    fmax = 2093.75  # C7, the benchmark's shared ceiling

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(use_cache=use_cache, **kwargs)
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        self.stems: list[tuple[Path, Path, Path, str]] = []
        for wav in sorted(self.root_dir.rglob("AuSep_*.wav")):
            if wav.name.startswith("._") or "__MACOSX" in str(wav):
                continue
            f0 = wav.parent / (wav.stem.replace("AuSep", "F0s", 1) + ".txt")
            notes = wav.parent / (wav.stem.replace("AuSep", "Notes", 1) + ".txt")
            if f0.exists() and notes.exists():
                self.stems.append((wav, f0, notes, wav.parent.name))
        if not self.stems:
            raise ValueError(f"No AuSep_*.wav stems with F0s/Notes found in '{root_dir}'")

    def __len__(self) -> int:
        return len(self.stems)

    def get_group(self, idx: int) -> str:
        return self.stems[idx][3]   # piece name -> leakage-safe CV groups

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        wav_path, f0_path, notes_path, _ = self.stems[idx]
        try:
            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform.squeeze()
        except Exception as e:
            raise OSError(f"Error loading audio file {wav_path}: {e!s}") from e

        # F0s_*.txt / Notes_*.txt are whitespace-separated; the shared loaders parse both (sep=r"\s+")
        # so URMP scores under the same label/hz->midi contract as the CSV corpora.
        times, pitch, periodicity = self._load_csv_f0_annotation(f0_path, sep=r"\s+")
        notes = self._load_notes_annotation(notes_path, sep=r"\s+")
        waveform, pitch, periodicity, notes = self.process_sample(
            waveform, pitch, periodicity, sr, notes, label_times=times
        )
        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "notes": notes,
            "wav_path": wav_path,
        }
