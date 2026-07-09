from pathlib import Path

import librosa
import pandas as pd
import torch
import torchaudio

from .base import PitchDataset


class PitchDatasetVocadito(PitchDataset):
    """
    Implementation of PitchDataset for the vocadito dataset.

    The vocadito dataset contains 40 short excerpts of solo, monophonic singing.
    For more details, see the technical report: https://zenodo.org/records/5578807

    Args:
        root_dir (str): Root directory of the vocadito dataset.
        use_cache (bool, optional): Whether to cache loaded data. Defaults to True.
        **kwargs: Additional arguments passed to the base PitchDataset.
    """

    fmin = 80
    fmax = 1000
    # Constant timing correction (seconds) added to the F0 annotation timestamps: the label-offset
    # sweep (scripts/check_dataset_alignment.py, TIMING.md) measures the annotation content a
    # systematic +2.7 ms after the CSV timestamps, agreeing to 0.1 ms across three calibrated
    # reference trackers (+2.65/+2.77/+2.73 ms) -- an annotation-tool frame convention offset.
    F0_LABEL_OFFSET_SECONDS = 0.0027

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(use_cache=use_cache, **kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        # Set up directory paths according to vocadito structure
        self.audio_dir = self.root_dir / "Audio"
        self.annot_f0_dir = self.root_dir / "Annotations" / "F0"
        self.annot_notes_dir = self.root_dir / "Annotations" / "Notes"
        self.metadata_path = self.root_dir / "vocadito_metadata.csv"

        if not all(
            [
                self.audio_dir.exists(),
                self.annot_f0_dir.exists(),
                self.annot_notes_dir.exists(),
                self.metadata_path.exists(),
            ]
        ):
            raise FileNotFoundError(
                "A required directory or file (Audio, Annotations/F0, Annotations/Notes, "
                "or vocadito_metadata.csv) was not found."
            )

        # Load metadata for grouping
        try:
            self.metadata = pd.read_csv(self.metadata_path)
            self.metadata.set_index("track_id", inplace=True)
        except Exception as e:
            raise OSError(f"Error loading metadata file {self.metadata_path}: {e!s}") from e

        # Find all valid wav-annotation pairs
        self.wav_f0_notes_pairs = self._find_wav_f0_notes_pairs()
        if not self.wav_f0_notes_pairs:
            raise ValueError(
                f"No valid wav-F0-notes annotation pairs found in '{root_dir}'"
            )

    def _find_wav_f0_notes_pairs(self) -> list[tuple[Path, Path, Path]]:
        """
        Find matching WAV, F0 CSV, and Notes CSV annotation file pairs.

        This method identifies audio files and their corresponding F0 and note annotation
        files. It specifically looks for 'vocadito_*_notesA1.csv' as the primary
        note annotation source. Pairs are only included if all three files exist.

        Returns:
            List[Tuple[Path, Path, Path]]: A list of tuples, where each tuple contains
                                           (audio_file_path, f0_csv_path, notes_csv_path).
        """
        pairs = []
        for wav_path in sorted(self.audio_dir.glob("vocadito_*.wav")):
            file_stem = wav_path.stem  # e.g., 'vocadito_1'
            f0_csv_path = self.annot_f0_dir / f"{file_stem}_f0.csv"
            # Using Annotator 1's notes for consistency in benchmarking
            notes_csv_path = self.annot_notes_dir / f"{file_stem}_notesA1.csv"

            if f0_csv_path.exists() and notes_csv_path.exists():
                pairs.append((wav_path, f0_csv_path, notes_csv_path))
            else:
                if not f0_csv_path.exists():
                    print(
                        f"Warning: Skipping {file_stem} due to missing F0 annotation: {f0_csv_path}"
                    )
                if not notes_csv_path.exists():
                    print(
                        f"Warning: Skipping {file_stem} due to missing Notes annotation: {notes_csv_path}"
                    )
        return pairs

    def _load_notes_annotation(self, csv_path: Path) -> list[dict]:
        """Load and convert annotations to standardized format."""
        notes = []
        try:
            data = pd.read_csv(csv_path, header=None).values
            for row in data:
                start = float(row[0])
                pitch_hz = float(row[1])
                duration = float(row[2])
                end = start + duration

                # Convert Hz to MIDI
                midi_pitch = librosa.hz_to_midi(pitch_hz)

                notes.append(
                    {
                        "start": start,
                        "end": end,
                        "midi_pitch": float(midi_pitch),
                    }
                )
        except Exception as e:
            print(f"Warning: Error loading {csv_path}: {e!s}")
            return []
        return notes

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.wav_f0_notes_pairs)

    def get_group(self, idx: int) -> str:
        """Return the group identifier for a sample, which is the singer_id."""
        # Extract track_id from filename, e.g., 'vocadito_1' -> 1
        wav_path = self.wav_f0_notes_pairs[idx][0]
        track_id = int(wav_path.stem.split("_")[1])
        # Look up singer_id from the metadata
        singer_id = self.metadata.loc[track_id, "singer_id"]
        return str(singer_id)

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        """Load one sample: audio, F0, periodicity, and musical notes.

        Returns a dict with "audio", "pitch" (F0 Hz), "periodicity", "notes" (list of dicts with
        'start'/'end'/'midi_pitch'), and "wav_path".
        """
        wav_path, f0_csv_path, notes_csv_path = self.wav_f0_notes_pairs[idx]

        try:
            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform.squeeze()
        except Exception as e:
            raise OSError(f"Error loading audio file {wav_path}: {e!s}") from e

        times, pitch, periodicity = self._load_csv_f0_annotation(f0_csv_path)
        times = times + self.F0_LABEL_OFFSET_SECONDS  # measured annotation offset (class comment)
        notes = self._load_notes_annotation(notes_csv_path)

        # Process the sample (resample labels at their true annotation timestamps)
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
