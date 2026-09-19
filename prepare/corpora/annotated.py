import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from constants import CORPUS_FMIN, MUSIC_FMAX
from grid import frame_times, is_voiced, resample_to_grid

from .base import PitchDataset, find_wav_f0_pairs, frame_rms

RMS_EPS = 1e-10
RMS_RATIO_FLOOR = 1e-5
SILENT_DB = -100.0


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


class PitchDatasetNSynth(PitchDataset):

    fmin = CORPUS_FMIN
    fmax = MUSIC_FMAX

    UNRELIABLE_PITCH_FAMILIES = {"synth_lead", "organ", "mallet"}
    SILENCE_THRESHOLD_DB = -40.0
    MAX_ITEM_SECONDS = 4.0

    def __init__(self, root_dir, instrument_sources=None, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        json_path = self.root_dir / "examples.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        with open(json_path) as f:
            self.metadata = json.load(f)

        self.examples = []
        for note_str, info in self.metadata.items():
            info["pitch_hz"] = float(midi_to_hz(info["pitch"]))
            if not self.fmin <= info["pitch_hz"] <= self.fmax:
                continue
            if instrument_sources and info["instrument_source_str"] not in instrument_sources:
                continue
            if info["instrument_family_str"] in self.UNRELIABLE_PITCH_FAMILIES:
                continue
            if "multiphonic" in info["qualities_str"]:
                continue
            self.examples.append((note_str, info))

        if not self.examples:
            raise ValueError("No NSynth examples admitted (instrument sources or frequency range)")

    def _detect_voiced_frames(self, waveform, num_frames):
        rms = frame_rms(waveform, self.hop_size, num_frames)

        max_rms = torch.max(rms)
        if max_rms > RMS_EPS:
            rms_db = 20 * torch.log10(torch.clamp(rms / max_rms, min=RMS_RATIO_FLOOR))
        else:
            rms_db = torch.full_like(rms, SILENT_DB)

        non_silent_frames = (rms_db > self.SILENCE_THRESHOLD_DB).nonzero(as_tuple=True)[0]

        voiced_mask = torch.zeros(num_frames)

        if len(non_silent_frames) > 0:
            last_voiced_frame_index = non_silent_frames[-1]
            voiced_mask[: last_voiced_frame_index + 1] = 1

        return voiced_mask

    def get_group(self, idx):
        _, info = self.examples[idx]
        return info["instrument_str"]

    def __len__(self):
        return len(self.examples)

    def _load_sample(self, idx):
        note_str, info = self.examples[idx]
        wav_path = self.root_dir / "audio" / f"{note_str}.wav"
        waveform, sample_rate = self._load_waveform(wav_path)

        num_frames = 1 + (waveform.size(-1) // self.hop_size)
        pitch = torch.full((num_frames,), info["pitch_hz"])

        periodicity = self._detect_voiced_frames(waveform, num_frames)

        label_times = frame_times(num_frames, self.hop_size, sample_rate)

        waveform, pitch, periodicity = self.process_sample(
            waveform, pitch, periodicity, sample_rate, label_times=label_times
        )

        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": wav_path,
        }


class StemSynthDataset(PitchDataset):

    fmin = CORPUS_FMIN
    fmax = MUSIC_FMAX

    def __init__(self, root_dir, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.audio_dir = self.root_dir / "audio_stems"
        self.annot_dir = self.root_dir / "annotation_stems"

        if not self.audio_dir.exists() or not self.annot_dir.exists():
            raise FileNotFoundError(
                f"{root_dir}: 'audio_stems' and 'annotation_stems' must both exist"
            )

        self.wav_f0_pairs = find_wav_f0_pairs(self.audio_dir, self.annot_dir)
        if not self.wav_f0_pairs:
            raise ValueError(f"No valid wav-annotation pairs found in '{root_dir}'")

    def __len__(self):
        return len(self.wav_f0_pairs)

    def item_grid_frames(self, idx):
        return self._grid_frames(self.wav_f0_pairs[idx][0])

    def item_grid_silent(self, idx):
        times, pitch, periodicity = self._load_csv_f0_annotation(self.wav_f0_pairs[idx][1])
        grid = frame_times(self.item_grid_frames(idx), self.hop_size, self.sample_rate)
        _p, per = resample_to_grid(np.asarray(pitch, dtype=float),
                                   np.asarray(periodicity, dtype=float),
                                   np.asarray(times, dtype=float), grid)
        return ~is_voiced(np.asarray(per))

    def _load_sample(self, idx):
        return self._sample_from_csv(*self.wav_f0_pairs[idx])


class PitchDatasetMDBStemSynth(StemSynthDataset):

    def get_group(self, idx):
        name = self.wav_f0_pairs[idx][0].name
        name = re.sub(r"\.RESYN\.wav$", "", name)
        return re.sub(r"_STEM_\d+$", "", name)


class PitchDatasetBach10Synth(StemSynthDataset):

    def get_group(self, idx):
        parts = self.wav_f0_pairs[idx][0].stem.split("_")
        return f"{parts[0]}_{parts[1]}"


class PitchDatasetURMP(PitchDataset):
    fmin = 32.70
    fmax = MUSIC_FMAX

    def __init__(self, root_dir, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        self.stems = []
        for wav in sorted(self.root_dir.rglob("AuSep_*.wav")):
            if wav.name.startswith("._") or "__MACOSX" in str(wav):
                continue
            f0 = wav.parent / (wav.stem.replace("AuSep", "F0s", 1) + ".txt")
            if f0.exists():
                self.stems.append((wav, f0, wav.parent.name))
        if not self.stems:
            raise ValueError(f"No AuSep_*.wav stems with F0s found in '{root_dir}'")

    def __len__(self):
        return len(self.stems)

    def item_grid_frames(self, idx):
        return self._grid_frames(self.stems[idx][0])

    def get_group(self, idx):
        return self.stems[idx][2]

    def _load_sample(self, idx):
        wav_path, f0_path, _ = self.stems[idx]
        return self._sample_from_csv(wav_path, f0_path, delimiter=None)


class PitchDatasetVocadito(PitchDataset):

    fmin = 80
    fmax = 1000
    F0_LABEL_OFFSET_SECONDS = 0.0027

    def __init__(self, root_dir, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        self.audio_dir = self.root_dir / "Audio"
        self.annot_f0_dir = self.root_dir / "Annotations" / "F0"
        self.metadata_path = self.root_dir / "vocadito_metadata.csv"

        if not all([self.audio_dir.exists(), self.annot_f0_dir.exists(), self.metadata_path.exists()]):
            raise FileNotFoundError(
                "A required directory or file (Audio, Annotations/F0, or vocadito_metadata.csv) "
                "was not found."
            )

        with open(self.metadata_path, newline="") as fh:
            self.singer_of = {int(row["track_id"]): str(row["singer_id"]) for row in csv.DictReader(fh)}

        self.wav_f0_pairs = []
        for wav_path in sorted(self.audio_dir.glob("vocadito_*.wav")):
            f0_csv_path = self.annot_f0_dir / f"{wav_path.stem}_f0.csv"
            if f0_csv_path.exists():
                self.wav_f0_pairs.append((wav_path, f0_csv_path))
            else:
                print(f"Warning: Skipping {wav_path.stem} due to missing F0 annotation: {f0_csv_path}")
        if not self.wav_f0_pairs:
            raise ValueError(f"No valid wav-F0 annotation pairs found in '{root_dir}'")

    def __len__(self):
        return len(self.wav_f0_pairs)

    def item_grid_frames(self, idx):
        return self._grid_frames(self.wav_f0_pairs[idx][0])

    def get_group(self, idx):
        wav_path = self.wav_f0_pairs[idx][0]
        return self.singer_of[int(wav_path.stem.split("_")[1])]

    def _load_sample(self, idx):
        return self._sample_from_csv(*self.wav_f0_pairs[idx],
                                     offset=self.F0_LABEL_OFFSET_SECONDS)
