import json
from pathlib import Path
from typing import ClassVar

import librosa
import numpy as np
import torch
import torchaudio

from .base import PitchDataset, frame_rms


class PitchDatasetNSynth(PitchDataset):
    """
    Dataset implementation for the NSynth (Neural Audio Synthesis) dataset.

    This class handles loading and processing of the NSynth dataset, which contains
    musical notes from various instruments. It provides filtering capabilities based
    on instrument types, families and note qualities.

    Args:
        root_dir (str): Path to NSynth dataset directory containing examples.json and audio files
        instrument_sources (Optional[List[str]]): Filter by instrument sources
            Valid options: ["acoustic", "electronic", "synthetic"]
        instrument_families (Optional[List[str]]): Filter by instrument families
            Valid options: ["bass", "brass", "flute", "guitar", "keyboard", "mallet",
                          "organ", "reed", "string", "synth_lead", "vocal"]
        qualities (Optional[List[str]]): Filter by note qualities
            Valid options: ["bright", "dark", "distortion", "fast_decay", "long_release",
                          "multiphonic", "nonlinear_env", "percussive", "reverb", "tempo-synced"]
        exclude_unreliable_pitch (bool): Drop families whose pitch label is an unreliable
            f0 ground truth (organ/mallet/synth_lead). Defaults to True
            (see UNRELIABLE_PITCH_FAMILIES).
        exclude_multiphonic (bool): Drop notes tagged "multiphonic" in the metadata
            (multiple simultaneous pitches contradict the single-pitch label). Defaults to True.
        use_cache (bool): Whether to cache loaded audio in memory. Defaults to True
        silence_threshold_db (float, optional): Threshold in dB below which audio is considered silent.
            Defaults to -40.0
        **kwargs: Additional arguments passed to PitchDataset base class

    Raises:
        ValueError: If no examples match the specified criteria or if invalid parameters are provided
        IOError: If there are errors loading the dataset metadata
    """

    fmin = 65
    fmax = 2093

    # Families where the labelled MIDI pitch is often not the dominant trackable f0:
    # mallet (inharmonic struck-bar partials), organ (octave/fifth-ambiguous registration),
    # synth_lead (arbitrary synthesis). Excluded by default; family-level heuristic.
    UNRELIABLE_PITCH_FAMILIES: ClassVar[set[str]] = {"synth_lead", "organ", "mallet"}

    def __init__(
        self,
        root_dir: str,
        instrument_sources: list[str] | None = None,
        instrument_families: list[str] | None = None,
        qualities: list[str] | None = None,
        exclude_unreliable_pitch: bool = True,
        exclude_multiphonic: bool = True,
        use_cache: bool = True,
        silence_threshold_db: float = -40.0,
        **kwargs,
    ):
        super().__init__(use_cache=use_cache, **kwargs)

        self.root_dir = Path(root_dir)
        self.exclude_unreliable_pitch = exclude_unreliable_pitch
        self.exclude_multiphonic = exclude_multiphonic

        self.silence_threshold_db = silence_threshold_db

        json_path = self.root_dir / "examples.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        with open(json_path) as f:
            self.metadata = json.load(f)

        self.examples = self._prepare_examples(
            instrument_sources, instrument_families, qualities
        )

        if not self.examples:
            raise ValueError(
                "No examples found matching the specified criteria. "
                "Try adjusting the filter parameters (instrument types, families, "
                "qualities, or frequency range)."
            )

    def _prepare_examples(
        self,
        instrument_sources: list[str] | None,
        instrument_families: list[str] | None,
        qualities: list[str] | None,
    ) -> list[tuple[str, dict]]:
        """
        Filters and prepares examples based on specified criteria including instrument sources.

        Args:
            instrument_sources: List of instrument source types to include
            instrument_families: List of instrument family types to include
            qualities: List of note qualities to include

        Returns:
            List[Tuple[str, Dict]]: List of (note_id, metadata) pairs meeting all criteria
        """
        examples = []
        for note_str, info in self.metadata.items():
            pitch_hz = librosa.midi_to_hz(info["pitch"])
            info["pitch_hz"] = pitch_hz

            if (
                instrument_sources
                and info["instrument_source_str"] not in instrument_sources
            ):
                continue
            if (
                instrument_families
                and info["instrument_family_str"] not in instrument_families
            ):
                continue
            if (
                self.exclude_unreliable_pitch
                and info["instrument_family_str"] in self.UNRELIABLE_PITCH_FAMILIES
            ):
                continue
            if self.exclude_multiphonic and "multiphonic" in info["qualities_str"]:
                continue
            if qualities and not any(q in info["qualities_str"] for q in qualities):
                continue

            examples.append((note_str, info))

        return examples

    def _detect_voiced_frames(
        self, waveform: torch.Tensor, num_frames: int, frame_length: int | None = None
    ) -> torch.Tensor:
        """Binary voicing mask: frames up to the last frame whose RMS clears the silence
        threshold (relative to the file's peak RMS) are voiced. frame_length defaults to hop_size."""
        # Per-frame RMS via the shared energy primitive, CENTERED on the grid (frame i is the audio
        # centered at i*hop); forward windows would place the note-release boundary half a window late.
        rms = frame_rms(waveform, self.hop_size, num_frames, frame_length, center=True)

        # Per-frame dB relative to the peak RMS; an all-silent file is -100 dB everywhere.
        eps = 1e-10
        max_rms = torch.max(rms)
        if max_rms > eps:
            rms_db = 20 * torch.log10(torch.clamp(rms / max_rms, min=1e-5))
        else:
            rms_db = torch.full_like(rms, -100.0)

        non_silent_frames = (rms_db > self.silence_threshold_db).nonzero(as_tuple=True)[
            0
        ]

        voiced_mask = torch.zeros(num_frames)

        # Mark everything up to the last non-silent frame as voiced (the note's sustain+release).
        if len(non_silent_frames) > 0:
            last_voiced_frame_index = non_silent_frames[-1]
            voiced_mask[: last_voiced_frame_index + 1] = 1

        return voiced_mask

    def get_group(self, idx: int) -> str:
        _, info = self.examples[idx]
        return info["instrument_family_str"]  # Unique instrument family ID

    def __len__(self) -> int:
        """Returns the number of examples in the filtered dataset."""
        return len(self.examples)

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        """Load and process one item from the dataset.

        Returns a dict with 'audio' [1, samples], 'pitch' [frames], 'periodicity' [frames], and
        'wav_path'. Raises IOError if the audio file fails to load.
        """
        note_str, info = self.examples[idx]
        wav_path = self.root_dir / "audio" / f"{note_str}.wav"

        try:
            waveform, sample_rate = torchaudio.load(wav_path)
        except Exception as e:
            raise OSError(f"Error loading audio file {wav_path}: {e!s}") from e

        # NSynth uses constant pitch throughout each note.
        num_frames = 1 + (waveform.size(-1) // self.hop_size)
        pitch = torch.full((num_frames,), info["pitch_hz"])

        periodicity = self._detect_voiced_frames(waveform, num_frames)

        # Frame i covers original samples [i*hop, (i+1)*hop), so its centre time (matching the
        # centre-aligned eval grid) is (i + 0.5) * hop / sr. Passing these true times routes the
        # labels through the one shared time-based resampler, same as every other dataset.
        label_times = (np.arange(num_frames) + 0.5) * (self.hop_size / sample_rate)

        waveform, pitch, periodicity = self.process_sample(
            waveform, pitch, periodicity, sample_rate, label_times=label_times
        )

        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": wav_path,
        }
