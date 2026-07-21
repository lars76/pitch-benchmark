import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio

from metrics import (
    is_voiced,  # single definition of "voiced" (periodicity >= VOICED_THRESHOLD)
)
from resampling import (
    resample_to_grid,  # one resampler shared with algorithms/ and scripts/
)


def frame_rms(
    waveform: torch.Tensor,
    hop_size: int,
    n_frames: int,
    frame_length: int | None = None,
    *,
    center: bool,
) -> torch.Tensor:
    """Per-frame RMS over `n_frames` hop-spaced windows of `frame_length` samples (default hop_size).

    `center` is REQUIRED (no default) because it decides which samples frame i's energy describes,
    and getting that silently wrong is exactly the half-hop timing-bug class the calibration work
    eliminated:
      - center=True:  frame i spans [i*hop - L//2, i*hop + L - L//2), centered on sample i*hop,
        matching the benchmark grid contract (frame i IS the audio centered at i*hop). Every
        voicing/silence gate in the repo uses this.
      - center=False: frame i spans [i*hop, i*hop + L), forward-looking, energy centered half a
        window LATE relative to the grid. Only for callers that explicitly want trailing windows.

    The signal is zero-padded (left for centering, right so the final window is complete); returns
    shape (n_frames,). This is the one per-frame energy primitive behind every RMS-vs-peak
    silence/voicing gate in the repo (NSynth's voicing detector, the laryngograph energy gate, and
    the offline consensus silence gate). Each caller keeps its OWN normalization/threshold/policy
    on top; this returns only the raw per-frame RMS they share."""
    if frame_length is None:
        frame_length = hop_size
    waveform = waveform.squeeze()
    if center:
        waveform = torch.nn.functional.pad(waveform, (frame_length // 2, 0))
    total_samples_needed = (n_frames - 1) * hop_size + frame_length
    padding_needed = max(0, total_samples_needed - waveform.size(-1))
    if padding_needed > 0:
        waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
    frames = waveform.unfold(0, frame_length, hop_size)[:n_frames]
    return torch.sqrt(torch.mean(frames**2, dim=1))


class PitchDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for audio datasets with pitch and periodicity processing.

    Provides core functionality for audio processing, pitch validation, and resampling
    while defining an interface that derived classes must implement.

    Args:
        sample_rate (int): Target sample rate in Hz
        hop_size (int): Number of audio samples between consecutive frames
        clip_pitch (bool, optional): How to handle ground-truth f0 outside [fmin, fmax].
            If False (default), out-of-range frames are marked UNVOICED (periodicity 0, pitch 0),
            since the label is unreliable there. If True, pitch is clamped into [fmin, fmax].
            Defaults to False
        normalize_audio (bool, optional): Whether to normalize audio to [-1, 1]. Defaults to True

    Ground-truth contract (what __getitem__ must return):
        Required keys, all 1-D and frame-aligned: length F = audio_samples // hop_size, where frame m
        is the audio CENTERED at sample m*hop_size (time m*hop_size / sample_rate). This center-aligned
        grid is shared with the predictions: both ground truth and estimates are resampled onto it
        (resampling.resample_to_grid), so they line up frame-for-frame.
          - "audio":       float32 tensor, shape (T,), range [-1, 1] at `sample_rate`.
          - "pitch":       float32 tensor (Hz). INVARIANT: pitch > 0 => voiced, i.e. pitch == 0 on
                           every unvoiced frame. The converse need NOT hold: a voiced frame may have
                           pitch == 0 (the "voiced, pitch uncertain" state below). NaN -> 0.
          - "periodicity": float32 tensor, a voicing CONFIDENCE in [0, 1] (a binary {0, 1} label
                           is the certain case). voiced <=> periodicity >= 0.5 (metrics.is_voiced),
                           the single definition of "voiced" across the benchmark.
        Optional keys: "pitch_conf" (float32 [0,1]; laryngograph consensus, gates RPA, not F1),
          "notes" (List[{start, end, midi_pitch}]; datasets with provides_notes=True), an identifier
          ("wav_path" or dataset-specific).
        The three frame states (see LaryngographSpeechDataset):
          voiced, pitch known    -> pitch > 0, periodicity >= 0.5   (scored for F1 and RPA)
          voiced, pitch uncertain-> pitch = 0, periodicity >= 0.5   (F1 only; finite-rule drops RPA)
          unvoiced               -> pitch = 0, periodicity <  0.5   (F1 negative)
        Out-of-range f0 policy: ground truth outside [fmin, fmax] is marked UNVOICED (not scored);
        predictions are CLAMPED to [fmin, fmax]. (Asymmetric by design: don't penalize a tracker for
        the label's range; constrain the tracker to its operating band.)
    """

    DEFAULT_FMIN = 46.875  # Default minimum frequency (G1)
    DEFAULT_FMAX = 2093.75  # Default maximum frequency (C7)

    # Capability flag: True on datasets whose __getitem__ also yields a "notes" key (ground-truth
    # note intervals). Notes are an OPTIONAL capability of a pitch dataset, not a separate dataset
    # type; the note track selects note datasets by this flag (via datasets.list_note_datasets) and
    # validates it up front, so a note run against a non-notes dataset errors instead of silently
    # scoring nothing.
    provides_notes = False

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        clip_pitch: bool = False,
        normalize_audio: bool = True,
        use_cache: bool = True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size

        # Subclass fmin/fmax if the subclass defines them, else the class defaults.
        self.fmin = getattr(self.__class__, "fmin", self.DEFAULT_FMIN)
        self.fmax = getattr(self.__class__, "fmax", self.DEFAULT_FMAX)

        self.clip_pitch = clip_pitch
        self.normalize_audio = normalize_audio
        # In-memory decode cache: __getitem__ stores each loaded sample dict so a second pass over the
        # dataset (the next algorithm) reuses it instead of re-decoding. See __getitem__.
        self.use_cache = use_cache
        self.data_cache: dict[int, dict[str, torch.Tensor | Path]] = {}
        self._validate_init_params(sample_rate, hop_size, self.fmin, self.fmax)

    def _validate_init_params(
        self, sample_rate: int, hop_size: int, fmin: float, fmax: float
    ) -> None:
        """Validates initialization parameters."""
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin} Hz) must be less than fmax ({fmax} Hz)")
        if fmin < 0:
            raise ValueError(f"fmin ({fmin} Hz) must be non-negative")
        if fmax > sample_rate / 2:
            raise ValueError(
                f"fmax ({fmax} Hz) must not exceed Nyquist frequency ({sample_rate / 2} Hz)"
            )

    def get_group(self, idx: int) -> str:
        """Return group identifier for sample (speaker/instrument)"""
        return str(idx)  # Default: each sample is its own group

    def _validate_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Validate, NaN-clean, optionally peak-normalize, and clamp audio to [-1, 1]."""
        if audio.dim() not in {1, 2}:
            raise ValueError(f"Audio must be 1D or 2D, got {audio.dim()}D")

        audio = torch.nan_to_num(audio, nan=0)

        if torch.all(audio == 0):
            raise ValueError("Silent audio!")

        if self.normalize_audio:
            max_abs = audio.abs().max()
            if max_abs > 1:  # Normalize only if the range exceeds -1 to 1
                audio = audio / max_abs

        return audio.clamp(-1.0, 1.0)

    def _apply_range_gate(
        self, pitch: torch.Tensor, periodicity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enforce the shared label contract: a NONZERO f0 outside [fmin, fmax] marks the frame
        unvoiced, then pitch is zeroed on every unvoiced frame (so pitch > 0 implies voiced). pitch==0
        is the "voiced, pitch uncertain" marker (periodicity may stay >= 0.5) and is left voiced. Used
        by both _validate_pitch (clip_pitch=False) and the laryngograph consensus path; with
        clip_pitch=True pitch is already clamped in range, so the gate only enforces the invariant."""
        # Only gate genuinely out-of-range (nonzero) pitch; keep the pitch==0 voiced-uncertain state.
        in_range = (pitch == 0) | ((pitch >= self.fmin) & (pitch <= self.fmax))
        periodicity = periodicity * in_range.to(periodicity.dtype)
        pitch = pitch * is_voiced(periodicity).to(pitch.dtype)
        return pitch, periodicity

    def _validate_pitch(
        self,
        pitch: torch.Tensor,
        periodicity: torch.Tensor,
        notes: list[dict] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict] | None]:
        """Enforce the label contract on pitch, periodicity, and optionally notes.

        NaN pitch -> 0; periodicity is clamped to a [0,1] confidence. By default (clip_pitch=False)
        f0 outside [fmin, fmax] marks the frame UNVOICED and out-of-range notes are dropped; with
        clip_pitch=True pitch and note frequencies are clamped into [fmin, fmax] instead. Finally
        pitch is zeroed on every unvoiced frame so that pitch > 0 <=> voiced.

        Notes carry 'start', 'end', 'midi_pitch' and get the same frequency constraint as pitch.
        """
        # Validate pitch and periodicity shapes
        if pitch.shape != periodicity.shape:
            raise ValueError(
                f"Pitch and periodicity shapes must match: {pitch.shape} vs {periodicity.shape}"
            )

        # NaN pitch means "no pitch" -> 0 (the unvoiced sentinel), matching the prediction
        # side and the global convention (NOT fmin, which would fake an in-range reading).
        pitch = torch.nan_to_num(pitch, nan=0.0)

        # periodicity is a voicing confidence in [0, 1] (a binary {0, 1} label is the certain case).
        periodicity = torch.nan_to_num(periodicity, nan=0.0).clamp(0, 1)

        if self.clip_pitch:
            pitch = torch.clamp(pitch, self.fmin, self.fmax)

            # Clamp note frequencies the same way.
            if notes is not None:
                processed_notes = []
                for note in notes:
                    midi_pitch = note["midi_pitch"]
                    freq_hz = librosa.midi_to_hz(midi_pitch)

                    freq_hz_clipped = max(self.fmin, min(freq_hz, self.fmax))
                    midi_pitch_clipped = librosa.hz_to_midi(freq_hz_clipped)

                    processed_note = note.copy()
                    processed_note["midi_pitch"] = float(midi_pitch_clipped)
                    processed_notes.append(processed_note)
                notes = processed_notes
        else:
            # Drop notes outside [fmin, fmax] (the tracker can't detect them).
            if notes is not None:
                processed_notes = []
                for note in notes:
                    midi_pitch = note["midi_pitch"]
                    freq_hz = librosa.midi_to_hz(midi_pitch)

                    if self.fmin <= freq_hz <= self.fmax:
                        processed_notes.append(note.copy())
                notes = processed_notes

        # Out-of-range f0 -> unvoiced, then pitch = 0 on every unvoiced frame (pitch > 0 <=> voiced).
        # With clip_pitch=True pitch is already clamped in range, so this only enforces the invariant.
        # periodicity stays a float32 [0,1] confidence (NOT bool) so the type is uniform across all
        # datasets; the metric thresholds it via metrics.is_voiced.
        pitch, periodicity = self._apply_range_gate(pitch, periodicity)
        return pitch, periodicity.float(), notes

    @staticmethod
    def _resample_labels_to_grid(
        pitch: torch.Tensor,
        periodicity: torch.Tensor,
        native_times: np.ndarray,
        target_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample native (f0, voicing) labels sampled at ``native_times`` onto ``target_times``
        by SAMPLING AT THE TRUE TIMES (``np.interp``, coordinate-based, so no index time-warp).

        f0 follows the mir_eval melody convention so it never interpolates *through* an unvoiced
        zero (which would fabricate boundary frequencies, e.g. 110 Hz at a voiced->unvoiced edge):
          1. forward-fill the last voiced value across the unvoiced gaps,
          2. interpolate in cents (log-Hz, perceptually uniform),
          3. re-mask the frames that were unvoiced in the source back to 0.
        Voicing is resampled nearest. Target times outside the annotation span are unvoiced.
        Reproduces ``mir_eval.melody.resample_melody_series`` (asserted in tests).

        Why not ``F.interpolate``: it is INDEX-based (it ignores ``native_times``) and stretches
        the contour onto the output length, time-warping it (~half a frame mid-clip on a fine
        annotation). ``np.interp`` samples at the actual timestamps, which is what we need.

        Delegates to ``resampling.resample_to_grid`` (voicing nearest, the binary-label setting) so
        the ground truth, the prediction wrappers and the consensus generator all align identically.
        """
        pitch_g, per_g = resample_to_grid(
            pitch.detach().cpu().numpy(),
            periodicity.detach().cpu().numpy(),
            native_times,
            target_times,
            voicing_kind="nearest",
        )
        return torch.from_numpy(pitch_g).float(), torch.from_numpy(per_g).float()

    def _prepare_audio(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Squeeze to mono, resample to self.sample_rate, and validate; returns (1, num_samples).
        This is the audio half of process_sample, factored out so the laryngograph consensus path can reuse
        it: that path ships precomputed grid-aligned labels and so skips label resampling, but still
        needs identical audio handling."""
        audio = audio.squeeze()
        if audio.dim() == 2:            # genuine multi-channel -> average to mono (squeeze only drops
            audio = audio.mean(0)       # size-1 dims, so a real stereo signal would slip through as 2ch)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if orig_sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio, orig_freq=orig_sr, new_freq=self.sample_rate
            )
        return self._validate_audio(audio)

    def _load_csv_f0_annotation(
        self, csv_path: Path, sep: str = ","
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Load (times, pitch, periodicity) from a headerless 2-column F0 file: column 0 = annotation
        timestamps (seconds), column 1 = F0 in Hz with 0 = unvoiced. Periodicity is binary (pitch >
        0). Shared by the comma-CSV loaders (MDB, Bach10Synth, Vocadito) and URMP's whitespace-
        separated F0s files (sep=r"\\s+")."""
        try:
            data = pd.read_csv(csv_path, header=None, sep=sep).values
            times = data[:, 0].astype(float)
            pitch = torch.from_numpy(data[:, 1]).float()
            periodicity = (pitch > 0).float()
            return times, pitch, periodicity
        except Exception as e:
            raise OSError(f"Error loading annotation file {csv_path}: {e!s}") from e

    def _load_notes_annotation(self, csv_path: Path, sep: str = ",") -> list[dict]:
        """Load note events from a headerless 3-column file: ``(onset_s, f0_hz, duration_s)`` -> a list
        of ``{start, end, midi_pitch}``, midi via ``librosa.hz_to_midi`` (the single hz->midi
        definition). Non-positive f0 rows are skipped (no MIDI pitch). Shared by Vocadito (comma) and
        URMP (whitespace, sep=r"\\s+"). Returns ``[]`` and warns if the file cannot be read."""
        try:
            data = pd.read_csv(csv_path, header=None, sep=sep).values
        except Exception as e:
            warnings.warn(f"Error loading notes file {csv_path}: {e!s}", stacklevel=2)
            return []
        notes = []
        for row in data:
            start, pitch_hz, duration = float(row[0]), float(row[1]), float(row[2])
            if pitch_hz <= 0:                       # a note with no positive f0 has no MIDI pitch
                continue
            notes.append(
                {"start": start, "end": start + duration, "midi_pitch": float(librosa.hz_to_midi(pitch_hz))}
            )
        return notes

    def process_sample(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor,
        periodicity: torch.Tensor,
        orig_sr: int,
        notes: list[dict] | None = None,
        *,
        label_times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict] | None]:
        """Resample audio and labels onto the eval grid, then enforce the label contract.

        Audio is squeezed to mono, resampled to self.sample_rate and validated; pitch/periodicity
        are resampled by true time onto the center-aligned grid and range-gated, with the same
        frequency constraint applied to any notes.

        label_times (np.ndarray): true time (seconds) of each input pitch/periodicity frame;
            required, so the labels resample by true time (no index warp).

        Returns processed (audio, pitch, periodicity[, notes]).
        """
        # Squeeze to mono, resample, and validate (shared with the laryngograph consensus path)
        audio = self._prepare_audio(audio, orig_sr)

        target_length = audio.size(-1) // self.hop_size
        if target_length < 1:
            # Labels could not be put on the eval grid at all; returning them at native length would
            # silently break the frame-alignment contract downstream.
            raise ValueError(
                f"{type(self).__name__}: audio too short for one frame "
                f"(samples={audio.size(-1)}, hop={self.hop_size})"
            )

        # Contract: one true frame time per label frame (and pitch/periodicity stay frame-aligned).
        label_times = np.asarray(label_times, dtype=np.float64).reshape(-1)
        if not (label_times.shape[0] == pitch.reshape(-1).numel() == periodicity.reshape(-1).numel()):
            raise ValueError(
                f"{type(self).__name__}: label_times ({label_times.shape[0]}), pitch "
                f"({pitch.reshape(-1).numel()}) and periodicity ({periodicity.reshape(-1).numel()}) "
                f"must be frame-aligned (one true time per pitch/periodicity frame)."
            )

        # Resample labels onto the eval grid at their TRUE frame times (label_times),
        # mir_eval-faithful, no index warp. Every dataset supplies label_times.
        target_times = np.arange(target_length) * (self.hop_size / self.sample_rate)
        pitch, periodicity = self._resample_labels_to_grid(
            pitch, periodicity, label_times, target_times
        )

        pitch, periodicity, notes = self._validate_pitch(pitch, periodicity, notes)

        if notes is None:
            return audio.squeeze(0), pitch, periodicity
        else:
            return audio.squeeze(0), pitch, periodicity, notes

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | Path]:
        """Return sample `idx` as a dict, caching the decoded result (see `use_cache`).

        The load itself lives in the subclass `_load_sample`; this wrapper owns the bounds check and
        the one shared decode-cache. Callers always get a fresh shallow copy, so replacing keys on a
        returned dict can never corrupt the cache (tensor values are still shared; consumers must
        replace, not mutate in place, which is what datasets.augment does)."""
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        if self.use_cache and idx in self.data_cache:
            return dict(self.data_cache[idx])
        sample = self._load_sample(idx)
        if self.use_cache:
            self.data_cache[idx] = sample
            return dict(sample)
        return sample

    @abstractmethod
    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        """Decode and process sample `idx` into its result dict (no caching; see __getitem__)."""
