"""Base class for laryngograph speech datasets, scored against cross-family consensus labels.

These corpora are recorded with an electroglottograph (EGG, a laryngograph). Ground-truth f0 is
derived offline from the EGG signal by a cross-family consensus (correlation = Praat+REAPER,
period = differentiated-EGG, instantaneous-frequency = Harvest); see
scripts/build_consensus_labels.py, which precomputes one ``<NAME>.npz`` per corpus beside this
module (e.g. ``datasets/laryngograph/PTDB.npz``), keyed by file stem with a (3, n) float32 array
of (voicing_conf, pitch_hz, pitch_conf) per frame. This runtime class only READS those labels; no
laryngograph signal or f0 estimator is needed at benchmark time.

Per-frame label semantics (the three states, expressed via the existing two arrays):
  voiced, pitch-confident   -> (f0 > 0, periodicity = 1)   scored for both F1 and RPA
  voiced, pitch-uncertain   -> (f0 = 0, periodicity = 1)   F1 positive; the metric's finite-frame
                                                           rule drops f0 = 0 from RPA automatically
  unvoiced                  -> (f0 = 0, periodicity = 0)   F1 negative

`label_source="author"` instead loads the dataset's shipped single-method f0 (for reproducibility);
laryngograph-only corpora that ship no author f0 raise from `_load_author_labels`.
"""
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio

from metrics import (
    is_voiced,  # single definition of "voiced" (periodicity >= VOICED_THRESHOLD)
)

from ..base import PitchDataset, frame_rms

# Consensus label files (``<NAME>.npz``) live beside this module, one per corpus.
CONSENSUS_DIR = Path(__file__).parent


def energy_voicing_gate(
    waveform: torch.Tensor, periodicity: torch.Tensor, hop_size: int, thr: float = 0.05
) -> torch.Tensor:
    """Zero periodicity on frames whose mic RMS (normalized to the per-file peak) is below `thr`.
    Frame i's window is CENTERED on sample i*hop, matching the grid contract (frame i is the audio
    centered at i*hop); the previous forward window [i*hop, (i+1)*hop) evaluated the gate half a hop
    late, shifting every voicing boundary ~8 ms (review finding A14; see TIMING.md). Shares the
    per-frame RMS primitive (base.frame_rms) with NSynth's voicing detector and the offline
    consensus silence gate (scripts/build_consensus_labels.py); the linear cutoff here is the policy."""
    n = periodicity.shape[-1]
    a = waveform.reshape(-1)
    if a.numel() < n * hop_size or n == 0:
        return periodicity
    rms = frame_rms(a, hop_size, n, center=True)
    rms_norm = rms / (rms.max() + 1e-8)
    periodicity = periodicity.clone()
    periodicity[rms_norm < thr] = 0
    return periodicity


class LaryngographSpeechDataset(PitchDataset):
    """Abstract base for laryngograph speech corpora. Subclasses set the class attr ``NAME`` (the
    consensus subfolder) and implement ``_discover`` (and optionally ``_load_author_labels``)."""

    NAME: str = None  # consensus subfolder name, e.g. "PTDB"
    # Native hop (seconds) of the shipped author f0, so the author path can resample the labels by
    # their TRUE frame times. A subclass that implements _load_author_labels MUST set this.
    AUTHOR_LABEL_HOP_SECONDS: float = None
    # Constant timing correction (seconds) ADDED to the author label times: the time of the
    # analysis content RELATIVE to the label's nominal i*hop stamp (e.g. the reference tracker's
    # intra-window offset). Measured per corpus by the label-offset sweep
    # (scripts/check_dataset_alignment.py); see TIMING.md.
    AUTHOR_LABEL_OFFSET_SECONDS: float = 0.0

    def __init__(
        self,
        root_dir: str,
        label_source: str = "consensus",
        use_cache: bool = True,
        silence_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(use_cache=use_cache, **kwargs)
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")
        if label_source not in ("consensus", "author"):
            raise ValueError(f"label_source must be 'consensus' or 'author', got '{label_source}'")
        self.label_source = label_source
        self.silence_threshold = silence_threshold

        self._consensus = None  # lazily-opened NpzFile (per process; DataLoader-worker safe)

        self.items: list[tuple[Path, str]] = self._discover()  # [(wav_path, stem), ...]
        if not self.items:
            raise ValueError(f"No audio files found for {self.NAME} in '{root_dir}'")

        # Consensus labels cover only the stems in the committed .npz (a partial generation is
        # common -- the builder also skips files it fails on). Restrict the dataset to labelled
        # stems here, so an unlabelled clip shrinks the dataset instead of raising a KeyError mid
        # DataLoader iteration that would abort the whole benchmark run.
        if label_source == "consensus":
            npz_path = CONSENSUS_DIR / f"{self.NAME}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Consensus labels missing: {npz_path}. Run "
                    f"scripts/build_consensus_labels.py --dataset {self.NAME}."
                )
            with np.load(npz_path) as z:               # read keys only; leave _consensus lazy/worker-safe
                labelled = set(z.files)
            n_discovered = len(self.items)
            self.items = [it for it in self.items if it[1] in labelled]
            if not self.items:
                raise ValueError(
                    f"None of the {n_discovered} discovered {self.NAME} clips have a consensus label "
                    f"in {npz_path.name}. Run scripts/build_consensus_labels.py --dataset {self.NAME}."
                )
            if len(self.items) < n_discovered:
                warnings.warn(
                    f"{self.NAME}: evaluating on {len(self.items)} of {n_discovered} discovered clips "
                    f"({len(labelled)} stems have consensus labels); skipping "
                    f"{n_discovered - len(self.items)} unlabelled. Regenerate with "
                    f"scripts/build_consensus_labels.py --dataset {self.NAME} to cover all.",
                    stacklevel=2,
                )

        if label_source == "consensus" and (self.sample_rate != 16000 or self.hop_size != 256):
            warnings.warn(
                f"{self.NAME} consensus labels were generated at 16 kHz / hop 256; running at "
                f"{self.sample_rate} Hz / hop {self.hop_size} may misalign labels.",
                stacklevel=2,
            )

    # ----- subclass hooks ----------------------------------------------------
    def _discover(self) -> list[tuple[Path, str]]:
        """Return [(wav_path, stem), ...]. `stem` keys both the cache and the consensus npz."""
        raise NotImplementedError

    def _load_author_labels(self, wav_path: Path, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pitch, periodicity) from the dataset's shipped f0. Default: none exist."""
        raise NotImplementedError(
            f"{self.NAME} ships no author f0 (laryngograph-only); use label_source='consensus'."
        )

    # ----- core --------------------------------------------------------------
    def _consensus_npz(self):
        """Lazily open the dataset's consensus label file (one .npz keyed by file stem)."""
        if self._consensus is None:
            p = CONSENSUS_DIR / f"{self.NAME}.npz"
            if not p.exists():
                raise FileNotFoundError(
                    f"Consensus labels missing: {p}. Run "
                    f"scripts/build_consensus_labels.py --dataset {self.NAME}."
                )
            self._consensus = np.load(p)
        return self._consensus

    def __len__(self) -> int:
        return len(self.items)

    def get_group(self, idx: int) -> str:
        parts = self.items[idx][1].split("_")
        return parts[1] if len(parts) >= 2 else "unknown"  # e.g. "M01"

    @staticmethod
    def _fit(x: torch.Tensor, n: int, pad) -> torch.Tensor:
        """Clip or zero/False-pad a 1-D label array to exactly n frames."""
        m = x.numel()
        if m == n:
            return x
        if m > n:
            return x[:n]
        out = torch.full((n,), pad, dtype=x.dtype)
        out[:m] = x
        return out

    def _load_consensus(self, stem: str, target_length: int):
        """Return (voicing_conf, pitch_hz, pitch_conf) as 1-D tensors fitted to target_length."""
        d = self._consensus_npz()
        if stem not in d.files:
            raise KeyError(
                f"No consensus label for '{stem}' in {self.NAME}.npz "
                f"(regenerate with scripts/build_consensus_labels.py --dataset {self.NAME})."
            )
        arr = np.asarray(d[stem], dtype=np.float32)   # (3, n): voicing_conf, pitch_hz, pitch_conf
        vconf = self._fit(torch.from_numpy(arr[0]), target_length, 0.0)
        phz = self._fit(torch.from_numpy(arr[1]), target_length, 0.0)
        pconf = self._fit(torch.from_numpy(arr[2]), target_length, 0.0)
        return vconf, phz, pconf

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        wav_path, stem = self.items[idx]
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.squeeze()
        except Exception as e:
            raise OSError(f"Error loading audio file {wav_path}: {e!s}") from e

        if self.label_source == "author":
            pitch, periodicity = self._load_author_labels(wav_path, stem)
            # Resample the author f0 at its TRUE frame times (frame i at i*hop) so it lands on the
            # eval grid by time, not by index (which would warp the contour by ~half a frame).
            if self.AUTHOR_LABEL_HOP_SECONDS is None:
                raise NotImplementedError(
                    f"{self.NAME} ships author f0 but did not set AUTHOR_LABEL_HOP_SECONDS "
                    f"(the label hop in seconds); set it so the labels can be timed onto the grid."
                )
            label_times = (
                np.arange(pitch.numel()) * self.AUTHOR_LABEL_HOP_SECONDS
                + self.AUTHOR_LABEL_OFFSET_SECONDS
            )
            waveform, pitch, periodicity = self.process_sample(
                waveform, pitch, periodicity, sr, label_times=label_times
            )
            periodicity = energy_voicing_gate(
                waveform, periodicity, self.hop_size, self.silence_threshold
            )
            pitch = pitch * is_voiced(periodicity).to(pitch.dtype)   # re-apply invariant after the gate
            item = {"audio": waveform, "pitch": pitch, "periodicity": periodicity}
        else:  # consensus: precomputed confidence labels, grid-aligned, returned directly.
            waveform = self._prepare_audio(waveform, sr).squeeze(0)
            vconf, phz, pconf = self._load_consensus(stem, waveform.size(-1) // self.hop_size)
            # Enforce the shared label contract (out-of-range f0 -> unvoiced, then pitch=0 on
            # unvoiced). Consensus f0 is generated in-range, so the range gate is defensive parity.
            phz, vconf = self._apply_range_gate(phz, vconf)
            item = {
                "audio": waveform,
                "pitch": phz,                  # pitch_hz (median; a real value where voiced)
                "periodicity": vconf,          # voicing confidence in [0,1]; metric thresholds at 0.5
                "pitch_conf": pconf,           # RPA gates on this (>=tau); F1 ignores it
            }

        return {**item, "wav_path": wav_path}
