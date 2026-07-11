"""Base class for laryngograph speech corpora.

These corpora are recorded with an electroglottograph (EGG, a laryngograph). Audio is ALWAYS decoded
straight from the original (extracted) dataset download -- each concrete subclass implements the
per-corpus format in ``_iter_originals`` (enumerate items) + ``_read_original`` (decode one item to
native-rate mono speech + EGG). The SAME reader is reused by scripts/build_consensus_labels.py, so a
corpus's on-disk format lives in exactly one place.

Ground truth is selected by ``label_source``:
  - ``consensus`` (default): the precomputed ``<NAME>.npz`` beside this module -- consensus over three
    independent EGG estimators (Praat / differentiated-EGG / Harvest), keyed by file stem, a (3, n) float32 array of
    (voicing_conf, pitch_hz, pitch_conf) per frame. Built once by scripts/build_consensus_labels.py
    and committed. No EGG signal or f0 estimator is needed at benchmark time -- only the audio.
  - ``reference``: the DATASET AUTHORS' shipped single-method f0, read from the archive. Only corpora
    whose download actually ships an f0 file support this (``SUPPORTS_REFERENCE = True``): PTDB
    (``REF/.f0``), KEELE (its reference track), FDA (``.fx``). The other corpora ship audio + EGG but
    no f0, so they are consensus-only (there is nothing to read for reference).

Per-frame label semantics (the three states, expressed via the two arrays):
  voiced, pitch-confident   -> (f0 > 0, periodicity = 1)   scored for both F1 and RPA
  voiced, pitch-uncertain   -> (f0 = 0, periodicity = 1)   F1 positive; the metric's finite-frame
                                                           rule drops f0 = 0 from RPA automatically
  unvoiced                  -> (f0 = 0, periodicity = 0)   F1 negative
"""
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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
    """Base for laryngograph speech corpora. A concrete subclass sets ``NAME`` (the npz key) and its
    pitch range, and implements the original-download reader (``_iter_originals`` + ``_read_original``);
    everything else is inherited. See the module docstring for the two label sources.

    A subclass whose download ships an author f0 sets ``SUPPORTS_REFERENCE = True`` and implements
    ``_load_reference_labels``; the rest are consensus-only."""

    NAME: str = None  # npz key / label file basename, e.g. "PTDB"
    DEFAULT_LABEL_SOURCE: str = "consensus"
    # Whether the dataset's download ships an author f0 (-> ``reference`` mode is available).
    SUPPORTS_REFERENCE: bool = False
    # FALLBACK native hop (seconds) for a reference that carries no per-frame times (e.g. PTDB's REF
    # .f0). A reference whose file ships a `time` column never uses it (the shipped times win).
    REFERENCE_LABEL_HOP_SECONDS: float = None
    # Constant timing correction (seconds) ADDED to the reference label times: the time of the
    # analysis content RELATIVE to the label's nominal stamp (the reference tracker's intra-window
    # offset). Measured per corpus by scripts/check_dataset_alignment.py; see TIMING.md.
    REFERENCE_LABEL_OFFSET_SECONDS: float = 0.0

    def __init__(
        self,
        root_dir: str,
        label_source: str | None = None,
        use_cache: bool = True,
        silence_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(use_cache=use_cache, **kwargs)
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")
        label_source = label_source or self.DEFAULT_LABEL_SOURCE
        if label_source not in ("consensus", "reference"):
            raise ValueError(
                f"label_source must be 'consensus' or 'reference', got '{label_source}'"
            )
        if label_source == "reference" and not self.SUPPORTS_REFERENCE:
            raise ValueError(
                f"{self.NAME} ships no author f0, so label_source='reference' is unavailable "
                f"(this corpus is consensus-only; its ground truth is the committed {self.NAME}.npz)."
            )
        self.label_source = label_source
        self.silence_threshold = silence_threshold

        self._consensus = None  # lazily-read {stem: array} dict (per process; DataLoader-worker safe)

        # [(locator, stem), ...] -- locator is whatever the subclass reader needs (a Path, or a
        # small tuple for multi-file corpora); stem keys the cache, the consensus npz, and get_group.
        self.items: list[tuple] = list(self._discover())
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

    # ----- per-corpus original-download reader (implemented by every subclass) ------------------
    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``(locator, stem)`` for every item in the extracted original download."""
        raise NotImplementedError(f"{cls.__name__} must implement _iter_originals")

    @classmethod
    def _read_original(cls, locator):
        """Decode one item -> ``(speech, egg, sr)`` (native-rate mono float; egg may be None). The
        EGG is used only by the offline consensus builder; the runtime loader keeps only speech."""
        raise NotImplementedError(f"{cls.__name__} must implement _read_original")

    @classmethod
    def _read_speech(cls, locator):
        """Decode ONLY the speech -> ``(speech, sr)`` for the runtime loader, skipping the EGG. The
        default reads the full item and drops the EGG (correct for stereo/single-file corpora where
        the EGG comes for free); a corpus whose EGG is a SEPARATE file overrides this to avoid the
        extra decode (the EGG is never needed at benchmark time -- only the consensus builder reads it)."""
        speech, _egg, sr = cls._read_original(locator)
        return speech, sr

    def _load_reference_labels(self, locator, stem):
        """Return ``(pitch, periodicity, label_times|None)`` from the dataset's shipped author f0.
        Only SUPPORTS_REFERENCE corpora implement this."""
        raise NotImplementedError(f"{self.NAME} does not ship an author reference")

    # ----- core --------------------------------------------------------------
    def _discover(self):
        return self._iter_originals(self.root_dir)

    @staticmethod
    def _loc_path(locator) -> Path:
        """A Path for logging (``wav_path``), whatever shape the locator is."""
        return Path(locator[0]) if isinstance(locator, (tuple, list)) else Path(locator)

    @staticmethod
    def _read_wav_mono(path) -> tuple[np.ndarray, int]:
        """Decode a WAV/container to native-rate mono float64, averaging any channels down. This is
        the single audio decode policy shared by every subclass reader, so a corpus's committed
        consensus npz and its runtime speech are always built from identical samples (channel-averaged
        float64); a change here can never leave one reader on a different decode than the others."""
        data, sr = sf.read(str(path), dtype="float64")
        return (data.mean(1) if data.ndim > 1 else data), sr

    def _consensus_npz(self):
        """Lazily read the dataset's consensus label file (one .npz keyed by file stem) into a
        {stem: array} dict. Read EAGERLY inside a context manager so the .npz file descriptor is
        closed immediately: np.load on a zip archive otherwise returns a lazy NpzFile that keeps the
        fd open for the object's lifetime, one per dataset instance AND per DataLoader worker, which
        accumulates across k-fold CV. The label arrays are tiny ((3, n) float32 per clip), so holding
        them resident costs far less than leaking a handle."""
        if self._consensus is None:
            p = CONSENSUS_DIR / f"{self.NAME}.npz"
            if not p.exists():
                raise FileNotFoundError(
                    f"Consensus labels missing: {p}. Run "
                    f"scripts/build_consensus_labels.py --dataset {self.NAME}."
                )
            with np.load(p) as z:
                self._consensus = {stem: z[stem] for stem in z.files}
        return self._consensus

    def __len__(self) -> int:
        return len(self.items)

    def get_group(self, idx: int) -> str:
        # Default: speaker id is the first "_"-delimited field of the stem (a stem with no "_" is one
        # speaker, so the whole stem groups). PTDB overrides. Each reader builds the stem so that this
        # yields the right speaker id (e.g. SVD encodes the SprecherID as the first field).
        return self.items[idx][1].split("_")[0]

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
        if stem not in d:
            raise KeyError(
                f"No consensus label for '{stem}' in {self.NAME}.npz "
                f"(regenerate with scripts/build_consensus_labels.py --dataset {self.NAME})."
            )
        arr = np.asarray(d[stem], dtype=np.float32)   # (3, n): voicing_conf, pitch_hz, pitch_conf
        vconf = self._fit(torch.from_numpy(arr[0]), target_length, 0.0)
        phz = self._fit(torch.from_numpy(arr[1]), target_length, 0.0)
        pconf = self._fit(torch.from_numpy(arr[2]), target_length, 0.0)
        return vconf, phz, pconf

    def _consensus_item(
        self, waveform: torch.Tensor, orig_sr: int, stem: str
    ) -> dict[str, torch.Tensor]:
        """Assemble the consensus-mode item (audio, pitch, periodicity, pitch_conf) from decoded
        speech: prepare the audio onto the eval grid, read the precomputed grid-aligned confidence
        labels, and apply the defensive range gate (consensus f0 is generated in-range, so this is
        parity, not expected to fire)."""
        waveform = self._prepare_audio(waveform, orig_sr).squeeze(0)
        vconf, phz, pconf = self._load_consensus(stem, waveform.size(-1) // self.hop_size)
        phz, vconf = self._apply_range_gate(phz, vconf)
        return {
            "audio": waveform,
            "pitch": phz,                  # pitch_hz (median; a real value where voiced)
            "periodicity": vconf,          # voicing confidence in [0,1]; metric thresholds at 0.5
            "pitch_conf": pconf,           # RPA gates on this (>=tau); F1 ignores it
        }

    def _reference_item(
        self, waveform: torch.Tensor, orig_sr: int, locator, stem: str
    ) -> dict[str, torch.Tensor]:
        """Assemble the reference-mode item from decoded speech + the shipped author f0. The labels
        are resampled onto the eval grid at their TRUE frame times (shipped `time` column preferred;
        else reconstructed from REFERENCE_LABEL_HOP_SECONDS), then the mic-energy gate is applied."""
        pitch, periodicity, times = self._load_reference_labels(locator, stem)
        if times is not None:
            label_times = np.asarray(times, dtype=np.float64) + self.REFERENCE_LABEL_OFFSET_SECONDS
        elif self.REFERENCE_LABEL_HOP_SECONDS is not None:
            label_times = (
                np.arange(pitch.numel()) * self.REFERENCE_LABEL_HOP_SECONDS
                + self.REFERENCE_LABEL_OFFSET_SECONDS
            )
        else:
            raise NotImplementedError(
                f"{self.NAME} reference has no per-frame times and no REFERENCE_LABEL_HOP_SECONDS."
            )
        waveform, pitch, periodicity = self.process_sample(
            waveform, pitch, periodicity, orig_sr, label_times=label_times
        )
        periodicity = energy_voicing_gate(
            waveform, periodicity, self.hop_size, self.silence_threshold
        )
        pitch = pitch * is_voiced(periodicity).to(pitch.dtype)   # re-apply invariant after the gate
        return {"audio": waveform, "pitch": pitch, "periodicity": periodicity}

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        locator, stem = self.items[idx]
        try:
            speech, sr = self._read_speech(locator)   # speech only; EGG is not needed at runtime
        except Exception as e:
            raise OSError(f"Error decoding {self._loc_path(locator)}: {e!s}") from e
        waveform = torch.from_numpy(np.ascontiguousarray(speech, dtype=np.float32))

        if self.label_source == "reference":
            item = self._reference_item(waveform, sr, locator, stem)
        else:
            item = self._consensus_item(waveform, sr, stem)

        return {**item, "wav_path": self._loc_path(locator)}
