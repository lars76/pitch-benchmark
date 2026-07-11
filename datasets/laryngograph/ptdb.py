from pathlib import Path

import numpy as np
import torch

from .base import LaryngographSpeechDataset


class PitchDatasetPTDB(LaryngographSpeechDataset):
    """Pitch Tracking Database (PTDB-TUG) speech. Read straight from the original download:
    ``{MALE,FEMALE}/MIC/mic_*.wav`` (close-talk microphone) paired with ``LAR/lar_*.wav`` (the
    laryngograph). Default ground truth is the committed cross-family consensus (``PTDB.npz``);
    ``label_source="reference"`` instead loads PTDB's shipped RAPT-on-laryngograph f0
    (``REF/ref_*.f0``, column 0), the dataset authors' own single-method reference.

    File-set policy (v2): every MIC/REF pair is included -- all utterances. Locally bad labels are
    handled per frame by the consensus semantics (a voiced-but-pitch-uncertain frame gets f0=0, which
    drops it from RPA while keeping it in voicing F1), so whole files are not discarded."""

    NAME = "PTDB"
    SUPPORTS_REFERENCE = True
    fmin = 65
    fmax = 300
    REFERENCE_LABEL_HOP_SECONDS = 0.01  # PTDB REF .f0 is a 10 ms-hop RAPT-on-laryngograph track
    # The REF track's frame i content sits ~21.7 ms AFTER the nominal i*10 ms stamp: half of RAPT's
    # 32 ms analysis window (16 ms) plus RAPT's forward NCCF offset (~6 ms, independently measured on
    # our RAPT wrapper by tests/test_time_calibration.py). Measured by the label-offset sweep with
    # three calibrated reference trackers agreeing within 1 ms (scripts/check_dataset_alignment.py).
    REFERENCE_LABEL_OFFSET_SECONDS = 0.0217

    @staticmethod
    def _lar_of(mic_wav: Path) -> Path:
        return Path(str(mic_wav).replace("/MIC/", "/LAR/")).with_name(
            mic_wav.name.replace("mic_", "lar_")
        )

    @staticmethod
    def _ref_of(mic_wav: Path) -> Path:
        return Path(str(mic_wav).replace("/MIC/", "/REF/")).with_name(
            mic_wav.name.replace("mic_", "ref_").replace(".wav", ".f0")
        )

    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``(mic_wav, stem)`` for each MIC utterance that has a REF f0 (the reference gate is
        kept so both label modes share one item set)."""
        root = Path(root)
        for gender in ("MALE", "FEMALE"):
            mic_dir = root / gender / "MIC"
            if not mic_dir.exists():
                continue
            for wav in sorted(mic_dir.rglob("*.wav")):
                if cls._ref_of(wav).exists():
                    yield wav, wav.stem

    @classmethod
    def _read_speech(cls, mic_wav):
        """Runtime speech-only: decode just the MIC wav, skipping the LAR/EGG decode."""
        return cls._read_wav_mono(Path(mic_wav))

    @classmethod
    def _read_original(cls, mic_wav):
        """Decode one utterance -> ``(mic_speech, lar_egg, sr)`` (native rate, mono float)."""
        mic_wav = Path(mic_wav)
        speech, sr = cls._read_speech(mic_wav)
        lar = cls._lar_of(mic_wav)
        egg, _ = cls._read_wav_mono(lar) if lar.exists() else (None, sr)
        return speech, egg, sr

    def get_group(self, idx: int) -> str:
        # PTDB stem is "mic_<speaker>_..." (e.g. "mic_M01_sa1"), so the speaker is the 2nd field.
        parts = self.items[idx][1].split("_")
        return parts[1] if len(parts) >= 2 else "unknown"

    def _load_reference_labels(self, mic_wav, stem):
        # The REF .f0 track carries no per-frame times, so return None and let the base reconstruct
        # them from REFERENCE_LABEL_HOP_SECONDS (10 ms) + the calibrated REFERENCE_LABEL_OFFSET.
        ref = self._ref_of(Path(mic_wav))
        try:
            pitch = torch.from_numpy(np.loadtxt(ref)[:, 0]).float()
        except Exception as e:
            raise OSError(f"Error loading F0 file {ref}: {e!s}") from e
        return pitch, (pitch > 0).float(), None
