from pathlib import Path

import numpy as np
import torch

from .base import LaryngographSpeechDataset


class PitchDatasetKEELE(LaryngographSpeechDataset):
    """KEELE pitch reference database: 10 speakers reading the North Wind passage, with a
    hand-corrected laryngograph f0 reference. Scored against the cross-family ``consensus`` (built
    from ``<stem>/laryngograph.wav``) by default, like every EGG corpus; ``label_source="reference"``
    instead loads KEELE's shipped hand-corrected f0 (the very gold the consensus recipe was validated
    against). Group = speaker (one passage per speaker).

    SOURCE (exception to the raw-download rule): the first-party Keele/SAM distribution is defunct
    (the KTH/MIT ``lost-contact`` mirror is offline and unarchived), so this loader reads KEELE from
    Bechtold's Zenodo redistribution (record 3921794, dissertation replication set), which repackages
    the corpus as a jbof dataframe while preserving the original 20 kHz / 16-bit audio unchanged. The
    per-item tree is therefore ``<stem>/signal.wav`` (speech), ``<stem>/laryngograph.wav`` (EGG), and
    ``<stem>/pitch.npy`` (the reference, a 10 ms ``(time, pitch)`` structured grid; NEGATIVE values
    mark uncertain frames). This is the one EGG corpus whose canonical form is a curated packaging
    rather than a first-party raw download -- see the README KEELE entry."""

    NAME = "KEELE"
    SUPPORTS_REFERENCE = True
    fmin = 50.0
    fmax = 500.0
    REFERENCE_LABEL_HOP_SECONDS = 0.01       # fallback; pitch.npy ships its own 10 ms time grid
    # Label-offset sweep (scripts/check_dataset_alignment.py, 10 files x 30 s):
    # Praat +2.80 / DIO +1.28 / SWIPE +2.58 ms -> consensus +2.58 ms (labels stamped early).
    REFERENCE_LABEL_OFFSET_SECONDS = 0.0026

    @classmethod
    def _iter_originals(cls, root: Path):
        root = Path(root)
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "signal.wav").exists() and (d / "pitch.npy").exists():
                yield d, d.name

    @classmethod
    def _read_speech(cls, item_dir):
        """Runtime speech-only: decode just signal.wav, skipping the laryngograph.wav/EGG decode."""
        return cls._read_wav_mono(Path(item_dir) / "signal.wav")

    @classmethod
    def _read_original(cls, item_dir):
        item_dir = Path(item_dir)
        speech, sr = cls._read_speech(item_dir)
        lar = item_dir / "laryngograph.wav"
        egg = cls._read_wav_mono(lar)[0] if lar.exists() else None
        return speech, egg, sr

    def _load_reference_labels(self, item_dir, stem):
        # KEELE marks frames where the laryngograph gave no reliable period with NEGATIVE sentinels
        # (e.g. -20000). Map them to voiced-but-pitch-uncertain (pitch 0, periodicity 1): F1 counts
        # them positive and the metric's finite-frame rule drops them from RPA -- the same three-state
        # semantics as the consensus labels. Scoring them as confident unvoiced would penalize
        # trackers on frames the corpus itself declines to judge.
        arr = np.load(Path(item_dir) / "pitch.npy")
        f0 = np.nan_to_num(np.asarray(arr["pitch"], dtype=np.float32), nan=0.0)
        uncertain = torch.from_numpy(f0 < 0)
        pitch = torch.from_numpy(f0).clamp(min=0.0).float()
        periodicity = ((pitch > 0) | uncertain).float()
        return pitch, periodicity, np.asarray(arr["time"], dtype=np.float64)
