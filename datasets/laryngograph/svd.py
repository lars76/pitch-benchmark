import csv
from pathlib import Path

import numpy as np

from .base import LaryngographSpeechDataset


def read_nsp(raw: bytes) -> tuple[np.ndarray, int]:
    """Parse a Kay/CSL FORMDS16 container (``.nsp`` speech / ``.egg`` laryngograph) into
    ``(float32 mono signal, sample_rate)``. Chunks are word-aligned; the sample rate lives in the
    HEDR chunk after a 20-byte date string, the samples in the SDA_ chunk as little-endian int16."""
    if raw[:8] != b"FORMDS16":
        raise ValueError("not an NSP FORMDS16 file")
    pos, sr, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        sz = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + sz]
        pos += 8 + sz + (sz & 1)                        # chunks are word-aligned
        if cid in (b"HEDR", b"HDR8"):
            sr = int.from_bytes(body[20:24], "little")
        elif cid in (b"SDA_", b"SD_A"):
            data = np.frombuffer(body[: sz - (sz % 2)], dtype="<i2")
    if data is None or not sr:
        raise ValueError("NSP missing SDA_/HEDR")
    if not 8000 <= sr <= 96000:                      # HEDR offset is fixed, not parsed: reject a garbage
        raise ValueError(f"NSP sample rate {sr} Hz out of range (HEDR parse suspect for this file)")
    return data.astype(np.float32) / 32768.0, int(sr)


class PitchDatasetSVD(LaryngographSpeechDataset):
    """Saarbruecken Voice Database: German speakers, each recording a connected-speech phrase with a
    simultaneous laryngograph (EGG). Read straight from the extracted zip tree: per recording
    ``<rec>/sentences/<rec>-phrase.nsp`` (speech) + ``<rec>/sentences/<rec>-phrase-egg.egg`` (EGG),
    both Kay FORMDS16 @ 50 kHz; ``overview.csv`` maps recording id -> speaker id. Consensus f0 on the
    EGG is the label; there is no shipped reference. One item per recording.

    Subset = which zip you extract, not code: SVD is a voice-DISORDER database (mostly pathological
    voices, whose irregular phonation makes unreliable f0 ground truth). Extracting ``healthy.zip``
    gives only the healthy control speakers -- the clean subset appropriate for an f0 benchmark, and
    the recommended ``--data-dir``. The reader simply enumerates whatever recording dirs (+ their
    ``overview.csv``) are under ``root``, so pointing it at an all-pathologies extraction would work
    too, but is not recommended.

    Grouping: stem is ``<speaker>_<rec>_phrase`` so get_group keeps a speaker's recordings together
    (via overview.csv's SprecherID); without that mapping the recording id would leak across folds.
    SVD ships no author f0, so this corpus is consensus-only."""

    NAME = "SVD"
    fmin = 50.0
    fmax = 500.0

    @classmethod
    def _speaker_map(cls, root: Path) -> dict:
        """recording id -> speaker id, from ``overview.csv`` (AufnahmeID -> SprecherID)."""
        ov = root / "overview.csv"
        if not ov.exists():
            return {}
        m = {}
        with open(ov, encoding="utf-8", errors="ignore", newline="") as f:
            for r in csv.DictReader(f):
                m[r["AufnahmeID"]] = r.get("SprecherID", r["AufnahmeID"])
        return m

    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``((nsp_path, egg_path), stem)`` for each recording's connected-speech phrase."""
        root = Path(root)
        spk_of = cls._speaker_map(root)
        for rec_dir in sorted(root.glob("*")):
            if not (rec_dir.is_dir() and rec_dir.name.isdigit()):
                continue
            rec = rec_dir.name
            nsp = rec_dir / "sentences" / f"{rec}-phrase.nsp"
            egg = rec_dir / "sentences" / f"{rec}-phrase-egg.egg"
            if nsp.exists() and egg.exists():
                spk = spk_of.get(rec, rec)
                yield (nsp, egg), f"{spk}_{rec}_phrase"

    @classmethod
    def _read_original(cls, loc):
        """Decode one recording -> ``(speech, egg, sr)`` (native 50 kHz, mono float)."""
        nsp, egg = loc
        speech, sr = read_nsp(Path(nsp).read_bytes())
        egg_sig, _ = read_nsp(Path(egg).read_bytes())
        m = min(len(speech), len(egg_sig))
        return speech[:m].astype(np.float64), egg_sig[:m].astype(np.float64), sr
