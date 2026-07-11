import gzip
import io
import json
import os
from pathlib import Path

import numpy as np

from .base import LaryngographSpeechDataset


def _condition_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """The OSF ``audio`` column carries a large DC offset (~1.96) that drowns the speech and clips a
    naive write. Remove DC, zero-phase high-pass at 80 Hz, peak-normalize -> usable speech."""
    import scipy.signal as ss
    x = audio.astype(np.float64) - float(np.mean(audio))
    sos = ss.butter(4, 80.0, "hp", fs=sr, output="sos")
    x = ss.sosfiltfilt(sos, x)
    return x / (np.max(np.abs(x)) + 1e-9)


class PitchDatasetOSFGlottis(LaryngographSpeechDataset):
    """OSF 'Physical Models of the Glottis' (osf.io/5yn2f): English subjects reading Harvard
    sentences with simultaneous EGG. Read straight from the extracted BIDS tree: per subject
    ``sub-XX/beh/sub-XX_task-speech_physio.tsv.gz`` with named columns ``[egg, audio, ...]`` @ 10 kHz
    (column names + rate in the ``_physio.json`` sidecar). Consensus f0 on the EGG is the label; no
    shipped reference. One item per subject. Subjects whose EGG is non-periodic simply yield few/no
    confident consensus frames (the 3-family agreement is self-protecting), so no separate drop list
    is needed at read time. OSF ships no author f0, so this corpus is consensus-only."""

    NAME = "OSFGlottis"
    fmin = 50.0
    fmax = 500.0

    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``((json_path, tsv_path), stem)`` per subject. ``stem`` = ``sub-XX``."""
        root = Path(root)
        for js in sorted(root.rglob("*_physio.json")):
            if js.name.startswith("._"):
                continue
            tsv = js.with_name(js.name[:-len("_physio.json")] + "_physio.tsv.gz")
            if tsv.exists():
                yield (js, tsv), js.name.split("_")[0]

    @classmethod
    def _read_original(cls, loc):
        """Decode one subject -> ``(speech, egg, sr)`` (native 10 kHz, mono float). The tsv.gz is
        large (~300 MB uncompressed) and np.loadtxt over it is slow; used by the offline consensus
        builder (which needs the EGG). The runtime loader uses the speech-only cache in _read_speech."""
        js, tsv = loc
        meta = json.loads(Path(js).read_text())
        cols, sr = meta["Columns"], int(meta["SamplingFrequency"])
        arr = np.loadtxt(io.BytesIO(gzip.decompress(Path(tsv).read_bytes())))
        egg = arr[:, cols.index("egg")].astype(np.float64)
        audio = arr[:, cols.index("audio")].astype(np.float64)
        return _condition_audio(audio, sr), egg, sr

    @classmethod
    def _read_speech(cls, loc):
        """Runtime speech-only. The ~300 MB tsv.gz is parsed at most ONCE per subject: the conditioned
        speech is cached as a float32 ``.speech.npy`` sidecar, so repeated benchmark passes (and every
        DataLoader worker) skip the gzip+loadtxt entirely. The consensus builder still uses the full
        _read_original (it needs the EGG)."""
        js, tsv = loc
        cache = Path(tsv).with_name(Path(tsv).name[: -len(".tsv.gz")] + ".speech.npy")
        sr = int(json.loads(Path(js).read_text())["SamplingFrequency"])
        if cache.exists():
            return np.load(cache), sr
        speech, _egg, sr = cls._read_original(loc)
        speech = speech.astype(np.float32)
        tmp = Path(str(cache) + ".tmp")
        np.save(tmp, speech)                             # writes str(tmp) + ".npy"
        os.replace(str(tmp) + ".npy", cache)             # atomic: no partial cache on crash
        return speech, sr
