import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import LaryngographSpeechDataset


class PitchDatasetAVID(LaryngographSpeechDataset):
    """AVID (Aalto Vocal Intensity Database): 50 speakers, ~15 min calibrated recordings in
    ``Repository 1`` as 44.1 kHz STEREO (ch0 = speech, ch1 = laryngograph/EGG; a fixed layout, so we
    do not channel-detect here -- the HF-energy heuristic is unreliable on AVID and DIO detection
    would pull in pyworld, an install extra that must stay out of the runtime). Consensus f0 on the
    EGG is the label; AVID ships no author f0, so this corpus is consensus-only. One item per recording
    (most speakers have one; a few have several, keyed by unique stem and grouped by speaker for CV)."""

    NAME = "AVID"
    fmin = 50.0
    fmax = 500.0
    SPEECH_CH, EGG_CH = 0, 1

    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``(wav_path, stem)`` for each ``Repository 1/Spk*_*.wav``. ``stem`` = the file stem
        ``SpkN_DATE`` -- unique per recording (a few speakers have >1 recording), while get_group still
        derives the speaker id ``SpkN`` from the first ``_``-field."""
        root = Path(root)
        for wav in sorted(root.rglob("Spk*_*.wav")):
            if wav.name.startswith("._") or "__MACOSX" in str(wav) or "Repository 1" not in str(wav):
                continue
            yield wav, wav.stem

    @classmethod
    def _read_original(cls, wav_path: Path):
        """Decode one AVID recording -> ``(speech, egg, sr)`` (native 44.1 kHz, mono float)."""
        stereo, sr = sf.read(str(wav_path))
        stereo = np.asarray(stereo, dtype=np.float64)
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError(f"AVID expects stereo, got shape {stereo.shape} for {wav_path}")
        speech, egg = stereo[:, cls.SPEECH_CH], stereo[:, cls.EGG_CH]
        # Cheap sanity check on a mid slice: the EGG (band-limited glottal signal) crosses zero far
        # less often than speech. Warn (do not fail) if the fixed layout looks inverted.
        s = slice(len(speech) // 3, len(speech) // 3 + min(len(speech), sr * 5))
        zc = lambda x: np.mean(np.abs(np.diff(np.sign(x[s]))) > 0)
        if zc(egg) > zc(speech):
            warnings.warn(f"AVID {wav_path.name}: EGG channel looks inverted (zcr egg>speech).",
                          stacklevel=2)
        return speech, egg, sr
