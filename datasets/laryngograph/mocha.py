from pathlib import Path

from .base import LaryngographSpeechDataset

# Stray non-speaker item shipped in some MOCHA distributions (near-silent, not a real speaker).
_DROP = {"tbar"}


class PitchDatasetMOCHA(LaryngographSpeechDataset):
    """MOCHA-TIMIT (CSTR): 2->10 speakers x 460 TIMIT sentences with a simultaneous laryngograph
    (EGG). Consensus f0 on the EGG is the label; there is no shipped reference. One item per
    utterance; group = speaker.

    Read straight from the raw CSTR download: each utterance is a flat ``<spk>_<num>.wav`` (speech) +
    ``<spk>_<num>.lar`` (EGG) pair, both NIST/SPHERE @ 16 kHz (the per-speaker ``<spk>.tar.gz`` archives
    from data.cstr.ed.ac.uk/mocha; the articulography ``.ema``/``.epg`` files alongside are ignored).
    MOCHA ships no author f0, so this corpus is consensus-only."""

    NAME = "MOCHA"
    fmin = 50.0
    fmax = 500.0

    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``(wav_path, stem)`` for each ``<spk>_<num>.wav`` that has a ``.lar`` EGG sibling."""
        root = Path(root)
        for wav in sorted(root.rglob("*.wav")):
            if wav.name.startswith("._") or not wav.with_suffix(".lar").exists():
                continue
            if wav.stem.split("_")[0] not in _DROP:
                yield wav, wav.stem

    @classmethod
    def _read_speech(cls, wav_path):
        """Runtime speech-only: decode just the speech wav, skipping the .lar/EGG decode."""
        return cls._read_wav_mono(Path(wav_path))

    @classmethod
    def _read_original(cls, wav_path):
        """Decode one utterance -> ``(speech, egg, sr)`` (native rate, mono float)."""
        wav_path = Path(wav_path)
        speech, sr = cls._read_speech(wav_path)
        egg, _ = cls._read_wav_mono(wav_path.with_suffix(".lar"))
        return speech, egg, sr
