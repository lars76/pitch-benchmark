from pathlib import Path

from .base import LaryngographSpeechDataset


class PitchDatasetAPLAWD(LaryngographSpeechDataset):
    """APLAWDW: the APLAWD corpus (recorded at UCL 1987-88 for the SPAR project; Lindsey, Breen &
    Nevard) in M. Brookes's 2015 Imperial College repackaging -- 151 short utterances x 10 British-RP
    speakers (a-e male, f-j female), 20 kHz speech + simultaneous laryngograph (EGG). Read straight
    from the extracted ``APLAWDW/`` tree: each utterance is ``<...>/a<wtype><NN><spk><rep>.wav`` (speech) +
    ``.egg`` (laryngograph), both ordinary 20 kHz mono RIFF. Consensus f0 on the EGG is the label;
    calibration tones (word-type ``c``) are skipped. APLAWD ships no author f0, so this corpus is
    consensus-only. One item per utterance; group = speaker letter."""

    NAME = "APLAWD"
    fmin = 50.0
    fmax = 500.0

    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``(wav_path, stem)`` for each speech utterance under the extracted APLAWDW tree.
        ``base`` = ``a<wtype><NN><spk><rep>``: word-type is char 1, speaker letter is char 4 (this is
        the corpus's own filename convention). ``stem`` = ``<spk>_<base[1:]>`` so get_group's first
        "_" field is the speaker. Calibration tones (word-type ``c``) and the paired ``.egg`` missing
        are skipped."""
        root = Path(root)
        for wav in sorted(root.rglob("*.wav")):
            if "/doc/" in str(wav) or wav.name.startswith("._"):
                continue
            base = wav.stem
            if len(base) < 5 or base[1] == "c":          # calibration tone, not speech
                continue
            if not wav.with_suffix(".egg").exists():
                continue
            yield wav, f"{base[4]}_{base[1:]}"

    @classmethod
    def _read_speech(cls, wav_path):
        """Runtime speech-only: decode just the speech wav, skipping the .egg/EGG decode."""
        return cls._read_wav_mono(Path(wav_path))

    @classmethod
    def _read_original(cls, wav_path: Path):
        """Decode one utterance -> ``(speech, egg, sr)`` (native 20 kHz, mono float)."""
        speech, sr = cls._read_speech(wav_path)
        egg, _ = cls._read_wav_mono(Path(wav_path).with_suffix(".egg"))
        return speech, egg, sr
