from pathlib import Path

import numpy as np
import soundfile as sf

from .base import LaryngographSpeechDataset


def _hf_energy_frac(sig: np.ndarray, sr: int, cut: float = 2000.0) -> float:
    """Fraction of spectral energy above ``cut`` Hz. Speech (fricatives/formants) carries far more
    than the band-limited EGG, so this cheaply discriminates the two channels (50-200x separation)
    with no pitch-tracker / pyworld dependency (which must stay out of the runtime path)."""
    x = sig * np.hanning(len(sig))
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(sig), 1.0 / sr)
    return float(X[f > cut].sum() / (X.sum() + 1e-12))


class PitchDatasetCMUArctic(LaryngographSpeechDataset):
    """CMU Arctic (festvox), EGG-recorded. Read straight from the extracted ``-WAVEGG`` download:
    ``cmu_us_<spk>_arctic/orig/*.wav`` are 32 kHz STEREO (one speech channel + one laryngograph/EGG
    channel). The EGG channel is detected per file by its lower high-frequency energy. Consensus f0
    (Praat / differentiated-EGG / Harvest on the EGG) is the label; CMU ships no author f0, so this
    corpus is consensus-only."""

    NAME = "CMUArctic"
    fmin = 50.0
    fmax = 500.0

    @classmethod
    def _iter_originals(cls, root: Path):
        """Yield ``(wav_path, stem)`` for every WAVEGG utterance. ``stem`` = ``<spk>_<uttid>``
        (e.g. ``bdl_arctic_a0001``), which keys the cache, the consensus npz, and get_group."""
        root = Path(root)
        for spk_dir in sorted(root.glob("cmu_us_*_arctic")):
            spk = spk_dir.name.split("_")[2]           # cmu_us_bdl_arctic -> bdl
            for wav in sorted((spk_dir / "orig").glob("*.wav")):
                if wav.name.startswith("._"):
                    continue
                yield wav, f"{spk}_{wav.stem}"

    @classmethod
    def _read_original(cls, wav_path: Path):
        """Decode one WAVEGG utterance -> ``(speech, egg, sr)`` (native rate, mono float)."""
        stereo, sr = sf.read(str(wav_path))
        stereo = np.asarray(stereo, dtype=np.float64)
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError(f"CMUArctic expects stereo WAVEGG, got shape {stereo.shape} for {wav_path}")
        egg_ch = 0 if _hf_energy_frac(stereo[:, 0], sr) < _hf_energy_frac(stereo[:, 1], sr) else 1
        return stereo[:, 1 - egg_ch], stereo[:, egg_ch], sr
