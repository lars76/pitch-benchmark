import csv
import gzip
import io
import json
import os
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from constants import CONSENSUS, CORPUS_FMIN, HOP_SIZE, POWER_FLOOR, RMS_FLOOR, SAMPLE_RATE, SPEECH_FMAX

from .base import PitchDataset


class LaryngographSpeechDataset(PitchDataset):

    NAME = None
    fmin = CORPUS_FMIN
    fmax = SPEECH_FMAX
    GATE = "peak"
    GATE_FLOOR_K = 2.5
    GATE_FLOOR_QUANTILE = 0.05
    SKIP_LEAD_SECONDS = 0.0

    def __init__(self, root_dir, consensus_dir=CONSENSUS, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        npz_path = Path(consensus_dir) / f"{self.NAME}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Consensus labels missing: {npz_path}. Run "
                f"python -m scripts.build_consensus_labels --dataset {self.NAME}."
            )
        with np.load(npz_path) as z:
            self._consensus = {stem: z[stem] for stem in z.files}

        discovered = list(self._iter_originals(self.root_dir))
        if not discovered:
            raise ValueError(f"No audio files found for {self.NAME} in '{root_dir}'")
        self.items = [it for it in discovered if it[1] in self._consensus]
        if not self.items:
            raise ValueError(
                f"None of the {len(discovered)} discovered {self.NAME} clips have a consensus label "
                f"in {npz_path.name}. Run python -m scripts.build_consensus_labels --dataset {self.NAME}."
            )
        if len(self.items) < len(discovered):
            warnings.warn(
                f"{self.NAME}: evaluating on {len(self.items)} of {len(discovered)} discovered clips "
                f"({len(self._consensus)} stems have consensus labels); skipping "
                f"{len(discovered) - len(self.items)} unlabelled. Regenerate with "
                f"python -m scripts.build_consensus_labels --dataset {self.NAME} to cover all.",
                stacklevel=2,
            )
        if self.sample_rate != SAMPLE_RATE or self.hop_size != HOP_SIZE:
            warnings.warn(
                f"{self.NAME} consensus labels were generated at {SAMPLE_RATE} Hz / hop {HOP_SIZE}; running at "
                f"{self.sample_rate} Hz / hop {self.hop_size} may misalign labels.",
                stacklevel=2,
            )

    @classmethod
    def _iter_originals(cls, root):
        raise NotImplementedError(f"{cls.__name__} must implement _iter_originals")

    @classmethod
    def _read_original(cls, locator):
        raise NotImplementedError(f"{cls.__name__} must implement _read_original")

    @classmethod
    def _read_speech(cls, locator):
        speech, _egg, sr = cls._read_original(locator)
        return speech, sr

    @staticmethod
    def _loc_path(locator):
        return Path(locator[0]) if isinstance(locator, (tuple, list)) else Path(locator)

    @staticmethod
    def _read_wav_mono(path):
        data, sr = sf.read(str(path), dtype="float64")
        return (data.mean(1) if data.ndim > 1 else data), sr

    def __len__(self):
        return len(self.items)

    def get_group(self, idx):
        return self.items[idx][1].split("_")[0]

    def item_grid_frames(self, idx):
        return int(self._consensus[self.items[idx][1]].shape[1]) - 1

    @staticmethod
    def _crop_pad(x, n, pad):
        m = x.numel()
        if m == n:
            return x
        if m > n:
            return x[:n]
        out = torch.full((n,), pad, dtype=x.dtype)
        out[:m] = x
        return out

    def _load_sample(self, idx):
        locator, stem = self.items[idx]
        try:
            speech, sr = self._read_speech(locator)
        except Exception as e:
            raise OSError(f"Error decoding {self._loc_path(locator)}: {e!s}") from e
        waveform = torch.from_numpy(np.ascontiguousarray(speech, dtype=np.float32))
        waveform = self._prepare_audio(waveform, sr).squeeze(0)
        n = waveform.size(-1) // self.hop_size
        arr = np.asarray(self._consensus[stem], dtype=np.float32)
        vconf = self._crop_pad(torch.from_numpy(arr[0]), n, 0.0)
        phz = self._crop_pad(torch.from_numpy(arr[1]), n, 0.0)
        pconf = self._crop_pad(torch.from_numpy(arr[2]), n, 0.0)
        phz, vconf = self._enforce_voicing_invariant(phz, vconf)
        return {
            "audio": waveform,
            "pitch": phz,
            "periodicity": vconf,
            "pitch_conf": pconf,
            "wav_path": self._loc_path(locator),
        }


class PitchDatasetAPLAWD(LaryngographSpeechDataset):

    NAME = "APLAWD"

    @classmethod
    def _iter_originals(cls, root):
        root = Path(root)
        for wav in sorted(root.rglob("*.wav")):
            if "/doc/" in str(wav) or wav.name.startswith("._"):
                continue
            base = wav.stem
            if len(base) < 5 or base[1] == "c":
                continue
            if not wav.with_suffix(".egg").exists():
                continue
            yield wav, f"{base[4]}_{base[1:]}"

    @classmethod
    def _read_speech(cls, wav_path):
        return cls._read_wav_mono(Path(wav_path))

    @classmethod
    def _read_original(cls, wav_path):
        speech, sr = cls._read_speech(wav_path)
        egg, _ = cls._read_wav_mono(Path(wav_path).with_suffix(".egg"))
        return speech, egg, sr


class PitchDatasetAVID(LaryngographSpeechDataset):

    NAME = "AVID"
    GATE = "floor"
    SKIP_LEAD_SECONDS = 8.0
    SPEECH_CH, EGG_CH = 0, 1

    @classmethod
    def _iter_originals(cls, root):
        root = Path(root)
        for wav in sorted(root.rglob("Spk*_*.wav")):
            if wav.name.startswith("._") or "__MACOSX" in str(wav) or "Repository 1" not in str(wav):
                continue
            yield wav, wav.stem

    @classmethod
    def _read_original(cls, wav_path):
        stereo, sr = sf.read(str(wav_path))
        stereo = np.asarray(stereo, dtype=np.float64)
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError(f"AVID expects stereo, got shape {stereo.shape} for {wav_path}")
        speech, egg = stereo[:, cls.SPEECH_CH], stereo[:, cls.EGG_CH]
        s = slice(len(speech) // 3, len(speech) // 3 + min(len(speech), sr * 5))

        def zc(x):
            return np.mean(np.abs(np.diff(np.sign(x[s]))) > 0)

        if zc(egg) > zc(speech):
            warnings.warn(f"AVID {wav_path.name}: EGG channel looks inverted (zcr egg>speech).",
                          stacklevel=2)
        return speech, egg, sr


def _hf_energy_frac(sig, sr, cut=2000.0):
    x = sig * np.hanning(len(sig))
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(sig), 1.0 / sr)
    return float(X[f > cut].sum() / (X.sum() + POWER_FLOOR))


class PitchDatasetCMUArctic(LaryngographSpeechDataset):

    NAME = "CMUArctic"

    @classmethod
    def _iter_originals(cls, root):
        root = Path(root)
        for spk_dir in sorted(root.glob("cmu_us_*_arctic")):
            spk = spk_dir.name.split("_")[2]
            for wav in sorted((spk_dir / "orig").glob("*.wav")):
                if wav.name.startswith("._"):
                    continue
                yield wav, f"{spk}_{wav.stem}"

    @classmethod
    def _read_original(cls, wav_path):
        stereo, sr = sf.read(str(wav_path))
        stereo = np.asarray(stereo, dtype=np.float64)
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError(f"CMUArctic expects stereo WAVEGG, got shape {stereo.shape} for {wav_path}")
        egg_ch = 0 if _hf_energy_frac(stereo[:, 0], sr) < _hf_energy_frac(stereo[:, 1], sr) else 1
        return stereo[:, 1 - egg_ch], stereo[:, egg_ch], sr


def _read_raw_i16(path):
    return np.fromfile(str(path), dtype=">i2").astype(np.float64) / 32768.0


class PitchDatasetFDA(LaryngographSpeechDataset):
    NAME = "FDA"
    SR = 20000

    @classmethod
    def _iter_originals(cls, root):
        root = Path(root)
        for spk in ("rl", "sb"):
            d = root / spk
            if not d.is_dir():
                continue
            for sig in sorted(d.glob("*.sig")):
                if sig.with_suffix(".fx").exists():
                    yield sig, sig.stem

    @classmethod
    def _read_speech(cls, sig_path):
        return _read_raw_i16(Path(sig_path)), cls.SR

    @classmethod
    def _read_original(cls, sig_path):
        sig_path = Path(sig_path)
        speech, sr = cls._read_speech(sig_path)
        lar = sig_path.with_suffix(".lar")
        egg = _read_raw_i16(lar) if lar.exists() else None
        return speech, egg, sr

    def get_group(self, idx):
        return self._loc_path(self.items[idx][0]).parent.name


class PitchDatasetKEELE(LaryngographSpeechDataset):

    NAME = "KEELE"

    @classmethod
    def _iter_originals(cls, root):
        root = Path(root)
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "signal.wav").exists() and (d / "pitch.npy").exists():
                yield d, d.name

    @classmethod
    def _read_speech(cls, item_dir):
        return cls._read_wav_mono(Path(item_dir) / "signal.wav")

    @classmethod
    def _read_original(cls, item_dir):
        item_dir = Path(item_dir)
        speech, sr = cls._read_speech(item_dir)
        lar = item_dir / "laryngograph.wav"
        egg = cls._read_wav_mono(lar)[0] if lar.exists() else None
        return speech, egg, sr


class PitchDatasetMOCHA(LaryngographSpeechDataset):

    NAME = "MOCHA"

    @classmethod
    def _iter_originals(cls, root):
        root = Path(root)
        for wav in sorted(root.rglob("*.wav")):
            if wav.name.startswith("._") or not wav.with_suffix(".lar").exists():
                continue
            if wav.stem.split("_")[0] not in {'tbar'}:
                yield wav, wav.stem

    @classmethod
    def _read_speech(cls, wav_path):
        return cls._read_wav_mono(Path(wav_path))

    @classmethod
    def _read_original(cls, wav_path):
        wav_path = Path(wav_path)
        speech, sr = cls._read_speech(wav_path)
        egg, _ = cls._read_wav_mono(wav_path.with_suffix(".lar"))
        return speech, egg, sr


def _condition_audio(audio, sr):
    import scipy.signal as ss
    x = audio.astype(np.float64) - float(np.mean(audio))
    sos = ss.butter(4, 80.0, "hp", fs=sr, output="sos")
    x = ss.sosfiltfilt(sos, x)
    return x / (np.max(np.abs(x)) + RMS_FLOOR)


class PitchDatasetOSFGlottis(LaryngographSpeechDataset):

    NAME = "OSFGlottis"
    GATE = "floor"

    @classmethod
    def _iter_originals(cls, root):
        root = Path(root)
        for js in sorted(root.rglob("*_physio.json")):
            if js.name.startswith("._"):
                continue
            tsv = js.with_name(js.name[:-len("_physio.json")] + "_physio.tsv.gz")
            if tsv.exists():
                yield (js, tsv), js.name.split("_")[0]

    @classmethod
    def _read_original(cls, loc):
        js, tsv = loc
        meta = json.loads(Path(js).read_text())
        cols, sr = meta["Columns"], int(meta["SamplingFrequency"])
        arr = np.loadtxt(io.BytesIO(gzip.decompress(Path(tsv).read_bytes())))
        egg = arr[:, cols.index("egg")].astype(np.float64)
        audio = arr[:, cols.index("audio")].astype(np.float64)
        return _condition_audio(audio, sr), egg, sr

    @classmethod
    def _read_speech(cls, loc):
        js, tsv = loc
        cache = Path(tsv).with_name(Path(tsv).name[: -len(".tsv.gz")] + ".speech.npy")
        sr = int(json.loads(Path(js).read_text())["SamplingFrequency"])
        if cache.exists():
            return np.load(cache), sr
        speech, _egg, sr = cls._read_original(loc)
        speech = speech.astype(np.float32)
        tmp = Path(str(cache) + ".tmp")
        np.save(tmp, speech)
        os.replace(str(tmp) + ".npy", cache)
        return speech, sr


class PitchDatasetPTDB(LaryngographSpeechDataset):

    NAME = "PTDB"

    @staticmethod
    def _lar_of(mic_wav):
        return Path(str(mic_wav).replace("/MIC/", "/LAR/")).with_name(
            mic_wav.name.replace("mic_", "lar_")
        )

    @staticmethod
    def _ref_of(mic_wav):
        return Path(str(mic_wav).replace("/MIC/", "/REF/")).with_name(
            mic_wav.name.replace("mic_", "ref_").replace(".wav", ".f0")
        )

    @classmethod
    def _iter_originals(cls, root):
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
        return cls._read_wav_mono(Path(mic_wav))

    @classmethod
    def _read_original(cls, mic_wav):
        mic_wav = Path(mic_wav)
        speech, sr = cls._read_speech(mic_wav)
        lar = cls._lar_of(mic_wav)
        egg, _ = cls._read_wav_mono(lar) if lar.exists() else (None, sr)
        return speech, egg, sr

    def get_group(self, idx):
        parts = self.items[idx][1].split("_")
        return parts[1] if len(parts) >= 2 else "unknown"


def read_nsp(raw):
    magic = b"FORMDS16"
    if raw[:len(magic)] != magic:
        raise ValueError("not an NSP FORMDS16 file")
    pos = len(magic) + 4
    sr = None
    data = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        sz = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + sz]
        pos += 8 + sz + (sz & 1)
        if cid in (b"HEDR", b"HDR8"):
            sr = int.from_bytes(body[20:24], "little")
        elif cid in (b"SDA_", b"SD_A"):
            data = np.frombuffer(body[: sz - (sz % 2)], dtype="<i2")
    if data is None or not sr:
        raise ValueError("NSP missing SDA_/HEDR")
    if not 8000 <= sr <= 96000:
        raise ValueError(f"NSP sample rate {sr} Hz out of range (HEDR parse suspect for this file)")
    return data.astype(np.float32) / 32768.0, int(sr)


class PitchDatasetSVD(LaryngographSpeechDataset):

    NAME = "SVD"

    @classmethod
    def _speaker_map(cls, root):
        ov = root / "overview.csv"
        if not ov.exists():
            return {}
        m = {}
        with open(ov, encoding="utf-8", errors="ignore", newline="") as f:
            for r in csv.DictReader(f):
                m[r["AufnahmeID"]] = r.get("SprecherID", r["AufnahmeID"])
        return m

    @classmethod
    def _iter_originals(cls, root):
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
        nsp, egg = loc
        speech, sr = read_nsp(Path(nsp).read_bytes())
        egg_sig, _ = read_nsp(Path(egg).read_bytes())
        m = min(len(speech), len(egg_sig))
        return speech[:m].astype(np.float64), egg_sig[:m].astype(np.float64), sr
