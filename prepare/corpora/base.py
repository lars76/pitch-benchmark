from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf
import torch
import torchaudio

from grid import frame_times, is_voiced, resample_to_grid

GRID_ATTRS = ("sample_rate", "hop_size", "fmin", "fmax")


def copy_eval_attrs(dst, src):
    for attr in GRID_ATTRS:
        setattr(dst, attr, getattr(src, attr))


def base_index(dataset, idx):
    fn = getattr(dataset, "base_index", None)
    return int(fn(idx)) if fn is not None else int(idx)


def find_wav_f0_pairs(audio_dir, annot_dir):
    pairs = []
    for wav_path in audio_dir.glob("*.RESYN.wav"):
        csv_path = annot_dir / wav_path.name.replace(".wav", ".csv")
        if csv_path.exists():
            pairs.append((wav_path, csv_path))
    return sorted(pairs)


def frame_rms(waveform, hop_size, n_frames):
    frame_length = hop_size
    waveform = torch.nn.functional.pad(waveform.squeeze(), (frame_length // 2, 0))
    total_samples_needed = (n_frames - 1) * hop_size + frame_length
    padding_needed = max(0, total_samples_needed - waveform.size(-1))
    if padding_needed > 0:
        waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
    frames = waveform.unfold(0, frame_length, hop_size)[:n_frames]
    return torch.sqrt(torch.mean(frames**2, dim=1))


class PitchDataset(ABC):

    def __init__(self, sample_rate, hop_size, fmin=None, fmax=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        if fmin is not None:
            self.fmin = fmin
        if fmax is not None:
            self.fmax = fmax

    @abstractmethod
    def get_group(self, idx):
        pass

    MAX_ITEM_SECONDS = None

    def item_grid_frames(self, idx):
        raise NotImplementedError(
            f"{type(self).__name__} has no cheap item_grid_frames; it cannot be segmented"
        )

    def item_grid_silent(self, idx):
        return np.zeros(self.item_grid_frames(idx), dtype=bool)

    def _grid_frames(self, path):
        info = sf.info(str(path))
        return int(info.frames / info.samplerate * self.sample_rate) // self.hop_size - 1

    def _load_waveform(self, path):
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr
        except Exception as e:
            raise OSError(f"Error loading audio file {path}: {e!s}") from e

    def _validate_audio(self, audio):
        if audio.dim() not in {1, 2}:
            raise ValueError(f"Audio must be 1D or 2D, got {audio.dim()}D")

        audio = torch.nan_to_num(audio, nan=0)

        if torch.all(audio == 0):
            raise ValueError("Silent audio!")

        max_abs = audio.abs().max()
        if max_abs > 1:
            audio = audio / max_abs

        return audio.clamp(-1.0, 1.0)

    def _enforce_voicing_invariant(self, pitch, periodicity):
        pitch = pitch * is_voiced(periodicity).to(pitch.dtype)
        return pitch, periodicity

    def _validate_pitch(self, pitch, periodicity):
        if pitch.shape != periodicity.shape:
            raise ValueError(
                f"Pitch and periodicity shapes must match: {pitch.shape} vs {periodicity.shape}"
            )

        pitch = torch.nan_to_num(pitch, nan=0.0)

        periodicity = torch.nan_to_num(periodicity, nan=0.0).clamp(0, 1)

        pitch, periodicity = self._enforce_voicing_invariant(pitch, periodicity)
        return pitch, periodicity.float()

    def _prepare_audio(self, audio, orig_sr):
        audio = audio.squeeze()
        if audio.dim() == 2:
            audio = audio.mean(0)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if orig_sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio, orig_freq=orig_sr, new_freq=self.sample_rate
            )
        return self._validate_audio(audio)

    def _load_csv_f0_annotation(self, csv_path, delimiter=","):
        try:
            data = np.loadtxt(csv_path, delimiter=delimiter, ndmin=2)
            times = data[:, 0].astype(float)
            pitch = torch.from_numpy(data[:, 1]).float()
            periodicity = (pitch > 0).float()
            return times, pitch, periodicity
        except Exception as e:
            raise OSError(f"Error loading annotation file {csv_path}: {e!s}") from e

    def _sample_from_csv(self, wav_path, csv_path, *, delimiter=",", offset=0.0):
        waveform, sr = self._load_waveform(wav_path)
        waveform = waveform.squeeze()
        times, pitch, periodicity = self._load_csv_f0_annotation(csv_path, delimiter)
        waveform, pitch, periodicity = self.process_sample(
            waveform, pitch, periodicity, sr, label_times=times + offset
        )
        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": wav_path,
        }

    def process_sample(self, audio, pitch, periodicity, orig_sr, *, label_times):
        audio = self._prepare_audio(audio, orig_sr)

        target_length = audio.size(-1) // self.hop_size
        if target_length < 1:
            raise ValueError(
                f"{type(self).__name__}: audio too short for one frame "
                f"(samples={audio.size(-1)}, hop={self.hop_size})"
            )

        label_times = np.asarray(label_times, dtype=np.float64).reshape(-1)
        if not (label_times.shape[0] == pitch.reshape(-1).numel() == periodicity.reshape(-1).numel()):
            raise ValueError(
                f"{type(self).__name__}: label_times ({label_times.shape[0]}), pitch "
                f"({pitch.reshape(-1).numel()}) and periodicity ({periodicity.reshape(-1).numel()}) "
                f"must be frame-aligned (one true time per pitch/periodicity frame)."
            )

        target_times = frame_times(target_length, self.hop_size, self.sample_rate)
        pitch_g, per_g = resample_to_grid(
            pitch.detach().cpu().numpy(), periodicity.detach().cpu().numpy(),
            label_times, target_times)
        pitch, periodicity = torch.from_numpy(pitch_g).float(), torch.from_numpy(per_g).float()

        pitch, periodicity = self._validate_pitch(pitch, periodicity)
        return audio.squeeze(0), pitch, periodicity

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return self._load_sample(idx)

    @abstractmethod
    def _load_sample(self, idx):
        pass
