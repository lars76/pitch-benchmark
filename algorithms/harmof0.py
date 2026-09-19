import hashlib
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from numpy.lib.stride_tricks import as_strided

from resampling import frame_times

from .base import ContinuousPitchAlgorithm, resample_audio, salience_band_mask

MODEL_SAMPLE_RATE = 16000
MODEL_HOP_LENGTH = 160
MODEL_WINDOW = 1024
FMIN_HZ = 27.5
BINS_PER_OCTAVE = 48
N_BINS = 352
N_HARMONICS = 12
CHANNELS = (32, 64, 128, 128)
DILATION = 48
TOP_DB = 80.0
BIN_HZ = FMIN_HZ * 2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE)
CENTS_MAPPING = 1200.0 * np.log2(BIN_HZ / 10.0)

MODEL_URL = ("https://github.com/WX-Wei/HarmoF0/raw/3b2223649953f7786b602c8b7d608bce1562da38/"
             "harmof0/checkpoints/mdb-stem-synth.pth")
MODEL_SHA256 = "1d47ce083858782cf3de78b1a0f61d98e5836b5a81745a3e47f2ad8ddafec99e"
MODEL_PATH = Path(__file__).parent / "harmof0-mdb-stem-synth.pth"
UNUSED_STATE_KEY = "waveform_to_logspecgram.waveform_to_specgram.window"


def verify_weights(path, source):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"{source} is {digest}, not the {MODEL_SHA256} this benchmark measured; "
            f"the published HarmoF0 row does not describe these weights")


def get_model_path():
    if not MODEL_PATH.exists():
        print(f"Downloading HarmoF0 weights from {MODEL_URL}...")
        partial = MODEL_PATH.with_name(f"{MODEL_PATH.name}.{os.getpid()}.part")
        try:
            urllib.request.urlretrieve(MODEL_URL, str(partial))
        except Exception as e:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download model: {e}") from e
        try:
            verify_weights(partial, MODEL_URL)
        except RuntimeError:
            partial.unlink(missing_ok=True)
            raise
        os.replace(partial, MODEL_PATH)
    return str(MODEL_PATH)


class MRDConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list):
        super().__init__()
        self.dilation_list = list(dilation_list)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) for _ in self.dilation_list])

    def forward(self, specgram):
        dilation = self.dilation_list[0]
        y = self.conv_list[0](specgram)
        y = nn.functional.pad(y, (0, dilation))
        y = y[:, :, :, dilation:]
        for conv, dilation in zip(self.conv_list[1:], self.dilation_list[1:]):
            x = conv(specgram)[:, :, :, dilation:]
            y[:, :, :, :x.size(3)] += x
        return y


class LogSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(MODEL_WINDOW)[None, None, :],
                             persistent=False)
        resolution = MODEL_SAMPLE_RATE / MODEL_WINDOW
        log_idxs = torch.from_numpy(BIN_HZ / resolution)
        floor = torch.floor(log_idxs)
        ceiling = torch.ceil(log_idxs)
        self.register_buffer("floor", floor.long(), persistent=False)
        self.register_buffer("floor_w", (log_idxs - floor).reshape(1, 1, N_BINS).float(),
                             persistent=False)
        self.register_buffer("ceiling", ceiling.long(), persistent=False)
        self.register_buffer("ceiling_w", (ceiling - log_idxs).reshape(1, 1, N_BINS).float(),
                             persistent=False)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)

    def forward(self, frames):
        spectrum = torch.fft.fft(frames * self.window)
        power = torch.abs(spectrum[:, :, :MODEL_WINDOW // 2 + 1]) ** 2
        warped = power[:, :, self.floor] * self.floor_w + power[:, :, self.ceiling] * self.ceiling_w
        return self.amplitude_to_db(warped)


def dila_conv_block(in_channel, out_channel, bins_per_octave, n_har, dilation_mode,
                    dilation_rate, dil_kernel_size, kernel_size, padding):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channel)
    if dilation_mode == "log_scale":
        dilation_list = [round(math.log2(k) * bins_per_octave) for k in range(1, n_har + 1)]
        dilated = MRDConv(out_channel, out_channel, dilation_list)
    elif dilation_mode == "fixed":
        dilated = nn.Conv2d(out_channel, out_channel, kernel_size=dil_kernel_size,
                            padding=(0, dilation_rate), dilation=(1, dilation_rate))
    else:
        raise ValueError(f"unknown dilation type: {dilation_mode}")
    return nn.Sequential(conv, nn.ReLU(), dilated, nn.ReLU(), batch_norm)


class HarmoF0(nn.Module):
    def __init__(self):
        super().__init__()
        self.waveform_to_logspecgram = LogSpectrogram()
        bins = BINS_PER_OCTAVE
        self.block_1 = dila_conv_block(1, CHANNELS[0], bins, N_HARMONICS, "log_scale", DILATION,
                                       (1, 3), kernel_size=(3, 3), padding=(1, 1))
        bins //= 2
        self.block_2 = dila_conv_block(CHANNELS[0], CHANNELS[1], bins, 3, "fixed", DILATION,
                                       (1, 3), kernel_size=(3, 3), padding=(1, 1))
        self.block_3 = dila_conv_block(CHANNELS[1], CHANNELS[2], bins, 3, "fixed", DILATION,
                                       (1, 3), kernel_size=(3, 3), padding=(1, 1))
        self.block_4 = dila_conv_block(CHANNELS[2], CHANNELS[3], bins, 3, "fixed", DILATION,
                                       (1, 3), kernel_size=(3, 3), padding=(1, 1))
        self.conv_5 = nn.Conv2d(CHANNELS[3], CHANNELS[3] // 2, kernel_size=(1, 1))
        self.conv_6 = nn.Conv2d(CHANNELS[3] // 2, 1, kernel_size=(1, 1))

    def forward(self, frames):
        x = self.waveform_to_logspecgram(frames)[None, :]
        x = self.block_4(self.block_3(self.block_2(self.block_1(x))))
        x = self.conv_6(torch.relu(self.conv_5(x)))
        return torch.sigmoid(x).squeeze(1)


class HarmoF0PitchAlgorithm(ContinuousPitchAlgorithm):

    CHUNK_SECONDS = 10.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = HarmoF0()
        state = torch.load(get_model_path(), map_location="cpu", weights_only=True)
        state.pop(UNUSED_STATE_KEY)
        self.model.load_state_dict(state)
        self.model.eval()
        self.band = salience_band_mask(CENTS_MAPPING, self.fmin, self.fmax)

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio = resample_audio(audio.astype(np.float32), self.sample_rate, MODEL_SAMPLE_RATE)
        audio = np.pad(audio, MODEL_WINDOW // 2, mode="constant", constant_values=0)
        n_frames = 1 + (len(audio) - MODEL_WINDOW) // MODEL_HOP_LENGTH
        frames = as_strided(audio, shape=(MODEL_WINDOW, n_frames),
                            strides=(audio.itemsize, MODEL_HOP_LENGTH * audio.itemsize))
        frames = torch.from_numpy(frames.transpose().copy())
        with torch.inference_mode():
            activation = self.model(frames[None])[0].numpy()
        activation = np.where(self.band, activation, 0.0)
        idx = activation.argmax(axis=1)
        pitch = BIN_HZ[idx]
        confidence = activation[np.arange(n_frames), idx]
        times = frame_times(n_frames, MODEL_HOP_LENGTH, MODEL_SAMPLE_RATE)
        return times, pitch, confidence
