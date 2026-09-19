import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel

from resampling import frame_times

from .base import ContinuousPitchAlgorithm, resample_audio, salience_band_mask

MODEL_SAMPLE_RATE = 16000
N_CLASS = 360
MODEL_HOP_LENGTH = 160
TIME_PAD_MULTIPLE = 32
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = 8000
WINDOW_LENGTH = 1024
MEL_CLAMP = 1e-5
BN_MOMENTUM = 0.01
KERNEL_SIZE = (2, 2)
N_BLOCKS = 4
N_ENCODERS = 5
N_INTERMEDIATE = 4
ENCODER_OUT_CHANNELS = 16
GRU_HIDDEN = 256
CENTS_MAPPING = np.linspace(0, 7180, N_CLASS) + 1997.3794084376191

MODEL_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
MODEL_SHA256 = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"
MODEL_PATH = Path(__file__).parent / "rmvpe.pt"


def verify_weights(path, source):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"{source} is {digest}, not the {MODEL_SHA256} this benchmark measured; "
            f"the published RMVPE row does not describe these weights")


def get_model_path():
    if not MODEL_PATH.exists():
        print(f"Downloading RMVPE weights from {MODEL_URL}...")
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


class MelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        mel_basis = mel(sr=MODEL_SAMPLE_RATE, n_fft=WINDOW_LENGTH, n_mels=N_MELS,
                        fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float(), persistent=False)
        self.register_buffer("hann_window", torch.hann_window(WINDOW_LENGTH), persistent=False)

    def forward(self, audio):
        fft = torch.stft(audio, n_fft=WINDOW_LENGTH, hop_length=MODEL_HOP_LENGTH,
                         win_length=WINDOW_LENGTH, window=self.hann_window, center=True,
                         return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, magnitude), min=MEL_CLAMP))


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        self.is_shortcut = in_channels != out_channels
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        return self.conv(x) + (self.shortcut(x) if self.is_shortcut else x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels)])
        for _ in range(N_BLOCKS - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for block in self.conv:
            x = block(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=KERNEL_SIZE,
                               padding=(1, 1), output_padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels)])
        for _ in range(N_BLOCKS - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels))

    def forward(self, x, concat_tensor):
        x = torch.cat((self.conv1(x), concat_tensor), dim=1)
        for block in self.conv2:
            x = block(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1, momentum=BN_MOMENTUM)
        self.layers = nn.ModuleList()
        in_channels, out_channels = 1, ENCODER_OUT_CHANNELS
        for _ in range(N_ENCODERS):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, KERNEL_SIZE))
            in_channels = out_channels
            out_channels *= 2
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None)])
        for _ in range(N_INTERMEDIATE - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(N_ENCODERS):
            self.layers.append(ResDecoderBlock(in_channels, in_channels // 2))
            in_channels //= 2

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet0(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel)
        self.decoder = Decoder(self.encoder.out_channel)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        return self.decoder(self.intermediate(x), concat_tensors)


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=1, batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class E2E0(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelSpectrogram()
        self.unet = DeepUnet0()
        self.cnn = nn.Conv2d(ENCODER_OUT_CHANNELS, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * N_MELS, GRU_HIDDEN),
            nn.Linear(2 * GRU_HIDDEN, N_CLASS),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mel = self.mel(x.reshape(-1, x.shape[-1]))
        n_frames = mel.shape[-1]
        n_pad = TIME_PAD_MULTIPLE * ((n_frames - 1) // TIME_PAD_MULTIPLE + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        x = self.unet(mel.transpose(-1, -2).unsqueeze(1))
        if n_pad > 0:
            x = x[:, :, :-n_pad, :]
        return self.fc(self.cnn(x).transpose(1, 2).flatten(-2))


def to_local_average_cents(salience):
    out = np.zeros(len(salience))
    for i, row in enumerate(salience):
        if np.max(row) <= 0:
            continue
        center = int(np.argmax(row))
        start, end = max(0, center - 4), min(N_CLASS, center + 5)
        out[i] = np.sum(row[start:end] * CENTS_MAPPING[start:end]) / np.sum(row[start:end])
    return out


class RMVPEPitchAlgorithm(ContinuousPitchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model = E2E0()
        state_dict = torch.load(get_model_path(), map_location="cpu")
        missing, _unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"RMVPE checkpoint {MODEL_PATH} does not fit E2E0: {len(missing)} parameters "
                f"unsupplied. First few: {sorted(missing)[:5]}")
        self.model = model.eval()

    def _extract_raw_pitch_and_periodicity(self, audio):
        audio16 = resample_audio(audio.astype(np.float32), self.sample_rate, MODEL_SAMPLE_RATE)
        with torch.inference_mode():
            salience = self.model(torch.from_numpy(audio16)).squeeze(0).cpu().numpy()
        salience = np.where(salience_band_mask(CENTS_MAPPING, self.fmin, self.fmax),
                            salience, 0.0)

        cents = to_local_average_cents(salience)
        f0 = np.where(cents != 0, 10.0 * 2.0 ** (cents / 1200.0), 0.0)
        periodicity = np.max(salience, axis=1)

        times = frame_times(len(f0), MODEL_HOP_LENGTH, MODEL_SAMPLE_RATE)
        return times, f0, periodicity

    def _get_default_threshold(self):
        return 0.425
