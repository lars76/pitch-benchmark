import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from constants import CORPUS_FMIN, SPEECH_FMAX
from grid import resample_to_grid
from seeds import item_seed

from .base import PitchDataset

F0_MEAN_HZ, F0_STD_HZ = 224.5344, 75.3236


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * out_channels))
        nn.init.normal_(self.depthwise_conv.weight, mean=0, std=std)
        nn.init.normal_(self.pointwise_conv.weight, mean=0, std=std)
        nn.init.zeros_(self.pointwise_conv.bias)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class SepConvLayer(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = nn.ReLU(inplace=True)
        self.conv1 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)
        self.conv2 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.activation_fn(self.conv1(x))
        x = self.dropout(x)
        x = self.activation_fn(self.conv2(x))
        x = self.dropout(x)
        return residual + x


class LightSpeech(nn.Module):
    def __init__(
        self,
        num_phones,
        num_speakers,
        num_mel_bins,
        num_tones=7,
        tone_embedding=16,
        d_model=512,
        layer_dropout=0.2,
        encoder_kernel_sizes=(5, 25, 13, 9),
        decoder_kernel_sizes=(17, 21, 9, 3),
        duration_layers=1,
        duration_kernel_size=3,
        duration_dropout=0.25,
        pitch_layers=6,
        pitch_kernel_size=5,
        pitch_dropout=0.25,
        padding_idx=0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model

        self.num_speakers = num_speakers
        if self.num_speakers > 1:
            self.speaker_embedding = nn.Embedding(self.num_speakers, d_model)
        self.embed_tokens = nn.Embedding(
            num_phones, d_model - tone_embedding, padding_idx=self.padding_idx
        )
        self.embed_tones = nn.Embedding(
            num_tones, tone_embedding, padding_idx=self.padding_idx
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.embed_pitch = nn.Conv1d(2, d_model, kernel_size=1)

        self.encoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in encoder_kernel_sizes
            ]
        )
        self.decoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in decoder_kernel_sizes
            ]
        )

        self.duration_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=1,
            num_layers=duration_layers,
            kernel_size=duration_kernel_size,
            dropout=duration_dropout,
        )
        self.pitch_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=2,
            num_layers=pitch_layers,
            kernel_size=pitch_kernel_size,
            dropout=pitch_dropout,
        )

        self.layer_norm = LayerNorm1d(d_model)
        self.layer_norm2 = LayerNorm1d(d_model)
        self.mel_out = nn.Conv1d(d_model, num_mel_bins, kernel_size=1)

    @staticmethod
    def _make_predictor(hidden_size, out_dim, num_layers, dropout=0.5, kernel_size=3):
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    ConvSeparable(hidden_size, hidden_size, kernel_size),
                    nn.ReLU(inplace=True),
                    LayerNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Conv1d(hidden_size, out_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def _length_regulator(self, x, durations):
        indices = torch.arange(x.shape[1], device=x.device)
        return x[:, torch.repeat_interleave(indices, durations[0].long(), dim=0)]

    def forward(self, speakers, tokens, tones):
        x = torch.cat(
            (self.embed_tokens(tokens), self.embed_tones(tones)), dim=-1
        ).transpose(1, 2)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        encoder_outputs = self.layer_norm(x).transpose(1, 2)

        if self.num_speakers > 1:
            encoder_outputs += self.speaker_embedding(speakers.long()).unsqueeze(1)

        duration_prediction = self.duration_predictor(
            encoder_outputs.transpose(1, 2)
        ).squeeze(1)

        durations = torch.clamp(torch.round(torch.exp(duration_prediction) - 1), min=0).long()

        decoder_inp = self._length_regulator(encoder_outputs, durations)
        decoder_inp = self.dropout(decoder_inp).transpose(1, 2)

        pitch_feat = self.pitch_predictor(decoder_inp)
        decoder_inp += self.embed_pitch(pitch_feat.clone().detach())

        for decoder_layer in self.decoder:
            decoder_inp = decoder_layer(decoder_inp)
        decoder_outputs = self.mel_out(self.layer_norm2(decoder_inp)).transpose(1, 2)

        return decoder_outputs, pitch_feat[:, 0], pitch_feat[:, 1]


class PitchDatasetSpeechSynth(PitchDataset):

    fmin = CORPUS_FMIN
    fmax = SPEECH_FMAX
    SEED = 0
    WORD_RANGE = (3, 9)
    PERIODICITY_THRESHOLD = 0.4
    MAX_ITEM_SECONDS = 8.0

    def __init__(self, root_dir, **kwargs):
        super().__init__(**kwargs)
        self.model_file = Path(root_dir)
        self.device = torch.device("cpu")

        try:
            state_dict = torch.load(self.model_file, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{self.model_file}' not found") from None
        except Exception as e:
            raise OSError(f"Error loading model file '{self.model_file}': {e!s}") from e

        self.tts_model = (
            LightSpeech(
                num_phones=state_dict["num_phones"],
                num_speakers=state_dict["num_speakers"],
                num_mel_bins=state_dict["num_mel_bins"],
            )
            .to(self.device)
            .eval()
        )
        self.tts_model.load_state_dict(state_dict["state_dict"], strict=True)

        try:
            self.vocoder = torch.hub.load(
                "lars76/bigvgan-mirror",
                state_dict.get("vocoder_name", "hifigan_universal_v1"),
                trust_repo=True,
                pretrained=True,
                verbose=False,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error loading vocoder: {e!s}") from e

        self.num_speakers = state_dict["num_speakers"]
        self.pinyin_to_ipa = state_dict["pinyin_dict"]
        self.ipa_to_token = state_dict["phone_dict"]
        if not self.ipa_to_token:
            raise ValueError("Phone dictionary is empty in the loaded model")

        self.available_words = [
            word
            for word in self.pinyin_to_ipa
            if not word.startswith("<") and not word.endswith(">")
        ]
        if not self.available_words:
            raise ValueError("No valid words found in pinyin dictionary")

    def _convert_pinyin_to_ipa(self, pinyin_text):
        ipa_string = ""

        for syllable in pinyin_text.split():
            syllable = syllable.strip()

            if not syllable[-1].isdigit():
                syllable += "5"

            ipa_key = syllable[:-1]
            tone = syllable[-1]

            ipas = self.pinyin_to_ipa.get(ipa_key)
            if ipas is None:
                continue

            ipa_string += ipas.replace(" ", "") + tone + " "

        return ipa_string.strip()

    def _convert_ipa_to_tokens(self, ipa_text):
        token_ids = []
        tone_ids = []

        sorted_phonemes = sorted(self.ipa_to_token.keys(), key=len, reverse=True)

        for token in ipa_text.split():
            if token[-1].isdigit():
                tone_id = int(token[-1]) + 1
                ipa_key = token[:-1]
            else:
                continue

            i = 0
            while i < len(ipa_key):
                matched = False
                for phoneme in sorted_phonemes:
                    if ipa_key[i:].startswith(phoneme):
                        token_ids.append(self.ipa_to_token[phoneme])
                        tone_ids.append(tone_id)
                        i += len(phoneme)
                        matched = True
                        break
                if not matched:
                    break

        return token_ids, tone_ids

    def _generate_word_sequence(self, idx):
        rng = random.Random(item_seed(self.SEED, idx))
        num_words = rng.randint(*self.WORD_RANGE)
        selected_words = rng.sample(
            self.available_words, min(num_words, len(self.available_words))
        )
        pinyin_text = " ".join(selected_words)
        ipa_text = self._convert_pinyin_to_ipa(pinyin_text)
        token_ids, tone_ids = self._convert_ipa_to_tokens(ipa_text)
        return token_ids, tone_ids

    def get_group(self, idx):
        return f"speaker_{idx}"

    def __len__(self):
        return self.num_speakers

    @torch.inference_mode()
    def _generate_tts_sample(self, idx):
        token_ids, tone_ids = self._generate_word_sequence(idx)

        sil_token = self.ipa_to_token["<sil>"]
        token_ids = [sil_token, *token_ids, sil_token]
        tone_ids = [1, *tone_ids, 1]

        speaker_id_tensor = torch.tensor([idx], dtype=torch.long).to(self.device)
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        tone_ids_tensor = torch.tensor([tone_ids], dtype=torch.long).to(self.device)

        try:
            mel, pitch, periodicity = self.tts_model(speaker_id_tensor, tokens_tensor, tone_ids_tensor)

            wav = self.vocoder(mel.transpose(1, 2)).flatten().cpu()

            pitch = F0_STD_HZ * pitch.flatten().cpu() + F0_MEAN_HZ
            periodicity = (periodicity.flatten().cpu() > self.PERIODICITY_THRESHOLD).float()
            pitch = pitch * periodicity

            native_hop = (wav.numel() / pitch.numel()) / self.vocoder.sampling_rate
            mel_center = (self.vocoder.hop_size - 1) / 2 / self.vocoder.sampling_rate
            label_times = np.arange(pitch.numel()) * native_hop + mel_center

            wav, pitch, periodicity, label_times = self._impose_f0(wav, pitch, label_times)

            wav, pitch, periodicity = self.process_sample(
                wav, pitch, periodicity, self.vocoder.sampling_rate, label_times=label_times
            )

            return wav, pitch, periodicity

        except Exception as e:
            raise RuntimeError(f"Error during TTS generation: {e!s}") from e

    def _impose_f0(self, wav, pitch, label_times):
        import pyworld

        fs = self.vocoder.sampling_rate
        x = np.ascontiguousarray(wav.numpy().astype(np.float64))
        frame_period_s = pyworld.default_frame_period / 1000.0
        t = np.arange(int(len(x) / fs / frame_period_s) + 1) * frame_period_s
        f0, _ = resample_to_grid(pitch.numpy(), (pitch.numpy() > 0).astype(np.float64),
                                 label_times, t)
        f0 = np.ascontiguousarray(f0)
        envelope = pyworld.cheaptrick(x, f0, t, fs)
        aperiodicity = pyworld.d4c(x, f0, t, fs)
        y = pyworld.synthesize(f0, envelope, aperiodicity, fs, pyworld.default_frame_period)
        pitch = torch.from_numpy(f0).float()
        return torch.from_numpy(y).float(), pitch, (pitch > 0).float(), t

    def _load_sample(self, idx):
        waveform, pitch, periodicity = self._generate_tts_sample(idx)
        return {"audio": waveform.float(), "pitch": pitch.float(), "periodicity": periodicity,
                "wav_path": f"speaker_{idx}"}
