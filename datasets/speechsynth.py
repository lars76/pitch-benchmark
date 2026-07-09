import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from resampling import resample_to_grid

from .augment import item_seed
from .base import PitchDataset


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvSeparable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0,
    ):
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

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class SepConvLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layer_norm = LayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = nn.ReLU(inplace=True)
        self.conv1 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)
        self.conv2 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
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
        num_phones: int,
        num_speakers: int,
        num_mel_bins: int,
        num_tones: int = 7,
        tone_embedding: int = 16,
        d_model: int = 512,
        layer_dropout: float = 0.2,
        encoder_kernel_sizes: tuple[int, ...] = (5, 25, 13, 9),
        decoder_kernel_sizes: tuple[int, ...] = (17, 21, 9, 3),
        duration_layers: int = 1,
        duration_kernel_size: int = 3,
        duration_dropout: float = 0.25,
        pitch_layers: int = 6,
        pitch_kernel_size: int = 5,
        pitch_dropout: float = 0.25,
        padding_idx: int = 0,
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
    def _make_predictor(
        hidden_size: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        kernel_size: int = 3,
    ):
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

    def _length_regulator(self, x: Tensor, mel_time: int, durations: Tensor) -> Tensor:
        bsz, time, feats = x.shape
        if bsz > 1:
            cumulative_durations = torch.cumsum(durations, dim=1)
            expanded_range = (
                torch.arange(mel_time, device=x.device).unsqueeze(0).expand(bsz, -1)
            )
            mask = expanded_range.unsqueeze(1) >= cumulative_durations.unsqueeze(2)
            source_indices = mask.long().sum(dim=1)
            # Clamp indices for the case mel_time > total_duration.
            source_indices = torch.clamp(source_indices, 0, time - 1)
            gather_indices = source_indices.unsqueeze(-1).expand(-1, -1, feats)
            return torch.gather(x, 1, gather_indices)
        else:
            indices = torch.arange(time, device=x.device)
            repeated_indices = torch.repeat_interleave(
                indices, durations[0].long(), dim=0
            )
            return x[:, repeated_indices]

    def forward(
        self,
        speakers: Tensor,
        tokens: Tensor,
        tones: Tensor,
        pitches: Tensor | None = None,
        periodicity: Tensor | None = None,
        durations: Tensor | None = None,
        mels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

        if mels is not None and durations is not None:
            durations = torch.clamp(torch.round(durations), min=0).long()
            mel_time = mels.shape[1]
            assert torch.max(torch.sum(durations, dim=1)).item() == mel_time
        else:
            duration_prediction = torch.exp(duration_prediction) - 1
            durations = torch.clamp(torch.round(duration_prediction), min=0).long()
            mel_time = torch.max(torch.sum(durations, dim=1)).long()

        decoder_inp = self._length_regulator(encoder_outputs, mel_time, durations)
        decoder_inp = self.dropout(decoder_inp).transpose(1, 2)

        pitch_feat = self.pitch_predictor(decoder_inp)
        new_feat = (
            torch.stack((pitches, periodicity), dim=2).transpose(1, 2)
            if pitches is not None
            else pitch_feat.clone()
        )
        new_feat = new_feat.detach()

        decoder_inp += self.embed_pitch(new_feat)

        for decoder_layer in self.decoder:
            decoder_inp = decoder_layer(decoder_inp)
        decoder_outputs = self.layer_norm2(decoder_inp)

        decoder_outputs = self.mel_out(decoder_outputs).transpose(1, 2)

        return (
            decoder_outputs,
            duration_prediction,
            pitch_feat[:, 0],
            pitch_feat[:, 1],
        )


class PitchDatasetSpeechSynth(PitchDataset):
    """
    Synthetic Mandarin speech with EXACT pitch/voicing ground truth, generated on the fly.

    How a sample is made (default, exact_f0=True):
      1. A LightSpeech TTS model (trained in lars76/fastspeech2-clean on AISHELL-3 + biaobei +
         more, 219 speakers) INVENTS a per-frame f0/periodicity contour and a mel spectrogram for
         a random Chinese word sequence. The contour is a generative output, not a measurement of
         any waveform -- it becomes the label.
      2. A HiFi-GAN vocoder renders the mel to audio. This audio is used only as a TIMBRE DONOR.
      3. WORLD re-synthesizes the final waveform FROM the label contour: CheapTrick (spectral
         envelope) and D4C (aperiodicity) analyze the donor's voice character, and
         pyworld.synthesize generates the excitation with pulses placed at exactly 1/f0(t) of the
         label, silence/noise where the label is unvoiced.

    No pitch DETECTION happens anywhere in this pipeline: pitch flows from the labels into the
    waveform, never the other way, so the labels are true by construction (WORLD's DIO/Harvest
    estimators are NOT used; only its synthesizer is). Verified empirically against
    timestamp-calibrated trackers: 4-8 cents label floor and +0.95 ms label-offset consensus,
    the same exactness class as MDB-stem-synth / Bach10Synth (see TIMING.md).

    Why the imposition step exists: the raw HiFi-GAN audio does NOT realize the conditioning
    contour faithfully (~25 cents median deviation; 42 cents residual even after a per-utterance
    linear fit) because LightSpeech's decoder was trained with mistimed pitch labels (index-warped
    PENN targets in the upstream preprocessing) and learned to treat pitch conditioning as a soft
    hint. exact_f0=False returns that raw audio with the therefore-mislabeled contour, for
    listening/debugging only -- never for benchmarking.

    Character caveats (documented, accepted): the audio is analysis/synthesis (WORLD-vocoded)
    rather than raw neural speech, and the invented prosody is somewhat idealized. Exactness of
    the ground truth is the property this corpus exists for.

    Requires pyworld for the default exact_f0 path: `uv sync --extra speechsynth` (also included
    in the dio/harvest extras and in --all-extras).

    Args:
        root_dir (str): Path to the TTS model checkpoint (LightSpeech architecture).
            Defaults to "datasets/speechsynth.pt".
        samples_per_speaker (int): Number of samples per speaker. Defaults to 1
        word_range (Tuple): Range of words per generated sample. Defaults to (3, 9)
        periodicity_threshold (float): Threshold for periodicity detection. Defaults to 0.4
        device (str): Device to run inference on. Defaults to "cuda"
        vocoder_name (Optional[str]): Vocoder model name; None (default) uses the vocoder recorded
            in the checkpoint (the artifact knows what it was built for)
        exact_f0 (bool): Re-synthesize with the label f0 imposed (see above). Defaults to True
        seed (int): Corpus seed; sample idx is a pure function of (checkpoint, seed, idx) via a
            per-index private RNG, so the corpus is identical across runs, processes and access
            orders. Deliberately independent of the benchmark run seed. Defaults to 0
        use_cache (bool): Whether to cache generated samples. Defaults to True
        **kwargs: Additional arguments passed to PitchDataset
    """

    fmin = 65
    fmax = 300

    def __init__(
        self,
        root_dir: str = "datasets/speechsynth.pt",
        samples_per_speaker: int = 1,
        word_range: tuple = (3, 9),
        periodicity_threshold: float = 0.4,
        device: str = "cuda",
        vocoder_name: str | None = None,
        exact_f0: bool = True,
        seed: int = 0,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(use_cache=use_cache, **kwargs)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.model_file = Path(root_dir)
        self.samples_per_speaker = samples_per_speaker
        self.word_range = word_range
        self.periodicity_threshold = periodicity_threshold
        self.device = torch.device(device)
        self.vocoder_name = vocoder_name
        self.exact_f0 = exact_f0
        self.seed = seed

        self._initialize_models()

        # Calculate total dataset size
        self.total_samples = self.num_speakers * self.samples_per_speaker

    def _initialize_models(self):
        """Initialize the TTS model and vocoder."""
        try:
            state_dict = torch.load(
                self.model_file, map_location=self.device, weights_only=False
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{self.model_file}' not found") from None
        except Exception as e:
            raise OSError(f"Error loading model file '{self.model_file}': {e!s}") from e

        # Initialize LightSpeech model
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

        # Initialize vocoder. None -> the one recorded in the checkpoint: the artifact knows what
        # it was built for (an earlier hardcoded default silently used hifigan_lj_ft_t2_v1, an
        # LJSpeech-finetuned single-speaker vocoder, while the checkpoint records the universal one).
        if self.vocoder_name is None:
            self.vocoder_name = state_dict.get("vocoder_name", "hifigan_universal_v1")
        try:
            self.vocoder = torch.hub.load(
                "lars76/bigvgan-mirror",
                self.vocoder_name,
                trust_repo=True,
                pretrained=True,
                verbose=False,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error loading vocoder: {e!s}") from e

        # Store model parameters and dictionaries
        self.num_speakers = state_dict["num_speakers"]
        self.pinyin_to_ipa = state_dict["pinyin_dict"]
        self.ipa_to_token = state_dict["phone_dict"]
        self.speaker_info = state_dict["speaker_dict"]

        if not self.ipa_to_token:
            raise ValueError("Phone dictionary is empty in the loaded model")

        # Extract available pinyin words (excluding special tokens)
        self.available_words = [
            word
            for word in self.pinyin_to_ipa
            if not word.startswith("<") and not word.endswith(">")
        ]

        if not self.available_words:
            raise ValueError("No valid words found in pinyin dictionary")

    def _convert_pinyin_to_ipa(self, pinyin_text: str) -> str:
        """Convert Pinyin text to IPA notation with tones."""
        ipa_string = ""

        for syllable in pinyin_text.split():
            syllable = syllable.strip()

            # Ensure each syllable ends with a digit for tone
            if not syllable[-1].isdigit():
                if syllable == "<sil>":
                    syllable += "1"
                else:
                    syllable += "5"  # Default to neutral tone

            ipa_key = syllable[:-1]
            tone = syllable[-1]

            ipas = self.pinyin_to_ipa.get(ipa_key)
            if ipas is None:
                continue

            ipa_string += ipas.replace(" ", "") + tone + " "

        return ipa_string.strip()

    def _convert_ipa_to_tokens(self, ipa_text: str) -> tuple[list[int], list[int]]:
        """Convert IPA text to token and tone IDs."""
        token_ids = []
        tone_ids = []

        # Sort phonemes by length in descending order for proper matching
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

    def _generate_word_sequence(self, idx: int) -> tuple[list[int], list[int]]:
        """Generate sample `idx`'s word sequence and convert to tokens.

        The words come from a PRIVATE RNG seeded per (dataset seed, idx) -- the same pattern the
        augmentation layer uses (datasets/augment.py) -- so sample `idx` is a pure function of
        (checkpoint, seed, idx). Drawing from the GLOBAL `random` stream instead would make each
        sample depend on everything that consumed the stream earlier (other libraries, access
        order, cache state), silently changing the corpus between otherwise-identical runs. The
        default seed is a constant, deliberately DECOUPLED from the benchmark run seed: the corpus
        stays fixed across runs (like every file-backed dataset) and the run seed varies only the
        augmentation noise."""
        rng = random.Random(item_seed(self.seed, idx))
        num_words = rng.randint(*self.word_range)
        selected_words = rng.sample(
            self.available_words, min(num_words, len(self.available_words))
        )
        pinyin_text = " ".join(selected_words)
        ipa_text = self._convert_pinyin_to_ipa(pinyin_text)
        token_ids, tone_ids = self._convert_ipa_to_tokens(ipa_text)
        return token_ids, tone_ids

    def get_group(self, idx: int) -> str:
        """Return group identifier for sample (speaker ID)"""
        speaker_id = idx // self.samples_per_speaker
        return f"speaker_{speaker_id}"

    def __len__(self) -> int:
        return self.total_samples

    @torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=torch.cuda.is_available(),
    )
    @torch.inference_mode()
    def _generate_tts_sample(
        self, speaker_id: int, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate sample `idx` (audio, pitch, periodicity) for `speaker_id`."""
        token_ids, tone_ids = self._generate_word_sequence(idx)

        # Pad with silence at both ends.
        sil_token = self.ipa_to_token["<sil>"]
        token_ids = [sil_token, *token_ids, sil_token]
        tone_ids = [1, *tone_ids, 1]

        speaker_id_tensor = torch.tensor([speaker_id], dtype=torch.long).to(self.device)
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        tone_ids_tensor = torch.tensor([tone_ids], dtype=torch.long).to(self.device)

        try:
            mel, _dur, pitch, periodicity = self.tts_model(
                speaker_id_tensor, tokens_tensor, tone_ids_tensor
            )

            wav = self.vocoder(mel.transpose(1, 2)).flatten().cpu()

            # De-normalize the model's pitch output back to Hz, binarize voicing, and zero the
            # contour on unvoiced frames (pitch > 0 <=> voiced, the shared label contract).
            pitch = 75.3236 * pitch.flatten().cpu() + 224.5344
            periodicity = (
                periodicity.flatten().cpu() > self.periodicity_threshold
            ).float()
            pitch = pitch * periodicity

            # Labels are at the mel frame rate; the vocoder upsamples each mel frame to
            # (len(wav)/len(pitch)) samples, so the native hop in seconds is that ratio over the
            # vocoder sample rate. Mel frame i is NOT centered at i*hop: the upstream preprocessing
            # (lars76/fastspeech2-clean, compute_mel) pads (n_fft - hop)/2 with center=False, so
            # the analysis window centre sits at i*hop + (hop-1)/2 samples (+5.8 ms at 22.05 kHz).
            # Anchoring the contour there keeps the imposed f0 in sync with the donor envelope's
            # phonetic content (and is the honest stamp for the exact_f0=False debug path).
            native_hop = (wav.numel() / pitch.numel()) / self.vocoder.sampling_rate
            mel_center = (self.vocoder.hop_size - 1) / 2 / self.vocoder.sampling_rate
            label_times = np.arange(pitch.numel()) * native_hop + mel_center

            if self.exact_f0:
                wav, pitch, periodicity, label_times = self._impose_f0(wav, pitch, label_times)

            wav, pitch, periodicity = self.process_sample(
                wav, pitch, periodicity, self.vocoder.sampling_rate, label_times=label_times
            )

            return wav, pitch, periodicity

        except Exception as e:
            raise RuntimeError(f"Error during TTS generation: {e!s}") from e

    def _impose_f0(
        self, wav: torch.Tensor, pitch: torch.Tensor, label_times: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """WORLD-resynthesize the neural audio with the label contour IMPOSED (class docstring).

        Returns (wav, pitch, periodicity, label_times) on WORLD's 5 ms analysis grid; the
        synthesized waveform realizes exactly that f0, so the labels are true by construction."""
        # Lazy: only the exact_f0 path needs pyworld.
        try:
            import pyworld
        except ImportError as e:
            raise ImportError(
                "SpeechSynth's exact_f0 resynthesis needs pyworld; install it with "
                "`uv sync --extra speechsynth` (also included in the dio/harvest extras)."
            ) from e

        fs = self.vocoder.sampling_rate
        x = np.ascontiguousarray(wav.numpy().astype(np.float64))
        frame_period_s = pyworld.default_frame_period / 1000.0  # 5 ms
        t = np.arange(int(len(x) / fs / frame_period_s) + 1) * frame_period_s
        # Label contour onto WORLD's grid at true times, gap-aware (never interpolates through
        # the unvoiced 0 sentinel) -- the benchmark's one resampler.
        f0, _ = resample_to_grid(
            pitch.numpy(),
            (pitch.numpy() > 0).astype(np.float64),
            label_times,
            t,
            voicing_kind="nearest",
        )
        f0 = np.ascontiguousarray(f0)
        envelope = pyworld.cheaptrick(x, f0, t, fs)
        aperiodicity = pyworld.d4c(x, f0, t, fs)
        y = pyworld.synthesize(f0, envelope, aperiodicity, fs, pyworld.default_frame_period)
        pitch = torch.from_numpy(f0).float()
        return torch.from_numpy(y).float(), pitch, (pitch > 0).float(), t

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | int]:
        # Calculate which speaker this sample belongs to
        speaker_id = idx // self.samples_per_speaker

        try:
            waveform, pitch, periodicity = self._generate_tts_sample(speaker_id, idx)
        except Exception as e:
            raise RuntimeError(f"Error generating sample at index {idx}: {e!s}") from e

        return {
            "audio": waveform.float(),
            "pitch": pitch.float(),
            "periodicity": periodicity,
            "speaker_id": speaker_id,
            "sample_idx": idx,
        }
