import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from .base import PitchDataset


class PitchDatasetM4Singer(PitchDataset):
    """M4Singer: 20 studio singers (Alto/Bass/Soprano/Tenor), ~21k Mandarin song utterances.

    Ground truth is derived from the committed `meta.json`, which gives per-PHONEME parallel arrays:
    `phs` (pinyin phonemes incl. the rest markers `<SP>` silent-pause / `<AP>` aspiration), `ph_dur`
    (phoneme durations in seconds, summing to the utterance length), and `notes` (the score MIDI note
    per phoneme; 0 = rest). We build piecewise-constant per-frame labels by walking `ph_dur`: a frame
    is VOICED where its phoneme's `notes > 0`, with pitch = midi_to_hz(note).

    LABEL QUALITY: BOTH axes are weak references, because the label is the score MIDI mapped through
    the phoneme forced-alignment. (1) Pitch is the *intended score* note (not performed f0) -> SCORE-
    GRADE, not a 50c/50ms pitch benchmark. (2) Voicing is only HALF-reliable: the syllable's note is
    assigned to its onset consonant phoneme too, so ~12-20% of `notes>0` time falls on voiceless
    obstruents (sh/x/s/p/t/k/...) that a correct tracker unvoices -> voicing RECALL/F1 are biased low
    (achievable recall capped ~85%), and forced-alignment jitter blurs every boundary. Only voicing
    PRECISION is clean (the `<SP>`/`<AP>` rests are real silence). Not on any leaderboard; if used for
    voicing at all, use precision, not F1/recall, and expect a systematic recall floor.

    Args:
        root_dir (str): the m4singer root (holds `meta.json` + one dir per `singer#song`).
        use_cache (bool): cache decoded samples. Defaults to True.
    """
    # Caveat: score-grade GT (the intended note, not the performed f0): voicing labels are
    # reliable, pitch-accuracy labels are not. Scoring M4Singer therefore measures the
    # annotation convention, and (because pitch feeds theta*) shifts every tracker's pitch
    # table. It is not excluded automatically -- it simply is not in a run unless you supply
    # its path. Note intervals are score MIDI, so COnP against it is the same 50-cent pitch
    # test with the same caveat.
    provides_notes = True
    fmin = 65
    fmax = 2093
    # Native label grid for the piecewise-constant score labels; fine enough that resampling onto the
    # 16 kHz / hop-256 eval grid never straddles a phoneme boundary by more than one native step.
    LABEL_HOP_SECONDS = 0.005

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(use_cache=use_cache, **kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")
        meta_path = self.root_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"m4singer meta.json not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)
        # item_name is "singer#song#stem" -> wav at root/"singer#song"/"stem.wav"
        self.items = []
        for it in meta:
            singer, song, stem = it["item_name"].split("#")
            wav = self.root_dir / f"{singer}#{song}" / f"{stem}.wav"
            if wav.exists():
                self.items.append((wav, singer, it))
        if not self.items:
            raise ValueError(f"No valid utterances found in '{root_dir}' (checked {len(meta)} meta rows)")

    def get_group(self, idx: int) -> str:
        """Leakage-safe cluster id = the singer (20 of them: Alto-1..7, Bass-1..3, ...)."""
        return self.items[idx][1]

    def __len__(self) -> int:
        return len(self.items)

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor | Path]:
        wav_path, _singer, it = self.items[idx]

        try:
            waveform, sr = torchaudio.load(wav_path)   # m4singer wavs are variable-rate (44.1/48/96k)
            waveform = waveform.squeeze()
            if waveform.dim() > 1:                      # any stray stereo -> mono
                waveform = waveform.mean(dim=0)
        except Exception as e:
            raise OSError(f"Error loading audio file {wav_path}: {e!s}") from e

        # Piecewise-constant score labels on a fine native grid: assign each grid time the MIDI note
        # of the phoneme interval it falls in (notes==0 / <SP> / <AP> -> unvoiced).
        ph_dur = np.asarray(it["ph_dur"], dtype=np.float64)
        notes = np.asarray(it["notes"], dtype=np.float64)
        bounds = np.concatenate([[0.0], np.cumsum(ph_dur)])       # phoneme boundaries in seconds
        total = float(bounds[-1])
        label_times = np.arange(0.0, total, self.LABEL_HOP_SECONDS)
        if label_times.size == 0:
            label_times = np.array([0.0])
        ph_idx = np.clip(np.searchsorted(bounds, label_times, side="right") - 1, 0, len(notes) - 1)
        midi = notes[ph_idx]
        voiced = midi > 0
        pitch_hz = np.zeros_like(midi)
        if np.any(voiced):
            pitch_hz[voiced] = librosa.midi_to_hz(midi[voiced])

        pitch = torch.from_numpy(pitch_hz).float()
        periodicity = torch.from_numpy(voiced.astype(np.float32))
        waveform, pitch, periodicity = self.process_sample(
            waveform, pitch, periodicity, sr, label_times=label_times
        )
        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": wav_path,
        }
