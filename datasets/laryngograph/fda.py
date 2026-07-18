"""FDA (Bagshaw / CSTR 'fda_eval'): 50 sentences x 2 speakers, studio quality at 20 kHz, with a
laryngograph. Scored against the cross-family ``consensus`` (built from ``.lar``) by default, like
every EGG corpus; ``label_source="reference"`` instead loads FDA's shipped ``.fx`` f0 contour (the
dataset authors' own reference).

Raw Edinburgh files (read directly from the extracted download):
  - ``<id>.sig``  headerless big-endian int16 PCM speech @ 20 kHz,
  - ``<id>.fx``   an XMG f0 contour (ASCII after a 0x0c byte; ``time_ms freq_hz`` pairs; ``=`` marks
                  an unvoiced break), the author reference,
  - ``<id>.lar``  the laryngograph waveform (same raw int16 @ 20 kHz), the EGG for consensus.
"""
from pathlib import Path

import numpy as np
import torch

from .base import LaryngographSpeechDataset


def _read_raw_i16(path: Path) -> np.ndarray:
    """Headerless big-endian int16 PCM (FDA .sig / .lar) -> float64 mono in [-1, 1]."""
    return np.fromfile(str(path), dtype=">i2").astype(np.float64) / 32768.0


class PitchDatasetFDA(LaryngographSpeechDataset):
    NAME = "FDA"
    SUPPORTS_REFERENCE = True
    fmin = 50.0
    fmax = 500.0
    SR = 20000                      # .sig / .lar native sample rate
    FX_STEP_S = 0.005               # the uniform grid the .fx contour is rasterized onto
    # (no REFERENCE_LABEL_HOP_SECONDS: _load_reference_labels always returns explicit `times`, so the
    # base's hop fallback is unreachable for FDA: the 5 ms grid is FX_STEP_S, defined once above.)
    # Label-offset sweep (scripts/check_dataset_alignment.py, 17 files x 30 s):
    # Praat +2.38 / DIO +1.22 / SWIPE +1.53 ms -> consensus +1.53 ms (labels stamped early).
    REFERENCE_LABEL_OFFSET_SECONDS = 0.0015

    @classmethod
    def _iter_originals(cls, root: Path):
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
        """Runtime speech-only: read just the .sig speech, skipping the .lar/EGG decode."""
        return _read_raw_i16(Path(sig_path)), cls.SR

    @classmethod
    def _read_original(cls, sig_path):
        """Decode one utterance -> ``(speech, egg, sr)`` from the raw .sig + .lar @ 20 kHz."""
        sig_path = Path(sig_path)
        speech, sr = cls._read_speech(sig_path)
        lar = sig_path.with_suffix(".lar")
        egg = _read_raw_i16(lar) if lar.exists() else None
        return speech, egg, sr

    def get_group(self, idx: int) -> str:
        return self._loc_path(self.items[idx][0]).parent.name   # speaker = "rl" / "sb"

    @staticmethod
    def _read_fx(path: Path) -> list[list[tuple[float, float]]]:
        """Parse an XMG .fx into voiced runs of (time_s, hz), split on '=' breaks."""
        raw = path.read_bytes()
        cut = raw.find(b"\x0c")                              # header ends at CTRL-L
        body = (raw[cut + 1:] if cut >= 0 else raw).decode("latin-1")
        runs: list[list[tuple[float, float]]] = []
        cur: list[tuple[float, float]] = []
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("="):
                if cur:
                    runs.append(cur)
                    cur = []
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    t_ms, hz = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                cur.append((t_ms / 1000.0, hz))
        if cur:
            runs.append(cur)
        return runs

    def _load_reference_labels(self, sig_path, stem):
        """Rasterize the .fx contour onto a uniform 5 ms grid (interpolate within each voiced run,
        0 in the breaks); the base then resamples it onto the eval grid by these true times."""
        runs = self._read_fx(Path(sig_path).with_suffix(".fx"))
        t_max = max((run[-1][0] for run in runs), default=0.0)
        n = int(t_max / self.FX_STEP_S) + 1
        times = np.arange(n) * self.FX_STEP_S
        pitch_np = np.zeros(n, dtype=np.float32)
        for run in runs:
            ts = np.array([t for t, _ in run])
            hs = np.array([h for _, h in run])
            m = (times >= ts[0]) & (times <= ts[-1])        # gaps between runs stay 0 (unvoiced)
            pitch_np[m] = np.interp(times[m], ts, hs)
        pitch = torch.from_numpy(pitch_np)
        return pitch, (pitch > 0).float(), times
