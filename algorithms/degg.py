"""Differentiated-EGG (dEGG) glottal-closure peak-picking: a period-marking f0 estimator that runs on
the EGG (laryngograph) signal, NOT on a microphone waveform.

The EGG tracks vocal-fold contact area; its derivative has one sharp peak per cycle at the
glottal-closure instant (GCI), and f0 = sr / (inter-GCI interval). A radiated mic signal has no single
sharp closure peak per cycle (formant ringing), so this method is meaningful ONLY on the EGG. It is
therefore deliberately left OUT of algorithms/__init__::_ALGORITHM_METADATA: it must never appear as a
selectable benchmark tracker. scripts/build_consensus_labels.py imports it directly as the
period-marking family of the cross-family consensus ground truth. Ref: Henrich et al. 2004.

It returns its per-cycle f0 SAMPLE-AND-HELD onto the eval frame grid, deliberately NOT routed through
resampling.resample_to_grid. dEGG's per-cycle f0 is jittery (period-doubling that passes the range
gate); interpolating between cycle midpoints smooths that away and moves a large fraction of frames.
Returning the track already on-grid (times == the eval grid) makes the base class's resample_to_grid an
exact no-op, so the hold semantics survive. CHUNK_SECONDS is disabled so the whole-file amplitude
reference / adaptive refractory and the on-grid identity hold for any clip length (the method is O(n)
numpy with no large activation buffer, so there is no memory reason to chunk).
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .base import ContinuousPitchAlgorithm


def _parabolic(d: np.ndarray, pk: np.ndarray) -> np.ndarray:
    """Refine integer peak indices to sub-sample positions via parabolic-vertex interpolation
    (removes sample-grid f0 quantisation)."""
    pk = pk[(pk > 0) & (pk < len(d) - 1)]
    y0, y1, y2 = d[pk - 1], d[pk], d[pk + 1]
    denom = y0 - 2 * y1 + y2
    delta = np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0)
    return pk + np.clip(delta, -0.5, 0.5)


class DEGGPitchAlgorithm(ContinuousPitchAlgorithm):
    # Whole-file statistics (90th-percentile peak height, median inter-GCI spacing) and the on-grid
    # no-op that preserves the per-cycle hold both require the whole signal in one pass, so chunking is
    # disabled (the method is O(n) numpy with no large activation buffer, so there is no memory reason to chunk).
    CHUNK_SECONDS = None

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sr, fmin, fmax = self.sample_rate, self.fmin, self.fmax
        times = self._compute_target_times(len(audio))   # return on-grid -> base resample is a no-op

        x = np.asarray(audio, dtype=float)
        x = x - np.mean(x)
        nyq = 0.5 * sr
        b, a = butter(2, [max(1.0, 0.5 * fmin) / nyq, min(0.95, 3.0 * fmax / nyq)], btype="band")
        d = np.diff(filtfilt(b, a, x))
        if np.mean(d ** 3) < 0:                       # orient: sharp closure peaks positive
            d = -d
        pos = d[d > 0]
        if pos.size == 0:                             # silence/noise -> no peaks, all unvoiced
            return times, np.zeros(len(times)), np.zeros(len(times), dtype=np.float32)
        ref = np.percentile(pos, 90)                  # robust amplitude of true closure peaks
        dist = max(1, int(sr / fmax))
        gci, _ = find_peaks(d, distance=dist, height=0.30 * ref, prominence=0.20 * ref)
        if len(gci) >= 4:                             # adaptive refractory: drop within-cycle
            dist = max(dist, int(0.62 * np.median(np.diff(gci))))  # secondary peaks
            gci, _ = find_peaks(d, distance=dist, height=0.30 * ref, prominence=0.20 * ref)
        gci = _parabolic(d, gci)                      # sub-sample peak refinement

        f0 = np.zeros(len(times))
        voiced = np.zeros(len(times), dtype=bool)
        if len(gci) >= 2:
            cyc = sr / np.diff(gci)                   # one f0 per inter-GCI interval
            ok = (cyc >= fmin) & (cyc <= fmax)        # validity gate: implied f0 in band
            interval = np.searchsorted(gci, times * sr) - 1   # sample-and-hold per-cycle f0 (no smoothing)
            inb = (interval >= 0) & (interval < len(cyc))
            sel = interval[inb]
            f0[inb] = np.where(ok[sel], cyc[sel], 0.0)
            voiced[inb] = ok[sel]
        return times, f0, voiced.astype(np.float32)
