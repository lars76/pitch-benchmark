import ctypes
import os
import sys
from contextlib import contextmanager, suppress

import numpy as np

from .base import ThresholdPitchAlgorithm


@contextmanager
def _silence_c_fds():
    """Silence C-level stdout/stderr around the REAPER call. REAPER's EpochTracker prints
    'Residual symmetry ...' diagnostics straight to the file descriptors, so a Python-level
    redirect_stdout would not catch them. We redirect fds 1 and 2 to /dev/null AND flush the C
    stdio buffer while still redirected (REAPER's output is fully buffered when stdout is a pipe,
    so it would otherwise flush after the fds are restored and leak to the real stdout)."""
    try:
        libc = ctypes.CDLL(None)
    except Exception:
        libc = None
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        if libc is not None:
            with suppress(Exception):
                libc.fflush(None)  # flush C stdio (all streams) to /dev/null before restoring
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


class REAPERPitchAlgorithm(ThresholdPitchAlgorithm):
    """REAPER (Robust Epoch And Pitch EstimatoR, D. Talkin) via pyreaper.

    A DSP epoch/pitch tracker. Its voicing operating point is REAPER's own dynamic-programming
    `unvoiced_cost` (higher -> more voiced), so the benchmark's [0, 1] threshold maps onto that knob
    and REAPER is re-run per threshold, the parameter-based sweep used by RAPT/SWIPE/YAAPT. This
    gives a full bidirectional precision/recall sweep with an f0 available at every operating point
    (unlike thresholding the correlation, which can only make voicing more conservative because
    REAPER reports no f0 on frames it calls unvoiced).

    REAPER reports per-frame `f0_times`, but those mark frame STARTS, not the analysis center:
    the timestamp calibration (tests/test_time_calibration.py) measures the content of frame m
    exactly frame_period/2 after its stamp at both 16 kHz and 22.05 kHz, with the chirp and step
    probes agreeing to 0.1 ms. We therefore shift the stamps by +frame_period/2 to the centers and
    let the base class resample to the eval grid.

    Note (transparency): REAPER is also one of the four EGG estimators behind the PTDB consensus
    labels (scripts/build_consensus_labels.py, run on the laryngograph signal). As a mic tracker it is therefore
    scored partly against labels it helped create, a mild conflict it shares with Praat and Harvest,
    which are likewise both consensus generators and benchmark trackers.
    """

    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import pyreaper

        # REAPER requires int16 PCM; audio is validated to [-1, 1] upstream.
        x = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        # Map [0, 1] -> REAPER's unvoiced_cost (default 0.9; higher = more voiced). threshold 0.5 ->
        # 0.9, so the sweep brackets REAPER's documented operating point.
        unvoiced_cost = 0.2 + threshold * 1.4  # [0.2, 1.6]
        with _silence_c_fds():
            _pm_t, _pm, f0_times, f0, _corr = pyreaper.reaper(
                x,
                self.sample_rate,
                minf0=self.fmin,
                maxf0=self.fmax,
                frame_period=self.hop_size / self.sample_rate,
                unvoiced_cost=unvoiced_cost,
            )
        f0 = np.asarray(f0, dtype=float)
        pitch = np.where(f0 > 0, f0, 0.0)  # REAPER uses -1 on unvoiced frames
        # f0_times are frame starts; stamp the frame centers (see class docstring).
        times = np.asarray(f0_times, dtype=float) + 0.5 * self.hop_size / self.sample_rate
        return times, pitch, (pitch > 0).astype(np.float32)

    def _get_default_threshold(self) -> float:
        return 0.5  # -> unvoiced_cost 0.9, REAPER's documented default
