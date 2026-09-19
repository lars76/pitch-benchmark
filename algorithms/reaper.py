import ctypes
import os
import sys
from contextlib import contextmanager, suppress

import numpy as np

from .base import PCM16_MAX, ThresholdPitchAlgorithm, threshold_to_param

UNVOICED_COST_RANGE = (0.2, 1.6)


@contextmanager
def _silence_c_fds():
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
                libc.fflush(None)
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


class REAPERPitchAlgorithm(ThresholdPitchAlgorithm):

    def _extract_pitch_with_threshold(self, audio, threshold):
        import pyreaper

        x = np.round(np.clip(audio, -1.0, 1.0) * PCM16_MAX).astype(np.int16)
        unvoiced_cost = threshold_to_param(threshold, UNVOICED_COST_RANGE)
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
        pitch = np.where(f0 > 0, f0, 0.0)
        times = np.asarray(f0_times, dtype=float) + 0.5 * self.hop_size / self.sample_rate
        return times, pitch, (pitch > 0).astype(np.float32)

    def _get_default_threshold(self):
        return 0.225
