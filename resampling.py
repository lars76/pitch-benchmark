import numpy as np

CENTS_REF_HZ = 10.0


def resample_to_grid(pitch, periodicity, source_times, target_times, voicing_kind="nearest"):
    f0 = np.asarray(pitch, dtype=np.float64).reshape(-1)
    per = np.asarray(periodicity, dtype=np.float64).reshape(-1)
    nt = np.asarray(source_times, dtype=np.float64).reshape(-1)
    g = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if not (len(f0) == len(per) == len(nt)):
        raise ValueError(
            f"resample_to_grid: pitch ({len(f0)}), periodicity ({len(per)}) and source_times "
            f"({len(nt)}) must describe the same frames (equal length)."
        )
    n = len(f0)
    if n == 0:
        return np.zeros(len(g)), np.zeros(len(g))

    if n > 1 and not np.all(np.diff(nt) > 0):
        keep = np.append(np.diff(nt) > 0, True)
        f0, per, nt = f0[keep], per[keep], nt[keep]
        n = len(nt)
        if n > 1 and not np.all(np.diff(nt) > 0):
            raise ValueError(
                "resample_to_grid: source_times must not go backwards; coincident stamps "
                "collapse to the last of each run, but a decreasing one has no reading.")

    if n == 1:
        near = np.zeros(len(g), dtype=np.intp)
        right = np.zeros(len(g), dtype=np.intp)
    else:
        right = np.clip(np.searchsorted(nt, g), 1, n - 1)
        near = np.where(np.abs(nt[right] - g) < np.abs(nt[right - 1] - g), right, right - 1)
    left = np.maximum(right - 1, 0)

    voiced = f0 > 0
    if voiced.any():
        span = np.where(nt[right] > nt[left], nt[right] - nt[left], 1.0)
        w = np.clip((g - nt[left]) / span, 0.0, 1.0)
        safe = np.where(voiced, f0, CENTS_REF_HZ)
        c_left = 1200.0 * np.log2(safe[left] / CENTS_REF_HZ)
        c_right = 1200.0 * np.log2(safe[right] / CENTS_REF_HZ)
        inside = voiced[left] & voiced[right] & (nt[right] > nt[left])
        interpolated = CENTS_REF_HZ * 2.0 ** ((c_left + w * (c_right - c_left)) / 1200.0)
        pitch_g = np.where(voiced[near], np.where(inside, interpolated, f0[near]), 0.0)
    else:
        pitch_g = np.zeros(len(g))

    if voicing_kind == "linear":
        per_g = np.interp(g, nt, per)
    else:
        per_g = per[near]

    return pitch_g, per_g


def model_hop_length(hop_size, sample_rate, model_sr):
    return max(1, int(hop_size * model_sr / sample_rate))


def frame_times(n_frames, hop_length, sample_rate):
    return np.arange(n_frames) * (hop_length / sample_rate)


class TimingProbe:

    HARMONICS = ((1, 1.0), (2, 0.5), (3, 0.25))
    FADE_SECONDS = 0.02
    PEAK = 0.9

    def __init__(self, sample_rate, f_lo, f_hi, leg_seconds=2.0, legs=4,
                 edge_seconds=0.15, vertex_seconds=0.10, min_frames_per_slope=30,
                 half_split_tol_ms=1.0, resid_limit_cents=100.0):
        self.sample_rate = int(sample_rate)
        self.f_lo, self.f_hi = float(f_lo), float(f_hi)
        self.leg_seconds, self.legs = float(leg_seconds), int(legs)
        self.edge_seconds, self.vertex_seconds = float(edge_seconds), float(vertex_seconds)
        self.min_frames_per_slope = int(min_frames_per_slope)
        self.half_split_tol_ms = float(half_split_tol_ms)
        self.resid_limit_cents = float(resid_limit_cents)
        self.seconds = self.leg_seconds * self.legs
        self.span_cents = 1200.0 * np.log2(self.f_hi / self.f_lo)
        self._audio = None

    @property
    def fmin(self):
        return self.f_lo / 2.0

    @property
    def fmax(self):
        return self.f_hi * 2.0

    def freq(self, t):
        pos = np.asarray(t, dtype=np.float64) / self.leg_seconds
        tri = np.abs(pos - 2.0 * np.floor(pos / 2.0 + 0.5))
        return self.f_lo * 2.0 ** (self.span_cents * tri / 1200.0)

    def slope(self, t):
        leg = np.floor(np.asarray(t, dtype=np.float64) / self.leg_seconds).astype(int)
        rate = self.span_cents / self.leg_seconds
        return np.where(leg % 2 == 0, rate, -rate)

    def audio(self):
        if self._audio is None:
            n = round(self.seconds * self.sample_rate)
            phase = (2.0 * np.pi / self.sample_rate) * np.cumsum(
                self.freq(np.arange(n) / self.sample_rate))
            x = sum(amp * np.sin(h * phase) for h, amp in self.HARMONICS)
            fade = int(self.FADE_SECONDS * self.sample_rate)
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
            self._audio = (x / np.abs(x).max() * self.PEAK).astype(np.float32)
        return self._audio

    def _usable(self, times, pitch):
        pos = times / self.leg_seconds
        return (np.isfinite(times) & np.isfinite(pitch) & (pitch > 0)
                & (times > self.edge_seconds)
                & (times < self.seconds - self.edge_seconds)
                & (np.abs(pos - np.round(pos)) * self.leg_seconds > self.vertex_seconds))

    def _fit(self, times, pitch):
        slope = self.slope(times)
        if min(int((slope > 0).sum()), int((slope < 0).sum())) < self.min_frames_per_slope:
            return None
        error = 1200.0 * np.log2(pitch / self.freq(times))
        error = (error + 600.0) % 1200.0 - 600.0
        basis = np.stack([np.ones_like(times), slope, slope * (times - times.mean())], axis=1)
        coef, *_ = np.linalg.lstsq(basis, error, rcond=None)
        resid = float((error - basis @ coef).std())
        if not np.all(np.isfinite(coef)) or not np.isfinite(resid):
            return None
        return (float(coef[1]), float(coef[2])) if resid <= self.resid_limit_cents else None

    def offset_ms(self, algo):
        times, pitch, _ = algo.native_frames(self.audio())
        times = np.asarray(times, dtype=np.float64)
        pitch = np.asarray(pitch, dtype=np.float64)
        keep = self._usable(times, pitch)
        times, pitch = times[keep], pitch[keep]
        whole = self._fit(times, pitch)
        if whole is None:
            return None
        middle = self.seconds / 2.0
        halves = [self._fit(times[m], pitch[m])
                  for m in (times < middle, times >= middle)]
        if any(h is None for h in halves):
            return None
        if abs(halves[0][0] - halves[1][0]) * 1e3 > self.half_split_tol_ms:
            return None
        offset, drift = whole
        reach = drift * self.seconds / 2.0
        return max(offset + reach, offset - reach, key=abs) * 1e3
