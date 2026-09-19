import numpy as np

from constants import VOICED_THRESHOLD

CENTS_REF_HZ = 10.0


def is_voiced(periodicity):
    return periodicity >= VOICED_THRESHOLD


def frame_times(n_frames, hop_size, sample_rate):
    return np.arange(n_frames) * (hop_size / sample_rate)


def cents(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(a / b)


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
