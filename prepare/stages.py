import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from constants import (
    ACTIVE_CUT_DB,
    ACTIVE_HOP_S,
    ACTIVE_WIN_S,
    CARRIER,
    DIRECT_PATH_DB,
    F_HI_HZ_LOG_RANGE,
    F_LO_HZ,
    HP_ORDER_CHOICES,
    LEVEL_PEAK_DBFS,
    LP_ORDER_CHOICES,
    PANEL_STAGES,
    POWER_FLOOR,
    RMS_FLOOR,
    SIR_STRATA,
    SOURCE_SLOTS,
    STAGE_ORDER,
)
from grid import is_voiced
from seeds import corpus_uid, item_rng

MIN_CLIP_POWER = 1e-8
SIR_SOLVE_ITERS = 6

DOUBLE_PASS_HALF_POWER = np.sqrt(2.0) - 1.0
FREQ_CLIP_EPS = 1e-9
NORM_FREQ_MIN, NORM_FREQ_MAX = 1e-4, 0.999
CORNER_GAP = 1e-4


def signal_rms(x):
    return float(np.sqrt(np.mean(x ** 2) + POWER_FLOOR))


def direct_path_index(h):
    e = np.abs(np.asarray(h, np.float64))
    peak = float(e.max()) if len(e) else 0.0
    if peak <= 0.0:
        return 0
    return int(np.argmax(e > 10 ** (DIRECT_PATH_DB / 20.0) * peak))


def voiced_power(audio, periodicity, hop):
    if periodicity is not None:
        per = is_voiced(np.asarray(periodicity).reshape(-1))
        if np.any(per):
            frame = np.minimum((np.arange(len(audio)) + hop // 2) // hop, len(per) - 1)
            mask = per[frame]
            if mask.any():
                return float(np.mean(audio[mask] ** 2) + POWER_FLOOR)
    return signal_rms(audio) ** 2


def active_power(seg, sample_rate):
    seg = np.asarray(seg)
    w = int(ACTIVE_WIN_S * sample_rate)
    h = int(ACTIVE_HOP_S * sample_rate)
    if len(seg) < w:
        return float(np.mean(seg ** 2))
    pw = np.array([np.mean(seg[i:i + w] ** 2)
                   for i in range(0, len(seg) - w + 1, h)])
    db = 10 * np.log10(pw + POWER_FLOOR)
    active = pw[db >= db.max() - ACTIVE_CUT_DB]
    if not active.size:
        return float(np.mean(seg ** 2))
    return float(np.mean(active))


def stage_salt(stage):
    return STAGE_ORDER.index(stage)


def convolve_aligned(x, h):
    x = np.asarray(x, np.float64)
    h = np.asarray(h, np.float64)
    p = direct_path_index(h)
    y = fftconvolve(x, h)
    return y[p:p + len(x)]


def _match_level_gain(y, ref):
    r = float(np.sqrt(np.mean(np.asarray(ref, np.float64) ** 2)))
    ry = float(np.sqrt(np.mean(np.asarray(y, np.float64) ** 2)))
    return 1.0 if (r <= RMS_FLOOR or ry <= RMS_FLOOR) else r / ry


def double_pass_corner(f_target, sample_rate, n, highpass):
    k = DOUBLE_PASS_HALF_POWER ** (1.0 / (2 * n))
    omega = np.tan(np.pi * np.clip(f_target / sample_rate, FREQ_CLIP_EPS, 0.5 - FREQ_CLIP_EPS))
    omega = omega * k if highpass else omega / k
    return float(sample_rate / np.pi * np.arctan(omega))


def micband_filter(y, sample_rate, f_lo, f_hi, hp_order, lp_order):
    nyq = sample_rate / 2.0
    n_hi = max(int(hp_order) // 2, 1)
    n_lo = max(int(lp_order) // 2, 1)
    c_lo = double_pass_corner(f_lo, sample_rate, n_hi, True)
    c_hi = double_pass_corner(f_hi, sample_rate, n_lo, False)
    w_lo = float(np.clip(c_lo / nyq, NORM_FREQ_MIN, NORM_FREQ_MAX))
    w_hi = float(np.clip(c_hi / nyq, w_lo + CORNER_GAP, NORM_FREQ_MAX))
    hp = butter(n_hi, w_lo, btype="high", output="sos")
    lp = butter(n_lo, w_hi, btype="low", output="sos")
    z = sosfilt(hp, sosfilt(hp, y))
    return sosfilt(lp, sosfilt(lp, z))


def _draw_scene_source(scene, rng, label_safe_only):
    entries = ([s for s in scene if s.label_safe]
               if label_safe_only else list(scene))
    if not entries:
        raise RuntimeError("scene has no admissible source")
    w = np.array([s.weight for s in entries], np.float64)
    return entries[int(rng.choice(len(entries), p=w / w.sum()))]


def _sir_db(t_mic, i_mic, sample_rate, per, hop, has_voice):
    ref = float(np.sqrt(np.mean(np.asarray(t_mic, np.float64) ** 2)))
    if ref > 0.0:
        t_mic, i_mic = t_mic / ref, i_mic / ref
    target_power = (voiced_power(np.asarray(t_mic, np.float32), per, hop) if has_voice
                    else active_power(t_mic, sample_rate))
    interferer_power = active_power(i_mic, sample_rate)
    if target_power <= POWER_FLOOR or interferer_power <= POWER_FLOOR:
        return float("nan")
    return 10.0 * np.log10(target_power / interferer_power)


def render_paths(target, parts, gain, rir, mic, sample_rate):
    point = sum((p[0] for p in parts if p[1]), np.zeros_like(target))
    diffuse = sum((p[0] for p in parts if not p[1]), np.zeros_like(target))
    if rir is None:
        t_pre, i_pre = target, gain * (point + diffuse)
    else:
        dry = target + gain * point
        wet = convolve_aligned(dry, rir)
        alpha = _match_level_gain(wet, dry)
        t_pre = alpha * convolve_aligned(target, rir)
        i_pre = alpha * convolve_aligned(gain * point, rir) + gain * diffuse
    y = t_pre + i_pre
    if mic is not None:
        filtered = micband_filter(y, sample_rate, *mic)
        beta = _match_level_gain(filtered, y)
        t_pre = beta * micband_filter(t_pre, sample_rate, *mic)
        i_pre = beta * micband_filter(i_pre, sample_rate, *mic)
        y = beta * filtered
    return y, t_pre, i_pre


def _solve_scene_gain(target, parts, rir, mic, sample_rate, per, hop, has_voice, sir):
    gain = 1.0
    for _ in range(SIR_SOLVE_ITERS):
        _y, t_mic, i_mic = render_paths(target, parts, gain, rir, mic, sample_rate)
        realized = _sir_db(t_mic, i_mic, sample_rate, per, hop, has_voice)
        if not np.isfinite(realized):
            return 0.0
        gain *= 10.0 ** ((realized - sir) / 20.0)
    return gain


def render_clip(x, sample_rate, periodicity, hop, *, panel, corpus, clip_idx,
                master_seed, scene=None, rooms=None):
    stages = PANEL_STAGES[panel]
    uid = corpus_uid(corpus)
    target = np.asarray(x, np.float64).reshape(-1)
    render_record = {}
    if stages:
        render_record["clip_uid"] = int(clip_idx)
        render_record["n_frames"] = (len(np.asarray(periodicity).reshape(-1))
                            if periodicity is not None else len(target) // hop)
        render_record["target_rms_dbfs"] = round(20 * np.log10(signal_rms(target) + RMS_FLOOR), 2)
    parts = []
    per = (np.asarray(periodicity).reshape(-1) if periodicity is not None else None)
    has_voice = per is None or bool(np.any(is_voiced(per)))
    sir = 0.0

    if "scene" in stages:
        if scene is None:
            raise ValueError(f"panel {panel!r} needs a scene")
        rng = item_rng(master_seed, uid, clip_idx, stage_salt("scene"))
        k = 1 + int(rng.integers(SOURCE_SLOTS))
        for _ in range(k):
            src = _draw_scene_source(scene, rng, label_safe_only=not has_voice)
            out, extra = src.generator(len(target), rng)
            meta = {"src": src.name, **extra}
            seg = np.asarray(out, np.float64)
            npow = active_power(seg, sample_rate)
            seg = seg / np.sqrt(npow) if npow > POWER_FLOOR else seg * 0.0
            parts.append([seg, src.point, meta])
        stratum, band = SIR_STRATA[int(rng.integers(len(SIR_STRATA)))]
        sir = float(rng.uniform(*band))
        render_record["k"] = k
        render_record["scene"] = {"stratum": stratum, "sir_db": round(sir, 2)}

    room_on = "room" in stages
    if room_on and rooms is None:
        raise ValueError(f"panel {panel!r} needs rooms")

    rir = None
    if room_on:
        rir, rir_id = rooms.draw(item_rng(master_seed, uid, clip_idx, stage_salt("room")))
        render_record["room"] = {"rir": rir_id}
    mic = None
    if "mic" in stages:
        rng = item_rng(master_seed, uid, clip_idx, stage_salt("mic"))
        f_lo = float(F_LO_HZ)
        f_hi = float(np.exp(rng.uniform(*np.log(F_HI_HZ_LOG_RANGE))))
        hp_order = int(HP_ORDER_CHOICES[int(rng.integers(len(HP_ORDER_CHOICES)))])
        lp_order = int(LP_ORDER_CHOICES[int(rng.integers(len(LP_ORDER_CHOICES)))])
        mic = (f_lo, f_hi, hp_order, lp_order)
        render_record["mic"] = {"f_lo_hz": round(f_lo, 1), "f_hi_hz": round(f_hi, 1),
                       "hp_order": hp_order, "lp_order": lp_order}

    gain = 0.0
    scale = float(np.sqrt(np.mean(target ** 2))) if len(target) else 0.0
    if parts and scale > 0.0:
        unit_gain = _solve_scene_gain(target / scale, parts, rir, mic, sample_rate,
                                      per, hop, has_voice, sir)
        gain = scale * unit_gain
        render_record["scene"]["gain_db"] = round(20 * np.log10(gain + RMS_FLOOR), 2)
        y, _t_mic, _i_mic = render_paths(target / scale, parts, unit_gain, rir, mic, sample_rate)
        y, _t_mic, _i_mic = y * scale, _t_mic * scale, _i_mic * scale
    else:
        y, _t_mic, _i_mic = render_paths(target, parts, gain, rir, mic, sample_rate)
    if parts:
        render_record["sources"] = [part[2] for part in parts]
        render_record["scene"]["realized_sir_db"] = round(
            _sir_db(_t_mic, _i_mic, sample_rate, per, hop, has_voice), 2)

    if "level" in stages:
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 0:
            y = y / peak * 10 ** (LEVEL_PEAK_DBFS / 20)
        render_record["level_peak_dbfs"] = LEVEL_PEAK_DBFS

    y = np.asarray(y, np.float64)
    if not np.isfinite(y).all():
        raise ValueError(f"panel {panel!r} rendered clip {clip_idx} of {corpus} to non-finite "
                         f"samples (a non-finite scene source?)")
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        raise ValueError(
            f"panel {panel!r} rendered clip {clip_idx} of {corpus} to peak {peak:.4f} > 1: "
            f"unreachable through the benchmark, where every scored panel ends in the "
            f"{CARRIER[0]!r} stage; a panel without it is unbounded, and rescaling here "
            f"would silently break the stage algebra")
    if stages:
        render_record["realized_rms_dbfs"] = round(20 * np.log10(signal_rms(y) + RMS_FLOOR), 2)
    return np.ascontiguousarray(y, dtype=np.float32), render_record
