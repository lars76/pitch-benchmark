"""Single source of truth for the benchmark's two resampling concerns:

  - resample_to_grid: a (pitch, periodicity) frame-label SERIES onto a target time grid (the label /
    prediction alignment used by the wrappers, loaders and consensus generator; documented below).
  - resample_audio: a raw audio WAVEFORM from one sample rate to another (the DSP step the fixed-rate
    fixed-rate neural trackers run before inference).

These are unrelated operations (label-time alignment vs signal sample-rate conversion); they live
together only so "resampling" has one home.

--- resample_to_grid ---

Resamples a (pitch, periodicity) frame series onto a target time grid. Shared by the prediction
wrappers (algorithms/base.py), the ground-truth loaders (datasets/base.py) and the consensus
generator (scripts/build_consensus_labels.py), so every part of the benchmark aligns frames the same
way.

The two arrays have different natures and use different strategies:

  - PITCH follows the mir_eval.melody convention: it never interpolates THROUGH the unvoiced 0
    sentinel (which fabricates boundary frequencies, e.g. an octave-low value at a voiced->unvoiced
    edge). It forward-fills the last voiced value across unvoiced gaps, interpolates in CENTS
    (log-Hz, perceptually uniform), then re-masks the frames unvoiced in the source.

  - VOICING is resampled NEAREST (zero-order hold) for a binary {0, 1} label, or LINEARLY for a
    continuous [0, 1] confidence (voicing_kind="linear"). Linear-then-threshold on a binary label
    would dilate voiced regions by ~1 frame at every boundary, so binary uses nearest.

Target times outside the source span are unvoiced (pitch 0, voicing 0). Pure numpy (no torch), so
it imports cleanly into the runtime, loaders and generation scripts alike.

Field-standard scheme: reproduces mir_eval.melody.resample_melody_series exactly. A no-op
for on-grid trackers whose native frame times equal the target grid, so only off-grid trackers (e.g. YAAPT) are affected.
"""
import numpy as np

_CENTS_REF_HZ = 10.0  # arbitrary cents reference; cancels out, kept consistent across the benchmark


def resample_to_grid(pitch, periodicity, source_times, target_times, voicing_kind="nearest"):
    """Resample (pitch, periodicity) sampled at `source_times` onto `target_times`.

    Args:
        pitch:        per-frame f0 in Hz, 0 on unvoiced frames.
        periodicity:  per-frame voicing; binary {0,1} label or continuous [0,1] confidence.
        source_times: the frames' true times in seconds (must be sorted ascending).
        target_times: the grid times in seconds to resample onto.
        voicing_kind: "nearest" (binary label, default) or "linear" (continuous confidence).

    Returns:
        (pitch_g, periodicity_g) as float64 numpy arrays of len(target_times).
    """
    f0 = np.asarray(pitch, dtype=np.float64).reshape(-1)
    per = np.asarray(periodicity, dtype=np.float64).reshape(-1)
    nt = np.asarray(source_times, dtype=np.float64).reshape(-1)
    g = np.asarray(target_times, dtype=np.float64).reshape(-1)
    # Contract: every source frame has a pitch, a voicing and a timestamp. Enforce it (rather than
    # silently truncating to the shortest) so a wrapper or loader that drops one is caught loudly.
    if not (len(f0) == len(per) == len(nt)):
        raise ValueError(
            f"resample_to_grid: pitch ({len(f0)}), periodicity ({len(per)}) and source_times "
            f"({len(nt)}) must describe the same frames (equal length)."
        )
    n = len(f0)
    if n == 0:                                   # no source frames -> all unvoiced
        return np.zeros(len(g)), np.zeros(len(g))

    # nearest source frame per target time (for voicing + the f0 re-mask)
    if n == 1:
        near = np.zeros(len(g), dtype=np.intp)
    else:
        j = np.clip(np.searchsorted(nt, g), 1, n - 1)
        near = np.where(np.abs(nt[j] - g) < np.abs(nt[j - 1] - g), j, j - 1)
    in_range = (g >= nt[0]) & (g <= nt[-1])

    # pitch: forward-fill across unvoiced gaps, interpolate in cents, re-mask (mir_eval melody)
    voiced = f0 > 0
    if voiced.any():
        idx = np.where(voiced, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)              # forward-fill last voiced index
        filled = f0[idx]
        filled[filled <= 0] = f0[voiced][0]              # leading gap (re-masked below)
        cents_g = np.interp(g, nt, 1200.0 * np.log2(filled / _CENTS_REF_HZ))
        pitch_g = np.where(voiced[near] & in_range, _CENTS_REF_HZ * 2.0 ** (cents_g / 1200.0), 0.0)
    else:
        pitch_g = np.zeros(len(g))

    # voicing: nearest (zero-order) for a binary label, linear for a continuous confidence
    if voicing_kind == "linear":
        per_g = np.where(in_range, np.interp(g, nt, per), 0.0)
    else:
        per_g = np.where(in_range, per[near], 0.0)

    return pitch_g, per_g


def model_hop_length(hop_size: int, sample_rate: int, model_sr: int) -> int:
    """The INTEGER hop actually available when a fixed-rate model strides audio resampled from
    ``sample_rate`` to ``model_sr``. int() truncation (not round) deliberately matches the striding
    code in the wrappers and in upstream libraries (e.g. torchcrepe.core.preprocess). What matters
    is that the SAME integer drives both the striding and the timestamps (frame_times): stamping
    with the nominal float hop instead drifts ~0.4%/frame at 22.05 kHz -> hop 256
    (the timestamp calibration catches this class)."""
    return max(1, int(hop_size * model_sr / sample_rate))


def frame_times(n_frames: int, hop_length: int, sample_rate: int):
    """Times (seconds) of ``n_frames`` frames strided ``hop_length`` samples apart at
    ``sample_rate``, frame 0 at t=0. The honest-stamp primitive shared by the fixed-rate model
    wrappers: times must come from the stride that actually produced the frames, never from a
    nominal hop it was truncated from. Seconds survive sample-rate conversion, so times computed
    on a model-rate timeline are directly valid on the original timeline."""
    return np.arange(n_frames) * (hop_length / sample_rate)


def resample_audio(audio, orig_sr, target_sr):
    """Resample a 1-D float audio waveform orig_sr -> target_sr (identity if equal). Prefers resampy;
    falls back to scipy if resampy is missing. Used by the fixed-rate neural trackers (CREPE, SPICE,
    RMVPE) to reach their model's native rate before inference. resampy/scipy are imported lazily so
    this module stays import-clean for the loaders and generation scripts that only need labels."""
    if orig_sr == target_sr:
        return audio
    try:
        from resampy import resample

        return resample(audio, orig_sr, target_sr)
    except ImportError:
        from scipy.signal import resample

        target_len = int(len(audio) * target_sr / orig_sr)
        return resample(audio, target_len).astype(np.float32)
