"""The speed track's per-cell measurement (a pure library; evaluate.py orchestrates).

run_speed_cell times ONE algorithm on the synthetic harmonic signal; it writes nothing.
Timing depends on machine state, which is why evaluate always overwrites speed cells and runs
them serially, never under a busy worker pool.
"""
import gc
import time

import numpy as np
import torch
from tqdm import tqdm

from algorithms import build_algorithm


def generate_harmonic_signal(sample_rate: int, duration: float) -> np.ndarray:
    """Generate a test signal with a fundamental frequency and harmonics."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.zeros_like(t)

    for harmonic in range(1, 4):
        amplitude = 1.0 / harmonic
        signal += amplitude * np.sin(2 * np.pi * 440 * harmonic * t)

    signal = signal.astype(np.float32)
    signal = signal / np.abs(signal).max()

    return signal


def benchmark_algorithm(
    algorithm_class: type,
    audio_signal: np.ndarray,
    sample_rate: int,
    hop_length: int,
    device: str,
    n_runs: int,
) -> float:
    """Benchmark a single pitch detection algorithm."""
    if algorithm_class.resolve_effective_device(device) != device:
        # The tracker would silently run on a different device (DSP -> cpu, torch/tf fallback);
        # record "unsupported" rather than timing one device under another device's label.
        return float("inf")

    def _sync():
        # Only synchronize a backend that is actually present; requesting an absent one raises.
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    try:
        algorithm = build_algorithm(algorithm_class, sample_rate, hop_length, 50.0, 1000.0, device=device)
        _sync()
        latencies = []
        for _ in range(n_runs):
            start_time = time.time()
            algorithm.extract_pitch(audio_signal)
            _sync()
            latencies.append(time.time() - start_time)
        # Same teardown as the accuracy tracks: this runs inside the orchestrator process, so a
        # big model must not linger after its timing is done.
        del algorithm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return sum(latencies) / len(latencies)
    except Exception as e:
        # A crashing tracker must not abort the whole speed run; record no measurement (rendered like
        # an unsupported device) and move on. Speed has no accuracy score, so there is no "count as 0".
        tqdm.write(f"  ({algorithm_class.get_name()} crashed on {device}: {e}), no timing recorded")
        return float("inf")


def run_speed_cell(algorithm_class, *, devices, sample_rate=22050, hop_length=256,
                   signal_length_sec=1.0, n_runs=10):
    """Time ONE algorithm on the synthetic harmonic signal across `devices` (pure measurement;
    evaluate.py owns writing and the always-overwrite policy). Returns {"device_results"}."""
    audio_signal = generate_harmonic_signal(sample_rate, signal_length_sec)
    algo_data = {"device_results": {}}
    for device in devices:
        latency = benchmark_algorithm(
            algorithm_class, audio_signal, sample_rate, hop_length, device, n_runs)
        # absolute_time_ms is None exactly when the device is unsupported, so a separate
        # "supported" flag would be a second encoding of the same fact
        supported = latency != float("inf")
        algo_data["device_results"][device] = {
            "absolute_time_ms": float(latency * 1000) if supported else None,
        }
    return algo_data
