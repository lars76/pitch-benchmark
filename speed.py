import argparse
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

import score
from algorithms import build_algorithm, get_algorithm, list_algorithms
from run import PIN_VARS

STARTUP_ALLOWANCE_S = 600.0
N_RUNS = 5
SIGNAL_SECONDS = 10.0
MIN_BATCH_SECONDS = 0.25
THREAD_POLICY = "library defaults; thread-limit environment variables cleared"
SLOWEST_EXPECTED_RTF = 3.0


def generate_harmonic_signal(sample_rate, seconds, f0_hz=440.0, n_harmonics=3):
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = np.zeros_like(t)
    for harmonic in range(1, n_harmonics + 1):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0_hz * harmonic * t)
    signal = signal.astype(np.float32)
    return signal / np.abs(signal).max()


def time_algorithm(algorithm_class, sample_rate=16000, hop_size=256, seconds=SIGNAL_SECONDS,
                   n_runs=N_RUNS, fmin=65.0, fmax=1200.0):
    audio = generate_harmonic_signal(sample_rate, seconds)
    run_ms, batch_calls, error = [], [], None
    try:
        algo = build_algorithm(algorithm_class, sample_rate, hop_size, fmin, fmax)
        algo.extract_pitch(audio)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            calls, elapsed = 0, 0.0
            while elapsed < MIN_BATCH_SECONDS:
                algo.extract_pitch(audio)
                calls += 1
                elapsed = time.perf_counter() - t0
            run_ms.append(elapsed * 1e3 / calls)
            batch_calls.append(calls)
    except Exception as e:
        run_ms, batch_calls, error = [], [], f"{type(e).__name__}: {e}"
    return ({"sample_rate": sample_rate, "hop_size": hop_size, "signal_seconds": seconds,
             "n_runs": n_runs, "fmin": fmin, "fmax": fmax, "cpu": cpu_name(),
             "min_batch_seconds": MIN_BATCH_SECONDS, "thread_policy": THREAD_POLICY,
             "signal_type": "harmonic", "fundamental_hz": 440.0, "harmonics": [1, 2, 3]},
            {"run_ms": run_ms, "batch_calls": batch_calls, "error": error})


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def write_speed_cell(path, *, algo, parameters, results):
    score.write_json(path, {
        "metadata": {"kind": "speed", "algorithm_name": algo},
        "parameters": parameters,
        "results": results})


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="cells")
    p.add_argument("--algorithms", nargs="+", default=None)
    p.add_argument("--child", metavar="ALGO", help=argparse.SUPPRESS)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.child:
        cls = get_algorithm(args.child, fail_silently=True)
        parameters, results = (time_algorithm(cls) if cls is not None
                               else ({}, {"run_ms": [], "error": "not installed"}))
        write_speed_cell(os.path.join(args.out, f"speed_{args.child}.json"), algo=args.child,
                         parameters=parameters, results=results)
        return

    algos = args.algorithms or list_algorithms()
    timeout = STARTUP_ALLOWANCE_S + (1 + N_RUNS) * SIGNAL_SECONDS * SLOWEST_EXPECTED_RTF
    for i, algo in enumerate(algos):
        env = {k: v for k, v in os.environ.items() if k not in PIN_VARS}
        env["TQDM_DISABLE"] = "1"
        path = os.path.join(args.out, f"speed_{algo}.json")
        if os.path.exists(path):
            os.remove(path)
        t0 = time.time()
        kind = None
        try:
            r = subprocess.run([sys.executable, __file__, "--out", args.out, "--child", algo],
                               env=env, timeout=timeout,
                               capture_output=True, text=True)
            if r.returncode != 0:
                kind = f"exit {r.returncode}"
        except subprocess.TimeoutExpired:
            kind = f"timeout > {timeout:.0f}s"
        if not os.path.exists(path):
            write_speed_cell(path, algo=algo, parameters={},
                             results={"run_ms": [], "error": kind or "no output"})
        print(f"{i + 1:>2}/{len(algos)}  {algo:<12} {kind or 'ok':<12} "
              f"{time.time() - t0:6.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
