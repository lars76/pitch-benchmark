import argparse
import json
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
N_ROUNDS = 5
ROUND_SEED = 0
SIGNAL_SECONDS = 10.0
MIN_BATCH_SECONDS = 2.0
THREAD_POLICY = ("one core: thread-limit environment variables set to 1, ONNX Runtime sessions built with one "
                 "thread, process pinned to one core; speed from process CPU time")
SLOWEST_EXPECTED_RTF = 10.0


def generate_harmonic_signal(sample_rate, seconds, f0_hz=440.0, n_harmonics=3):
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = np.zeros_like(t)
    for harmonic in range(1, n_harmonics + 1):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0_hz * harmonic * t)
    signal = signal.astype(np.float32)
    return signal / np.abs(signal).max()


def time_round(algorithm_class, core, sample_rate=16000, hop_size=256, seconds=SIGNAL_SECONDS,
               fmin=65.0, fmax=1200.0):
    os.sched_setaffinity(0, {core})
    audio = generate_harmonic_signal(sample_rate, seconds)
    try:
        algo = build_algorithm(algorithm_class, sample_rate, hop_size, fmin, fmax)
        algo.extract_pitch(audio)
        wall0, cpu0 = time.perf_counter(), time.process_time()
        calls = 0
        while time.perf_counter() - wall0 < MIN_BATCH_SECONDS:
            algo.extract_pitch(audio)
            calls += 1
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"run_ms": cpu * 1e3 / calls, "wall_ms": wall * 1e3 / calls, "batch_calls": calls}


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cpu_governor(core):
    try:
        with open(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError:
        return None


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
    p.add_argument("--round-out", help=argparse.SUPPRESS)
    p.add_argument("--core", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.child:
        cls = get_algorithm(args.child, fail_silently=True)
        result = time_round(cls, args.core) if cls is not None else {"error": "not installed"}
        score.write_json(args.round_out, result)
        return

    algos = args.algorithms or list_algorithms()
    if not hasattr(os, "sched_setaffinity"):
        p.error("speed.py pins every tracker to one CPU core, which needs Linux")
    core = max(os.sched_getaffinity(0))
    timeout = STARTUP_ALLOWANCE_S + 3 * SIGNAL_SECONDS * SLOWEST_EXPECTED_RTF
    rounds = {algo: [] for algo in algos}
    rng = np.random.default_rng(ROUND_SEED)
    for r in range(N_ROUNDS):
        order = list(algos)
        rng.shuffle(order)
        for i, algo in enumerate(order):
            env = {**os.environ, "TQDM_DISABLE": "1", **dict.fromkeys(PIN_VARS, "1")}
            round_path = os.path.join(args.out, f".speed_{algo}.round.json")
            if os.path.exists(round_path):
                os.remove(round_path)
            cmd = [sys.executable, __file__, "--out", args.out, "--child", algo,
                   "--round-out", round_path, "--core", str(core)]
            t0 = time.time()
            kind = None
            try:
                proc = subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
                if proc.returncode != 0:
                    kind = f"exit {proc.returncode}"
            except subprocess.TimeoutExpired:
                kind = f"timeout > {timeout:.0f}s"
            if os.path.exists(round_path):
                with open(round_path) as f:
                    result = json.load(f)
                os.remove(round_path)
            else:
                result = {"error": kind or "no output"}
            rounds[algo].append(result)
            print(f"round {r + 1}/{N_ROUNDS} {i + 1:>2}/{len(order)}  {algo:<12} "
                  f"{result.get('error') or 'ok':<12} {time.time() - t0:6.1f}s",
                  file=sys.stderr, flush=True)

    parameters = {"sample_rate": 16000, "hop_size": 256, "signal_seconds": SIGNAL_SECONDS,
                  "fmin": 65.0, "fmax": 1200.0, "cpu": cpu_name(), "core": core, "governor": cpu_governor(core),
                  "rounds": N_ROUNDS, "round_seed": ROUND_SEED,
                  "min_batch_seconds": MIN_BATCH_SECONDS, "thread_policy": THREAD_POLICY,
                  "signal_type": "harmonic", "fundamental_hz": 440.0,
                  "harmonics": [1, 2, 3]}
    for algo, results in rounds.items():
        ok = [res for res in results if "error" not in res]
        errors = [res["error"] for res in results if "error" in res]
        write_speed_cell(os.path.join(args.out, f"speed_{algo}.json"), algo=algo,
                         parameters=parameters,
                         results={key: [res[key] for res in ok]
                                  for key in ("run_ms", "wall_ms", "batch_calls")}
                         | {"error": errors[0] if errors else None})


if __name__ == "__main__":
    main()
