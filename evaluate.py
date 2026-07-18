#!/usr/bin/env python3
"""THE benchmark: the one entry point, the one orchestrator.

The benchmark = every registered dataset x every condition x four tracks
(frame, note, ood, speed), evaluated as ATOMIC CELLS -- one (track, dataset-or-family,
condition, algorithm) unit each. The track modules (pitch/note/ood/speed_benchmark) are pure
measurement libraries; THIS module owns everything else: cell enumeration, dataset paths,
filenames, result envelopes, cache-as-done, crash recording, process isolation, and the worker
pool. If the question is "who spawns / who names / who caches / who writes", the answer is
always: evaluate.

Dataset location -- the user says where each dataset's files start, explicitly:
`paths={"PTDB": "/my/ptdb_dir", ...}` / `--data PTDB=/my/ptdb_dir` (the loader then reads its
corpus's documented structure from exactly there -- no searching). As a convenience, a dataset
WITHOUT an explicit path resolves to `<root>/<Name>` when --root is given (the README's
convention layout). SpeechSynth is self-contained (a committed synthetic corpus inside this
repo) and needs neither. The chime/demand noise banks resolve the same way under the names
`chime_home` / `DEMAND`.

One sizing decision: robustness (non-clean) frame cells default to the LEADERBOARD CAP of
30 clips / 10 s (`max_samples=30, max_seconds=10.0`) -- adequate for ranking DIFFERENT trackers
(gaps >= 0.04: paired CI ~ +-0.02 at n=30) and the only affordable mode across many trackers (a
full pass measures 0.5-60 CORE-HOURS per tracker). Override or uncap (0/None) as needed: full
cells are required for experiment verdicts (deltas of 0.005-0.02 need hundreds of clips) and
affordable for an experiment's 1-3 algorithms. Honesty is structural, not knob-based: capped
cells always carry the probe tag (filename + metadata.probe) and assert_full() rejects them, so
a verdict must explicitly uncap.

Execution policy (the whole table):
  frame/note   cache-as-done. workers=1: in-process, cells grouped by (dataset, condition)
               over ONE shared dataset instance (decode + degradation synthesis paid once per
               group -- measured as the only redundancy worth avoiding). workers>1: one child
               process per cell (isolation for free; resume can't livelock).
  ood          cache-as-done; cells ALWAYS run in child processes (the synthetic stimuli are
               exactly what makes some C-extension trackers SIGSEGV) -- except injected
               classes, which cannot exist in a fresh interpreter and run in-process.
  speed        always overwrites (timing depends on machine state; a stale cached number is
               worse than a re-run); always serial and in-process (timing under a busy pool
               would be noise).
A child that dies without writing its cell gets a crashed cell written by the orchestrator.

An algorithm IS a PitchAlgorithm subclass; a string is shorthand for a registry class,
resolved once at the entry. A custom class exists only in this process, so runs with one stay
in-process (no child processes).

    from evaluate import run_cells, load_cells, compare, assert_full
    cells = run_cells([MyTracker, "SwiftF0"], root="/data", datasets=["KEELE"],
                      max_samples=None, max_seconds=None)   # uncapped = verdict mode
    delta, lo, hi = compare(cells, "MyTracker", "SwiftF0", metric="voicing_f1")
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np
import torch
from tqdm import tqdm

import metrics
import note_benchmark
import ood_benchmark
import pitch_benchmark
import speed_benchmark
from algorithms import PitchAlgorithm, get_algorithm, get_available_algorithms
from datasets import (
    Augment,
    Truncate,
    build_pipeline,
    get_pitch_dataset,
    list_note_datasets,
    list_pitch_datasets,
    subset,
)
from datasets.augment import REGISTRY as CONDITION_REGISTRY

# ---------------------------------------------------------------------------- #
# THE BENCHMARK DEFINITION -- registered = benchmarked, nothing more to declare:
# datasets from datasets.list_pitch_datasets(), conditions from datasets.augment.REGISTRY,
# note membership from the provides_notes capability, tracks = the four measurement libraries.
# Opt out per run with --skip-datasets / --datasets; interpretation caveats (score-grade GT,
# train/test overlap) live on the dataset classes and in the report, not as exclusions here.
# Robustness (non-clean) frame cells default to the leaderboard cap of 30 clips / 10 s -- see
# run_cells(max_samples=, max_seconds=); capped cells are always tagged (filename +
# metadata.probe) and assert_full() rejects them, so a verdict must explicitly uncap.
# Everything below this section is execution machinery, not definition.
# ---------------------------------------------------------------------------- #
CONDITIONS = tuple(CONDITION_REGISTRY)
TRACKS = ("frame", "note", "ood", "speed")


# Data that ships committed inside this repo (no download, no user path needed).
BUNDLED = {"SpeechSynth": os.path.join(REPO, "datasets", "speechsynth.pt")}


def _data_dir(name, paths, root, required=True):
    """Where `name`'s files start: explicit path > BUNDLED > <root>/<name>. With required=False
    (the chime_home/DEMAND noise banks) an unresolved name is None -- an error only if a run
    actually requests that degradation."""
    if paths and name in paths:
        return paths[name]
    if name in BUNDLED:
        return BUNDLED[name]
    if root:
        return os.path.join(root, name)
    if not required:
        return None
    raise ValueError(
        f"no data location for {name}: pass paths={{'{name}': DIR}} / --data {name}=DIR, "
        f"or --root with the <root>/<Name> layout (see README)")


# ---------------------------------------------------------------------------- #
# Dataset build (the one sequence) -- used by the orchestrator and by tests that need
# custom-capped cells (e.g. an algorithm's own parity/regression tests).
# ---------------------------------------------------------------------------- #
def build_eval_dataset(dataset, data_dir, *, sample_rate=16000, hop_size=256, max_samples=None,
                       max_seconds=None, degradation=None, chime_dir=None, demand_dir=None,
                       seed=42):
    """load -> probe cap (even-stride subset) -> truncate -> degrade (Augment; skipped when
    degradation is None -- an empty pipeline would be a literal pass-through anyway). Returns
    (eval_dataset, is_probe)."""
    if degradation == "chime" and not chime_dir:
        raise ValueError("chime_dir is required for the chime degradation")
    if degradation == "demand" and not demand_dir:
        raise ValueError("demand_dir is required for the demand degradation")
    if degradation == "room":
        import importlib.util
        if importlib.util.find_spec("pyroomacoustics") is None:
            raise ValueError("the room degradation requires pyroomacoustics (run `uv sync`)")
    base = get_pitch_dataset(dataset)(
        root_dir=data_dir, sample_rate=sample_rate, hop_size=hop_size,
    )
    is_probe = bool(max_samples or max_seconds)
    if max_samples and max_samples < len(base):
        idxs = sorted({round(i) for i in np.linspace(0, len(base) - 1, max_samples)})
        base = subset(base, idxs)
    if max_seconds:
        base = Truncate(base, float(max_seconds))
    if degradation is None:
        return base, is_probe
    pipeline = build_pipeline(
        degradation, chime_dir=chime_dir, demand_dir=demand_dir, sample_rate=sample_rate,
    )
    return Augment(base, pipeline, seed=seed), is_probe


# ---------------------------------------------------------------------------- #
# Filenames, cache, atomic writes (readers route by metadata, never by filename)
# ---------------------------------------------------------------------------- #
def frame_cell_filename(dataset, algo, degradation, *, is_probe=False, max_samples=None,
                        max_seconds=None, sample_rate=16000, hop_size=256, device="cpu", seed=42):
    param_str = (
        f"{degradation}_"
        + (f"probe-n{max_samples}-t{max_seconds}_" if is_probe else "")
        + f"sr{int(sample_rate / 1000)}k_"
        + f"hop{hop_size}_"
        + device
    )
    return f"{dataset}_{algo}_{param_str}_seed{seed}.json"


def note_cell_filename(algo, dataset, condition, seed=42):
    return f"notes_{algo}_{dataset}_{condition}_seed{seed}.json"


def ood_cell_filename(algo, family, device):
    return (f"ood_{algo}_{family}_{device}_"
            f"sr{ood_benchmark.SR // 1000}k_hop{ood_benchmark.HOP}.json")


def speed_cell_filename(algo, sample_rate, hop_length, signal_length_sec, n_runs):
    return (f"speed_{algo}_sr{int(sample_rate / 1000)}k_hop{hop_length}_"
            f"len{signal_length_sec}s_runs{n_runs}.json")


def write_cell(result_path, metadata, parameters, results):
    """Temp file + atomic rename: a kill mid-write must not leave a truncated JSON that
    cache_skip treats as finished."""
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    obj = {"metadata": metadata, "parameters": parameters, "results": results}
    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metrics.to_json_safe(obj), f, indent=4)
    os.replace(tmp_path, result_path)


# ---------------------------------------------------------------------------- #
# Cell enumeration: the benchmark matrix as cells
# ---------------------------------------------------------------------------- #
def enumerate_cells(algos, *, datasets=None, conditions=None, tracks=TRACKS,
                    families=None, skip_datasets=()):
    """The benchmark matrix as atomic cells: dicts of (track, dataset, condition, algo).
    Enumeration is PURE structure -- cell sizing (the robustness cap) is an execution decision
    applied by run_cells. `datasets` / `conditions` / `families` narrow the matrix (for
    screens); narrowing is visible in the result coverage, never silent."""
    frame_ds = [d for d in (datasets or list_pitch_datasets()) if d not in skip_datasets]
    unknown = set(frame_ds) - set(list_pitch_datasets())
    if unknown:
        raise ValueError(f"not registered datasets: {sorted(unknown)}")
    conds = tuple(conditions or CONDITIONS)
    unknown = set(conds) - set(CONDITION_REGISTRY)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    fams = tuple(families or ood_benchmark.ALL_FAMILIES)
    unknown = set(fams) - set(ood_benchmark.ALL_FAMILIES)
    if unknown:
        raise ValueError(f"unknown ood families: {sorted(unknown)}")
    cells = []
    if "frame" in tracks:
        for c in conds:                                  # clean first (never capped)
            for d in frame_ds:
                for a in algos:
                    cells.append(dict(track="frame", dataset=d, condition=c, algo=a))
    if "note" in tracks:
        note_ds = [d for d in (datasets or list_note_datasets()) if d in list_note_datasets()]
        for d in note_ds:
            if d in skip_datasets:
                continue
            for a in algos:
                cells.append(dict(track="note", dataset=d, condition="clean", algo=a))
    if "ood" in tracks:
        for f in fams:
            for a in algos:
                cells.append(dict(track="ood", dataset=f, condition=None, algo=a))
    if "speed" in tracks:
        for a in algos:
            cells.append(dict(track="speed", dataset=None, condition=None, algo=a))
    return cells


def _cell_cap(cell, cap):
    """The one sizing rule: the run's cap applies exactly to robustness (non-clean) frame
    cells; clean and the other tracks always run full."""
    return cap if (cell["track"] == "frame" and cell["condition"] != "clean") else {}


def _algo_classes(algos):
    """The algorithm contract: an algorithm IS a PitchAlgorithm subclass; a string is shorthand
    for a registry class, resolved once here. Returns ({name: class_or_None}, any_custom).
    class None = a named backend that is not installed (executors record it as a crashed cell).
    any_custom = at least one class that the registry cannot resolve by name -- such a class
    exists only in THIS process, so those runs stay in-process (no child processes)."""
    cls_map, custom = {}, False
    for a in algos:
        if isinstance(a, type) and issubclass(a, PitchAlgorithm):
            name = a.get_name()
            cls_map[name] = a
            custom = custom or (get_algorithm(name, fail_silently=True) is not a)
        else:
            cls_map[a] = get_algorithm(a, fail_silently=True)
    return cls_map, custom


# ---------------------------------------------------------------------------- #
# Cell execution
# ---------------------------------------------------------------------------- #
def run_and_write_frame_cell(eval_dataset, cls, *, out_dir, dataset, condition, is_probe,
                             max_samples, max_seconds, seed, sample_rate, hop_size, device,
                             algo_name=None):
    """Execute ONE frame cell on an already-built dataset and write it (skip if cached).
    `cls` is the algorithm class (None = an uninstalled named backend -> crashed cell;
    `algo_name` then supplies the name). Also the reference path for external parity tests that
    need custom-capped cells."""
    algo_name = algo_name or (cls.get_name() if cls is not None else "unknown")
    if cls is None:
        print(f"FATAL: {algo_name} is not installed -- recording as crashed")
        eff = device
    else:
        eff = cls.resolve_effective_device(device)
    path = os.path.join(out_dir, frame_cell_filename(
        dataset, algo_name, condition, is_probe=is_probe, max_samples=max_samples,
        max_seconds=max_seconds, sample_rate=sample_rate, hop_size=hop_size,
        device=eff, seed=seed))
    if os.path.exists(path):   # cache-as-done
        print(f"[frame] skip {os.path.basename(path)} (exists; delete to redo)")
        return path
    t0 = time.time()
    if cls is None:
        result, crashed = metrics.to_json_safe(pitch_benchmark._failure_dict(0)), True
    else:
        result, crashed = pitch_benchmark.run_single_evaluation(
            dataset=eval_dataset, algorithm_class=cls,
            thresholds=metrics.DEFAULT_THRESHOLDS, device=device)
    write_cell(
        path,
        metadata={
            "algorithm_name": algo_name, "dataset_name": dataset, "condition": condition,
            "probe": is_probe, "seed": seed, "device": eff, "crashed": crashed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": round(time.time() - t0, 2),
        },
        parameters={
            "sample_rate": sample_rate, "hop_size": hop_size,
            "max_samples": max_samples, "max_seconds": max_seconds,
            "fmin": eval_dataset.fmin, "fmax": eval_dataset.fmax,
        },
        results=result,
    )
    score, thr = result.get("combined_score"), result.get("optimal_threshold")
    status = "CRASHED" if crashed else (
        f"score={score:.4f} @ thr={thr:.2f}" if score is not None and thr is not None else "empty")
    print(f"[frame] {dataset}/{condition} {algo_name}: {status} "
          f"({time.time() - t0:.1f}s) -> {os.path.basename(path)}")
    return path


def _run_frame_cells(cells, *, paths, root, out_dir, device, seed, run_cap, cls_map):
    """Frame cells grouped by (dataset, condition): build the dataset ONCE per group and run the
    group's algorithms over it (decode + degradation synthesis shared)."""
    groups = {}
    for c in cells:
        groups.setdefault((c["dataset"], c["condition"]), []).append(c)
    for (ds, cond), group in groups.items():
        cap = _cell_cap(group[0], run_cap)
        if all(os.path.exists(_expected_path(c, out_dir, device, seed, cap,
                                             cls_map[c["algo"]])) for c in group):
            for c in group:
                print(f"[frame] skip {ds}/{cond}/{c['algo']} (cached)")
            continue
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"[frame] {ds}/{cond}: {len(group)} algorithms (seed {seed})")
        eval_ds, is_probe = build_eval_dataset(
            ds, _data_dir(ds, paths, root), degradation=cond, seed=seed,
            chime_dir=_data_dir("chime_home", paths, root, required=False),
            demand_dir=_data_dir("DEMAND", paths, root, required=False), **cap)
        for c in group:
            run_and_write_frame_cell(
                eval_ds, cls_map[c["algo"]], out_dir=out_dir, dataset=ds, condition=cond,
                is_probe=is_probe, max_samples=cap.get("max_samples"),
                max_seconds=cap.get("max_seconds"), seed=seed,
                sample_rate=16000, hop_size=256, device=device, algo_name=c["algo"])


def _run_note_cells(cells, *, paths, root, out_dir, device, seed, cls_map):
    """Note cells grouped by dataset (always clean, never capped): one shared dataset build."""
    groups = {}
    for c in cells:
        groups.setdefault(c["dataset"], []).append(c)
    for ds, group in groups.items():
        pending = [c for c in group if not os.path.exists(os.path.join(
            out_dir, note_cell_filename(c["algo"], ds, "clean", seed)))]
        for c in group:
            if c not in pending:
                print(f"[note] skip {ds}/{c['algo']} (exists; delete to redo)")
        if not pending:
            continue
        eval_ds, _ = build_eval_dataset(ds, _data_dir(ds, paths, root), seed=seed)
        thresholds = np.round(np.arange(0.0, 1.01, 0.1), 2)
        for c in pending:
            _run_note_cell(eval_ds, c["algo"], cls_map[c["algo"]], ds, "clean", thresholds,
                           out_dir, device, seed)


def _run_note_cell(eval_dataset, algo_name, cls, dataset, cond, thresholds, out_dir, device,
                   seed):
    path = os.path.join(out_dir, note_cell_filename(algo_name, dataset, cond, seed))
    t0 = time.time()
    if cls is None:
        print(f"FATAL: {algo_name} is not installed -- recording as crashed")
    result, crashed = (
        note_benchmark.run_note_evaluation(eval_dataset, cls, thresholds, device=device)
        if cls is not None else ({"conp": float("nan"), "conpoff": float("nan")}, True))
    write_cell(
        path,
        metadata={
            "track": "notes", "algorithm_name": algo_name, "dataset_name": dataset,
            "condition": cond, "seed": seed, "crashed": crashed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": round(time.time() - t0, 2),
        },
        parameters={
            "sample_rate": eval_dataset.sample_rate, "hop_size": eval_dataset.hop_size,
            "thresholds": [float(t) for t in thresholds],
            "lam_grid": note_benchmark.LAM_GRID,
            "onset_tolerance_s": 0.05, "pitch_tolerance_cents": 50.0, "offset_ratio": 0.2,
        },
        results=result,
    )
    print(f"[note] {dataset} {algo_name}: COnP={result.get('conp')} "
          f"({time.time() - t0:.1f}s) -> {os.path.basename(path)}")


def _write_ood_cell(out_dir, algo, family, ftype, device, results, crashed=False, error=None):
    meta = {
        "benchmark_type": "ood", "algorithm_name": algo, "family": family,
        "family_type": ftype, "device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if crashed:
        meta["crashed"] = True
        meta["error"] = error
    write_cell(
        os.path.join(out_dir, ood_cell_filename(algo, family, device)),
        metadata=meta,
        parameters={
            "sample_rate": ood_benchmark.SR, "hop_size": ood_benchmark.HOP,
            "fmin": ood_benchmark.FMIN, "fmax": ood_benchmark.FMAX,
            "n_seconds": ood_benchmark.N / ood_benchmark.SR,
            "f0s": ood_benchmark.FAMILIES.get(family, (None, None))[1],
        },
        results=results,
    )


def _run_one_ood_cell(cell, *, out_dir, device, in_process, cls_map):
    """One ood cell: 'skip' | None (done) | a crash kind (recorded)."""
    cls = cls_map[cell["algo"]]
    algo, family = cell["algo"], cell["dataset"]
    eff = cls.resolve_effective_device(device) if cls is not None else device
    ftype = "control" if family in ood_benchmark.CONTROL_FAMILIES else "voiced"
    path = os.path.join(out_dir, ood_cell_filename(algo, family, eff))
    if os.path.exists(path):     # cache-as-done
        return "skip"
    if in_process:
        try:
            if cls is None:
                raise RuntimeError(f"algorithm {algo} is not installed")
            _write_ood_cell(out_dir, algo, family, ftype, eff,
                            ood_benchmark.run_ood_cell(cls, family, device))
            return None
        except Exception as e:
            _write_ood_cell(out_dir, algo, family, ftype, eff, {}, crashed=True, error=str(e))
            return str(e)
    # ood needs no data paths, and its stimuli are seed-frozen by design (item_rng with fixed
    # family ids) -- the child's --seed is unused, so any value works here.
    return _spawn_cell(cell, paths=None, root=None, out_dir=out_dir, device=device, seed=42,
                       cap={}, expected=path,
                       on_crash=lambda k, _a=algo, _f=family, _t=ftype, _e=eff:
                           _write_ood_cell(out_dir, _a, _f, _t, _e, {},
                                           crashed=True, error=k))


def _run_ood_cells(cells, *, out_dir, device, in_process, cls_map):
    if not cells:
        return
    counts = {"done": 0, "skip": 0, "crash": 0}
    bar = tqdm(cells, desc="[ood] cells", unit="cell")
    for cell in bar:
        kind = _run_one_ood_cell(cell, out_dir=out_dir, device=device,
                                 in_process=in_process, cls_map=cls_map)
        if kind == "skip":
            counts["skip"] += 1
        elif kind:
            counts["crash"] += 1
            bar.write(f"[ood] CRASH {cell['algo']}/{cell['dataset']} ({kind})")
        else:
            counts["done"] += 1
        bar.set_postfix(**counts)


def _run_speed_cells(cells, *, out_dir, device, cls_map, sample_rate=22050, hop_length=256,
                     signal_length_sec=1.0, n_runs=10):
    """Serial + in-process, and ALWAYS overwrites: timing depends on machine state."""
    if not cells:
        return
    names = [a["algo"] for a in cells]
    baseline = "CREPE" if "CREPE" in names else names[0]
    devices = ["cpu", device] if device in ("cuda", "mps") else ["cpu"]
    ordered = sorted(cells, key=lambda a: a["algo"] != baseline)   # baseline first
    baseline_times = None
    timestamp = datetime.now(timezone.utc).isoformat()
    for a in ordered:
        cls = cls_map[a["algo"]]
        if cls is None:
            print(f"[speed] skip {a['algo']} (not installed)")
            continue
        results = speed_benchmark.run_speed_cell(
            cls, devices=devices, baseline_times=baseline_times,
            is_baseline=(a["algo"] == baseline), sample_rate=sample_rate,
            hop_length=hop_length, signal_length_sec=signal_length_sec, n_runs=n_runs)
        if a["algo"] == baseline:
            baseline_times = {
                d: (r["absolute_time_ms"] / 1000.0 if r["supported"] else float("inf"))
                for d, r in results["device_results"].items()}
        write_cell(
            os.path.join(out_dir, speed_cell_filename(
                a["algo"], sample_rate, hop_length, signal_length_sec, n_runs)),
            metadata={
                "benchmark_type": "speed", "algorithm_name": a["algo"],
                "baseline_algorithm": baseline, "timestamp_utc": timestamp,
                "devices_tested": devices, "cuda_available": torch.cuda.is_available(),
            },
            parameters={
                "sample_rate": sample_rate, "hop_length": hop_length,
                "signal_length_seconds": signal_length_sec, "n_runs": n_runs,
                "signal_type": "harmonic", "fundamental_frequency": 440,
                "harmonics": [1, 2, 3],
            },
            results=results,
        )
        print(f"[speed] {a['algo']} done")


def _spawn_cell(cell, *, paths, root, out_dir, device, seed, cap, expected, on_crash):
    """ONE child process for one cell: a narrowed evaluate.py CLI invocation, threads pinned,
    tqdm silenced (child bars would garble the parent console; the [track] result lines
    survive). If the child dies without writing its cell, the orchestrator records the crash."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "TQDM_DISABLE": "1"}
    cmd = [sys.executable, os.path.join(REPO, "evaluate.py"), "--_inproc",
           "--algorithms", cell["algo"], "--tracks", cell["track"],
           "--output-dir", out_dir, "--device", device, "--seed", str(seed),
           "--max-samples", str(cap.get("max_samples") or 0),
           "--max-seconds", str(cap.get("max_seconds") or 0)]
    if cell["track"] in ("frame", "note"):
        cmd += ["--datasets", cell["dataset"]]
        # ONE --data flag, many NAME=DIR values (argparse nargs="+": a repeated flag would
        # silently keep only its last occurrence)
        specs = [f"{cell['dataset']}={_data_dir(cell['dataset'], paths, root)}"]
        specs += [f"{n}={_data_dir(n, paths, root, required=False)}"
                  for n in ("chime_home", "DEMAND")
                  if _data_dir(n, paths, root, required=False) is not None]
        cmd += ["--data", *specs]
        if cell["track"] == "frame":
            cmd += ["--conditions", cell["condition"]]
    elif cell["track"] == "ood":
        cmd += ["--families", cell["dataset"]]
    # Only ood cells get a timeout: their runtime is known and bounded (fixed 2-s synthetic
    # clips) AND their stimuli are what makes fragile C-extension trackers hang. Frame/note
    # runtimes are data x algorithm dependent -- any fixed bound would kill honest work.
    timeout = 300 if cell["track"] == "ood" else None
    kind = None
    try:
        r = subprocess.run(cmd, env=env, cwd=REPO, timeout=timeout)
        if r.returncode != 0:
            kind = f"exit {r.returncode}"
    except subprocess.TimeoutExpired:
        kind = f"timeout > {timeout}s"
    if not os.path.exists(expected):
        on_crash(kind or "no output")
        return kind or "no output"
    return None


def _expected_path(cell, out_dir, device, seed, cap, cls):
    eff = cls.resolve_effective_device(device) if cls is not None else device
    if cell["track"] == "frame":
        return os.path.join(out_dir, frame_cell_filename(
            cell["dataset"], cell["algo"], cell["condition"], is_probe=bool(cap),
            max_samples=cap.get("max_samples"), max_seconds=cap.get("max_seconds"),
            device=eff, seed=seed))
    if cell["track"] == "note":
        return os.path.join(out_dir, note_cell_filename(
            cell["algo"], cell["dataset"], cell["condition"], seed))
    return os.path.join(out_dir, ood_cell_filename(cell["algo"], cell["dataset"], eff))


def _crash_writer(cell, out_dir, device, seed, cap, cls):
    """The crashed-cell record for a frame/note child that died without output."""
    def _write(kind):
        print(f"[{cell['track']}] CRASHED {cell['dataset']}/{cell['algo']} ({kind})")
        if cell["track"] == "frame":
            write_cell(
                _expected_path(cell, out_dir, device, seed, cap, cls),
                metadata={"algorithm_name": cell["algo"], "dataset_name": cell["dataset"],
                          "condition": cell["condition"], "probe": bool(cap), "seed": seed,
                          "device": device, "crashed": True, "crash_kind": kind,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                parameters={"max_samples": cap.get("max_samples"),
                            "max_seconds": cap.get("max_seconds")},
                results=metrics.to_json_safe(pitch_benchmark._failure_dict(0)))
        elif cell["track"] == "note":
            write_cell(
                _expected_path(cell, out_dir, device, seed, cap, cls),
                metadata={"track": "notes", "algorithm_name": cell["algo"],
                          "dataset_name": cell["dataset"], "condition": cell["condition"],
                          "seed": seed, "crashed": True, "crash_kind": kind,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                parameters={}, results={"conp": None, "conpoff": None})
    return _write


def run_cells(algos, *, paths=None, root=None, max_samples=30, max_seconds=10.0,
              out_dir="results", device="cpu", datasets=None, conditions=None, tracks=TRACKS,
              families=None, skip_datasets=(), seed=42, workers=1, _inproc=False):
    """Run every missing cell, then return load_cells(out_dir). Data locations: `paths` maps
    dataset name -> the directory where its files start (always wins); `root` is the optional
    <root>/<Name> convention fallback -- a frame/note run needs one of the two per dataset.
    Robustness (non-clean) frame cells run under (max_samples, max_seconds) -- default: the
    30-clip/10-s leaderboard cap; pass None/0 for both to uncap (verdict mode; assert_full
    accepts nothing less)."""
    max_samples = max_samples or None
    max_seconds = float(max_seconds) if max_seconds else None
    cap = ({"max_samples": max_samples, "max_seconds": max_seconds}
           if (max_samples or max_seconds) else {})
    cls_map, custom = _algo_classes(algos)
    names = list(cls_map)
    cells = enumerate_cells(names, datasets=datasets, conditions=conditions,
                            tracks=tracks, families=families, skip_datasets=skip_datasets)
    os.makedirs(out_dir, exist_ok=True)
    frame_cells = [c for c in cells if c["track"] == "frame"]
    note_cells = [c for c in cells if c["track"] == "note"]
    ood_cells = [c for c in cells if c["track"] == "ood"]
    speed_cells = [c for c in cells if c["track"] == "speed"]
    if cap and frame_cells and any(c["condition"] == "clean" for c in frame_cells):
        print("[frame] note: clean cells always run FULL -- the cap applies to robustness "
              "(non-clean) cells only")
    if workers > 1 and custom:
        print("evaluate: custom algorithm class -> running in-process (workers ignored)")
        workers = 1

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        def _child(cell):
            cell_cap = _cell_cap(cell, cap)
            expected = _expected_path(cell, out_dir, device, seed, cell_cap,
                                      cls_map[cell["algo"]])
            if cell["track"] != "ood" and os.path.exists(expected):
                return
            if cell["track"] == "ood":
                kind = _run_one_ood_cell(cell, out_dir=out_dir, device=device,
                                         in_process=False, cls_map=cls_map)
                if kind and kind != "skip":
                    print(f"[ood] CRASH {cell['algo']}/{cell['dataset']} ({kind})")
                return
            _spawn_cell(cell, paths=paths, root=root, out_dir=out_dir, device=device,
                        seed=seed, cap=cell_cap, expected=expected,
                        on_crash=_crash_writer(cell, out_dir, device, seed, cell_cap,
                                               cls_map[cell["algo"]]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_child, frame_cells + note_cells + ood_cells))
    else:
        _run_frame_cells(frame_cells, paths=paths, root=root, out_dir=out_dir,
                         device=device, seed=seed, run_cap=cap, cls_map=cls_map)
        _run_note_cells(note_cells, paths=paths, root=root, out_dir=out_dir,
                        device=device, seed=seed, cls_map=cls_map)
        _run_ood_cells(ood_cells, out_dir=out_dir, device=device,
                       in_process=(custom or _inproc), cls_map=cls_map)
    _run_speed_cells(speed_cells, out_dir=out_dir, device=device, cls_map=cls_map)
    return load_cells(out_dir, algos=names)


# ---------------------------------------------------------------------------- #
# Results: load, certify, compare
# ---------------------------------------------------------------------------- #
def load_cells(results_dir, algos=None):
    """{(track, dataset_or_family, condition, algo): cell_json}, routed by metadata content
    (filenames are never parsed)."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        m = d.get("metadata", {})
        algo = m.get("algorithm_name")
        if algo is None or (algos and algo not in algos):
            continue
        if m.get("track") == "notes":
            key = ("note", m.get("dataset_name"), m.get("condition", "clean"), algo)
        elif m.get("benchmark_type") == "ood":
            key = ("ood", m.get("family"), None, algo)
        elif m.get("benchmark_type") == "speed":
            key = ("speed", None, None, algo)
        elif m.get("dataset_name"):
            key = ("frame", m.get("dataset_name"), m.get("condition"), algo)
        else:
            continue
        cells[key] = d
    return cells


def assert_full(cells, algos, *, datasets=None, conditions=None, tracks=("frame", "note"),
                skip_datasets=()):
    """Certify that `cells` IS the full benchmark for `algos`: every expected cell present and
    no probe-sized cell anywhere. A subset cannot masquerade as the full benchmark."""
    names = list(_algo_classes(algos)[0])
    missing, probed = [], []
    for cell in enumerate_cells(names, datasets=datasets, conditions=conditions,
                                tracks=tracks, skip_datasets=skip_datasets):
        if cell["track"] not in ("frame", "note"):
            continue
        key = (cell["track"], cell["dataset"], cell["condition"], cell["algo"])
        got = cells.get(key)
        if got is None:
            missing.append(key)
        elif got.get("metadata", {}).get("probe"):
            probed.append(key)
    if missing or probed:
        raise AssertionError(
            f"not the full benchmark: {len(missing)} missing cells, {len(probed)} probe-sized "
            f"cells. Missing: {missing[:8]}{'...' if len(missing) > 8 else ''}; "
            f"probe: {probed[:8]}{'...' if len(probed) > 8 else ''}"
        )


def compare(cells, algo_a, algo_b, metric="voicing_f1", *, datasets=None, conditions=None):
    """Paired cluster-bootstrap comparison of two algorithms POOLED over every frame cell both
    completed (optionally narrowed). Clusters are (dataset, condition, group), so correlated
    clips stay together and shared clip difficulty cancels. Returns (delta, lo, hi) of A - B."""
    keyed = {algo_a: {}, algo_b: {}}
    for (track, ds, cond, algo), cell in cells.items():
        if track != "frame" or algo not in keyed:
            continue
        if (datasets and ds not in datasets) or (conditions and cond not in conditions):
            continue
        if cell.get("metadata", {}).get("crashed"):
            continue
        pc = cell.get("results", {}).get("per_clip")
        if not pc or not pc.get("rows"):
            continue
        k, _n = metrics.frame_keyed(pc)
        for g, sums in k.items():
            keyed[algo][(ds, cond, g)] = sums
    return metrics.compare_keyed(keyed[algo_a], keyed[algo_b], metric)


def main():
    p = argparse.ArgumentParser(
        description="Run the benchmark (the one entry point for all tracks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-samples", type=int, default=30,
                   help="Clip cap for robustness (non-clean) cells; 0 = uncapped. The default "
                        "is the leaderboard cap -- uncapped verdict runs are affordable only "
                        "for a few algorithms (hours), never a whole leaderboard (days)")
    p.add_argument("--max-seconds", type=float, default=10.0,
                   help="Per-clip duration cap for robustness cells; 0 = uncapped")
    p.add_argument("--data", nargs="+", default=[], metavar="NAME=DIR",
                   help="Explicit dataset locations (where the files start), e.g. "
                        "--data 'PTDB=/x/my ptdb' KEELE=/y/keele chime_home=/z/chime. "
                        "Always wins over --root")
    p.add_argument("--root", default=None,
                   help="Optional convention fallback: a dataset without --data resolves to "
                        "<root>/<Name> (see README)")
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="Registry names (default: every installed algorithm)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tracks", nargs="+", default=list(TRACKS), choices=TRACKS)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--conditions", nargs="+", default=None, choices=CONDITIONS)
    p.add_argument("--families", nargs="+", default=None,
                   choices=ood_benchmark.ALL_FAMILIES)
    p.add_argument("--skip-datasets", nargs="+", default=[])
    p.add_argument("--report", action="store_true",
                   help="Generate the markdown report from --output-dir after the run")
    p.add_argument("--_inproc", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    algos = args.algorithms or get_available_algorithms()
    if not algos:
        p.error("no algorithms installed (uv sync --all-extras)")
    known = set(list_pitch_datasets()) | {"chime_home", "DEMAND"}
    paths = {}
    for spec in args.data:
        name, sep, d = spec.partition("=")
        if not sep or name not in known:
            p.error(f"--data expects NAME=DIR with a registered NAME, got: {spec}")
        paths[name] = d
    cells = run_cells(
        algos, paths=paths or None, root=args.root,
        max_samples=args.max_samples, max_seconds=args.max_seconds,
        out_dir=args.output_dir, device=args.device, datasets=args.datasets,
        conditions=args.conditions, tracks=tuple(args.tracks), families=args.families,
        skip_datasets=tuple(args.skip_datasets), seed=args.seed, workers=args.workers,
        _inproc=args._inproc,
    )
    if not args._inproc:                       # children skip the summary (parent owns it)
        cap_desc = ("uncapped" if not (args.max_samples or args.max_seconds)
                    else f"capped n{args.max_samples or '-'}/t{args.max_seconds or '-'}")
        print(f"\n=== {len(cells)} result cells in {args.output_dir} "
              f"(robust cells {cap_desc}, algos={len(algos)}) ===")
    if args.report:
        subprocess.run([sys.executable, os.path.join(REPO, "generate_report.py"),
                        "--results-dir", args.output_dir], cwd=REPO, check=False)


if __name__ == "__main__":
    main()
