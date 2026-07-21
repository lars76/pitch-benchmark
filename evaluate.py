#!/usr/bin/env python3
"""THE benchmark: the one entry point, the one orchestrator.

The benchmark = every registered dataset x every condition x four measurement suites
(frame, note, synthetic, speed), evaluated as ATOMIC CELLS, one (suite, dataset-or-family,
condition, algorithm) unit each. The suite modules (frame/note/synthetic/speed_benchmark) are pure
measurement libraries; THIS module owns everything else: cell enumeration, dataset paths,
filenames, result envelopes, cache-as-done, crash recording, process isolation, and the worker
pool. If the question is "who spawns / who names / who caches / who writes", the answer is
always: evaluate.

Dataset locations are explicit: the user says which datasets to run AND where their files
start in ONE map, `datasets={"PTDB": "/my/ptdb_dir", "SpeechSynth": None, ...}` -- the keys
are the matrix, a string value is that dataset's directory, and None means its own default
(only bundled corpora like SpeechSynth have one). The recorded conditions chime/demand are NOT
datasets; each is a typed Condition carrying its noise corpus dir (`Recorded("chime",
corpus_dir=DIR)`), so it cannot run without its data and there is no separate location map. On
the CLI, `--data NAME=DIR` supplies both dataset and recorded-corpus dirs, partitioned by name.

One sizing decision: robustness (non-clean) frame cells default to the LEADERBOARD CAP of
30 clips / 10 s (`max_clips=30, max_seconds=10.0`), adequate for ranking DIFFERENT trackers
(gaps >= 0.04: paired CI ~ +-0.02 at n=30) and the only affordable mode across many trackers (a
full pass measures 0.5-60 CORE-HOURS per tracker). Override or uncap (0/None) as needed: full
cells are required for experiment verdicts (deltas of 0.005-0.02 need hundreds of clips) and
affordable for an experiment's 1-3 algorithms. Honesty is structural, not knob-based: capped
cells are identified by their cap, and assert_full() rejects any cell that carries one, so
a verdict must explicitly uncap.

Execution policy (the whole table):
  frame/note   cache-as-done. workers=1: in-process, cells grouped by (dataset, condition)
               over ONE shared dataset instance (decode + degradation synthesis paid once per
               group, measured as the only redundancy worth avoiding). workers>1: one child
               process per cell (isolation for free; resume can't livelock).
  synthetic    cache-as-done; cells ALWAYS run in child processes (the synthetic stimuli are
               exactly what makes some C-extension trackers SIGSEGV), except custom
               classes, which cannot exist in a fresh interpreter and run in-process.
  speed        always overwrites (timing depends on machine state; a stale cached number is
               worse than a re-run); always serial and in-process (timing under a busy pool
               would be noise).
A child that dies without writing its cell gets a crashed cell written by the orchestrator.

An algorithm IS a PitchAlgorithm subclass; a string is shorthand for a registry class,
resolved once at the entry. A custom class exists only in this process, so runs with one stay
in-process (no child processes).

    from evaluate import run_cells, load_cells, compare, assert_full
    cells = run_cells([MyTracker, "SwiftF0"], datasets={"KEELE": "/d/KEELE"})  # uncapped verdict
    d = compare(cells, "MyTracker", "Baseline", metric="voicing_f1")
    print(d.value, d.lo, d.hi, d.significant)
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
import synthetic_benchmark
import frame_benchmark
import speed_benchmark
# The scoring layer, re-exported: evaluate is the ONE import surface for consumers.
from metrics import (  # noqa: F401
    CellKey,
    Delta,
    Score,
    Suite,
    Tracks,
    cap_of,
    cost_summary,
    factorization,
    overall,
    stationary_families,
    synthetic_recall,
    theta_star,
    track_ci,
    track_scores,
)
from algorithms import PitchAlgorithm, get_algorithm, get_available_algorithms
from datasets import (
    BY_NAME,
    CANONICAL as CANONICAL_CONDITIONS,
    Augment,
    Condition,
    Recorded,
    Truncate,
    get_pitch_dataset,
    list_note_datasets,
    list_pitch_datasets,
    subset,
)

# ---------------------------------------------------------------------------- #
# THE BENCHMARK DEFINITION: registered = benchmarked, nothing more to declare.
# datasets from datasets.list_pitch_datasets(), conditions from datasets.augment.CANONICAL,
# note membership from the provides_notes capability, suites = the four measurement libraries.
# Opt out per run with --skip-datasets / --datasets; interpretation caveats (score-grade GT,
# train/test overlap) live on the dataset classes and in the report, not as exclusions here.
# A run is uncapped by default -- the full, certifiable measurement. A cap (run_cells
# max_clips=/max_seconds=) applies to every frame cell, clean included; the cap is part of a
# cell's key, and assert_full() rejects any cell carrying one, so a verdict must run uncapped.
# The affordable leaderboard is two runs (capped-all + uncapped-clean), see the README.
# Everything below this section is execution machinery, not definition.
# ---------------------------------------------------------------------------- #
CONDITIONS = tuple(BY_NAME)
# The four measurement libraries a run can invoke, derived from the Suite enum so the set is
# declared once. Deliberately NOT called "tracks": a track is one of the six SCORED questions
# in the report, and one word for two axes made "--tracks frame" and "Track 1: Correctness"
# look related when they are not.
SUITES = tuple(Suite)


# Data that ships committed inside this repo (no download, no user path needed).
BUNDLED = {"SpeechSynth": os.path.join(REPO, "datasets", "speechsynth.pt")}
# A recorded condition (chime/demand) mixes in a real noise corpus at a machine-specific path,
# carried on its own object (Recorded.corpus_dir). PROCEDURAL is everything else -- fully baked,
# runnable with no external data; it is the default frame axis. RECORDED_NAMES is used only to
# validate the CLI's --data partition and the frame narrowing.
PROCEDURAL = tuple(c for c in CANONICAL_CONDITIONS if not isinstance(c, Recorded))
RECORDED_NAMES = tuple(c.name for c in CANONICAL_CONDITIONS if isinstance(c, Recorded))


def all_conditions(chime_dir, demand_dir):
    """The full 13-condition axis for a leaderboard run: every procedural condition plus the two
    recorded ones bound to their corpora. `suites={"frame": all_conditions(...)}`."""
    dirs = {"chime": chime_dir, "demand": demand_dir}
    return list(PROCEDURAL) + [Recorded(n, corpus_dir=dirs[n]) for n in RECORDED_NAMES]
# Set in a spawned worker's environment (see _spawn_cell) to mark it a child, so the private
# is_child() reads it instead of run_cells carrying a public "_inproc" argument.
CHILD_ENV = "PITCH_BENCH_CHILD"


def is_child():
    return os.environ.get(CHILD_ENV) == "1"


def _data_dir(name, datasets):
    """Where `name`'s files start. `datasets` is the {name: path | None} map: a string path
    overrides, None (or an absent key) means the dataset's own default -- and only bundled
    datasets have one, so anything else raises."""
    override = (datasets or {}).get(name)
    if override is not None:
        return override
    if name in BUNDLED:
        return BUNDLED[name]
    raise ValueError(
        f"no location for {name}: pass datasets={{'{name}': DIR}} / --data {name}=DIR")


# ---------------------------------------------------------------------------- #
# Dataset build (the one sequence), used by the orchestrator and by tests that need
# custom-capped cells (e.g. an algorithm's own parity/regression tests).
# ---------------------------------------------------------------------------- #
def build_eval_dataset(dataset, data_dir, *, sample_rate=16000, hop_size=256, max_clips=None,
                       max_seconds=None, condition=None, seed=42):
    """load -> probe cap (even-stride subset) -> truncate -> degrade. `condition` is a typed
    Condition object; it owns its parameters (incl. a recorded corpus) and knows how to build its
    transforms -- clean builds an empty pipeline (a literal pass-through). Returns
    (eval_dataset, is_probe)."""
    base = get_pitch_dataset(dataset)(
        root_dir=data_dir, sample_rate=sample_rate, hop_size=hop_size,
    )
    is_probe = bool(max_clips or max_seconds)
    if max_clips and max_clips < len(base):
        idxs = sorted({round(i) for i in np.linspace(0, len(base) - 1, max_clips)})
        base = subset(base, idxs)
    if max_seconds:
        base = Truncate(base, float(max_seconds))
    pipeline = condition.build(sample_rate) if condition is not None else []
    if not pipeline:
        return base, is_probe
    return Augment(base, pipeline, seed=seed), is_probe


# ---------------------------------------------------------------------------- #
# Filenames, cache, atomic writes (readers route by metadata, never by filename)
# ---------------------------------------------------------------------------- #
def frame_cell_filename(dataset, algo, degradation, *, is_probe=False, max_clips=None,
                        max_seconds=None, sample_rate=16000, hop_size=256, device="cpu", seed=42):
    param_str = (
        f"{degradation}_"
        + (f"probe-n{max_clips}-t{max_seconds}_" if is_probe else "")
        + f"sr{int(sample_rate / 1000)}k_"
        + f"hop{hop_size}_"
        + device
    )
    return f"{dataset}_{algo}_{param_str}_seed{seed}.json"


def note_cell_filename(algo, dataset, condition, seed=42):
    return f"notes_{algo}_{dataset}_{condition}_seed{seed}.json"


def synthetic_cell_filename(algo, family, device):
    return (f"synthetic_{algo}_{family}_{device}_"
            f"sr{synthetic_benchmark.SR // 1000}k_hop{synthetic_benchmark.HOP}.json")


def speed_cell_filename(algo, sample_rate, hop_length, signal_length_sec, n_runs):
    return (f"speed_{algo}_sr{int(sample_rate / 1000)}k_hop{hop_length}_"
            f"len{signal_length_sec}s_runs{n_runs}.json")


def write_cell(result_path, *, suite, metadata, parameters, results):
    """Temp file + atomic rename: a kill mid-write must not leave a truncated JSON that
    cache_skip treats as finished.

    `suite` is required so no cell can be written without declaring which measurement it is;
    load_cells reads that one field rather than inferring the suite from which other fields
    happen to be present."""
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    # every cell records what it is and the format it was written under, so a reader can
    # refuse one written under different metric definitions instead of misreading it
    obj = {"metadata": {**metadata, "suite": Suite(suite).value,
                        "format": metrics.format_id()},
           "parameters": parameters, "results": results}
    tmp_path = f"{result_path}.{os.getpid()}.tmp"    # per-process: concurrent writers cannot collide
    with open(tmp_path, "w") as f:
        # compact: per-threshold per-clip stats make cells big; indented arrays would 4x them
        json.dump(metrics.to_json_safe(obj), f, separators=(",", ":"))
    os.replace(tmp_path, result_path)


# ---------------------------------------------------------------------------- #
# Cell enumeration: the benchmark matrix as cells
# ---------------------------------------------------------------------------- #
def _check_names(kind, given, known):
    """Reject unknown narrowing names with the closest match. A typo that merely narrows
    the matrix produces an empty run that looks like a completed one, so it must raise."""
    unknown = [g for g in given if g not in known]
    if not unknown:
        return
    import difflib
    lines = [f"unknown {kind}: {unknown}"]
    for u in unknown:
        # match case-insensitively: a wrong case is the commonest typo and would
        # otherwise score too low to be suggested at all
        lowered = {str(k).lower(): str(k) for k in known}
        near = [lowered[m] for m in
                difflib.get_close_matches(str(u).lower(), list(lowered), n=3, cutoff=0.6)]
        if near:
            lines.append(f"  did you mean: {', '.join(near)}?")
    lines.append(f"  available: {', '.join(sorted(str(k) for k in known))}")
    raise ValueError("\n".join(lines))


def enumerate_cells(algos, *, datasets=None, conditions=None, suites=SUITES, families=None):
    """The benchmark matrix as atomic cells: dicts of (suite, dataset, condition, algo).
    Enumeration is PURE structure; cell sizing (the robustness cap) is an execution decision
    applied by run_cells. `datasets` (names) / `conditions` / `families` narrow the matrix (for
    screens); to drop a dataset, leave it out of `datasets` -- there is one selector, not two.
    Narrowing is visible in the result matrix, never silent."""
    _check_names("suites", suites, SUITES)
    frame_ds = list(datasets or list_pitch_datasets())
    _check_names("datasets", frame_ds, list_pitch_datasets())
    conds = tuple(conditions if conditions is not None else CONDITIONS)
    _check_names("conditions", conds, CONDITIONS)
    fams = tuple(families or synthetic_benchmark.ALL_FAMILIES)
    _check_names("synthetic families", fams, synthetic_benchmark.ALL_FAMILIES)
    cells = []
    if Suite.FRAME in suites:
        for c in conds:                                  # clean first (never capped)
            for d in frame_ds:
                for a in algos:
                    cells.append(CellKey(suite=Suite.FRAME, subject=d, condition=c, algo=a))
    if Suite.NOTE in suites:
        note_ds = [d for d in (datasets or list_note_datasets()) if d in list_note_datasets()]
        for d in note_ds:
            for a in algos:
                cells.append(CellKey(suite=Suite.NOTE, subject=d, condition="clean", algo=a))
    if Suite.SYNTHETIC in suites:
        for f in fams:
            for a in algos:
                cells.append(CellKey(suite=Suite.SYNTHETIC, subject=f, condition=None, algo=a))
    if Suite.SPEED in suites:
        for a in algos:
            cells.append(CellKey(suite=Suite.SPEED, subject=None, condition=None, algo=a))
    return cells


def _cell_cap(cell, cap):
    """The one sizing rule: the run's cap applies to every FRAME cell, clean included. The
    other suites ignore it (their runners take no cap). A capped run therefore produces clean
    at the same cap as the degraded conditions, which is exactly the same-clips partner the
    report's Delta-from-clean pairing needs. The full-corpus clean headline comes from a
    separate uncapped run (its clean cell has cap=None, a distinct key), which theta_star
    then prefers -- see the standard leaderboard run in the README."""
    return cap if cell.suite is Suite.FRAME else {}


def _algo_classes(algos):
    """The algorithm contract: an algorithm IS a PitchAlgorithm subclass; a string is shorthand
    for a registry class, resolved once here. Returns ({name: class_or_None}, in_process).
    class None = a named backend that is not installed (executors record it as a crashed cell).

    `in_process` is the set of algorithms a child process could not rebuild -- a class the
    registry cannot resolve by name exists only in THIS interpreter. Those run in-process;
    everything else is isolated. A SET rather than a run-wide flag, so that passing one
    custom class cannot de-isolate the registry trackers that segfault: theirs is the crash
    that must be recorded as a crashed cell rather than take the run down."""
    cls_map, in_process = {}, set()
    for a in algos:
        if isinstance(a, type) and issubclass(a, PitchAlgorithm):
            name = a.get_name()
            cls_map[name] = a
            if get_algorithm(name, fail_silently=True) is not a:
                in_process.add(name)
        else:
            cls_map[a] = get_algorithm(a, fail_silently=True)
    return cls_map, in_process


# ---------------------------------------------------------------------------- #
# Cell execution
# ---------------------------------------------------------------------------- #
def run_and_write_frame_cell(eval_dataset, cls, *, out_dir, dataset, condition, is_probe,
                             max_clips, max_seconds, seed, sample_rate, hop_size, device,
                             algo_name=None):
    """Execute ONE frame cell on an already-built dataset and write it (skip if cached).
    `cls` is the algorithm class (None = an uninstalled named backend -> crashed cell;
    `algo_name` then supplies the name). Also the reference path for external parity tests that
    need custom-capped cells."""
    algo_name = algo_name or (cls.get_name() if cls is not None else "unknown")
    if cls is None:
        print(f"FATAL: {algo_name} is not installed, recording as crashed")
        eff = device
    else:
        eff = cls.resolve_effective_device(device)
    path = os.path.join(out_dir, frame_cell_filename(
        dataset, algo_name, condition, is_probe=is_probe, max_clips=max_clips,
        max_seconds=max_seconds, sample_rate=sample_rate, hop_size=hop_size,
        device=eff, seed=seed))
    if os.path.exists(path):   # cache-as-done
        print(f"[frame] skip {os.path.basename(path)} (exists; delete to redo)")
        return path
    t0 = time.time()
    if cls is None:
        result, crashed = frame_benchmark._failure_dict(0), True
    else:
        result, crashed = frame_benchmark.run_single_evaluation(
            dataset=eval_dataset, algorithm_class=cls,
            thresholds=metrics.DEFAULT_THRESHOLDS, device=device)
    write_cell(
        path,
        suite=Suite.FRAME,
        metadata={
            "algorithm_name": algo_name, "dataset_name": dataset, "condition": condition,
            "probe": is_probe, "seed": seed, "device": eff, "crashed": crashed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": round(time.time() - t0, 2),
        },
        parameters={
            "sample_rate": sample_rate, "hop_size": hop_size,
            "max_clips": max_clips, "max_seconds": max_seconds,
            "fmin": eval_dataset.fmin, "fmax": eval_dataset.fmax,
        },
        results=result,
    )
    sweep = result.get("sweep") or []
    peak = max((e["pitch"]["f1"] for e in sweep if e.get("pitch")), default=None)
    status = "CRASHED" if crashed else (
        f"pitch F1(peak)={peak:.4f}" if peak is not None else "empty")
    print(f"[frame] {dataset}/{condition} {algo_name}: {status} "
          f"({time.time() - t0:.1f}s) -> {os.path.basename(path)}")
    return path


def _run_frame_cells(cells, *, datasets, conditions, out_dir, device, seed, run_cap, cls_map):
    """Frame cells grouped by (dataset, condition): build the dataset ONCE per group and run the
    group's algorithms over it (decode + degradation synthesis shared). `conditions` maps a
    condition name to its typed Condition object (carrying any recorded corpus)."""
    groups = {}
    for c in cells:
        groups.setdefault((c.subject, c.condition), []).append(c)
    for (ds, cond), group in groups.items():
        cap = _cell_cap(group[0], run_cap)
        if all(os.path.exists(_expected_path(c, out_dir, device, seed, cap,
                                             cls_map[c.algo])) for c in group):
            for c in group:
                print(f"[frame] skip {ds}/{cond}/{c.algo} (cached)")
            continue
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"[frame] {ds}/{cond}: {len(group)} algorithms (seed {seed})")
        eval_ds, is_probe = build_eval_dataset(
            ds, _data_dir(ds, datasets), condition=conditions[cond], seed=seed, **cap)
        for c in group:
            run_and_write_frame_cell(
                eval_ds, cls_map[c.algo], out_dir=out_dir, dataset=ds, condition=cond,
                is_probe=is_probe, max_clips=cap.get("max_clips"),
                max_seconds=cap.get("max_seconds"), seed=seed,
                sample_rate=16000, hop_size=256, device=device, algo_name=c.algo)


def _run_note_cells(cells, *, datasets, out_dir, device, seed, cls_map):
    """Note cells grouped by dataset (always clean, never capped): one shared dataset build."""
    groups = {}
    for c in cells:
        groups.setdefault(c.subject, []).append(c)
    for ds, group in groups.items():
        pending = [c for c in group if not os.path.exists(os.path.join(
            out_dir, note_cell_filename(c.algo, ds, "clean", seed)))]
        for c in group:
            if c not in pending:
                print(f"[note] skip {ds}/{c.algo} (exists; delete to redo)")
        if not pending:
            continue
        eval_ds, _ = build_eval_dataset(ds, _data_dir(ds, datasets), seed=seed)
        thresholds = np.round(np.arange(0.0, 1.01, 0.1), 2)
        for c in pending:
            _run_note_cell(eval_ds, c.algo, cls_map[c.algo], ds, "clean", thresholds,
                           out_dir, device, seed)


def _run_note_cell(eval_dataset, algo_name, cls, dataset, cond, thresholds, out_dir, device,
                   seed):
    path = os.path.join(out_dir, note_cell_filename(algo_name, dataset, cond, seed))
    t0 = time.time()
    if cls is None:
        print(f"FATAL: {algo_name} is not installed, recording as crashed")
    result, crashed = (
        note_benchmark.run_note_evaluation(eval_dataset, cls, thresholds, device=device)
        if cls is not None else ({"conp": float("nan"), "conpoff": float("nan")}, True))
    write_cell(
        path,
        suite=Suite.NOTE,
        metadata={
            "algorithm_name": algo_name, "dataset_name": dataset,
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


def _write_synthetic_cell(out_dir, algo, family, ftype, device, results, crashed=False, error=None):
    meta = {
        "algorithm_name": algo, "family": family,
        "family_type": ftype, "device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if crashed:
        meta["crashed"] = True
        meta["error"] = error
    write_cell(
        os.path.join(out_dir, synthetic_cell_filename(algo, family, device)),
        suite=Suite.SYNTHETIC,
        metadata=meta,
        parameters={
            "sample_rate": synthetic_benchmark.SR, "hop_size": synthetic_benchmark.HOP,
            "fmin": synthetic_benchmark.FMIN, "fmax": synthetic_benchmark.FMAX,
            "n_seconds": synthetic_benchmark.N / synthetic_benchmark.SR,
            "f0s": synthetic_benchmark.FAMILIES.get(family, (None, None))[1],
        },
        results=results,
    )


def _run_one_synthetic_cell(cell, *, out_dir, device, in_process, cls_map):
    """One synthetic cell: 'skip' | None (done) | a crash kind (recorded)."""
    cls = cls_map[cell.algo]
    algo, family = cell.algo, cell.subject
    eff = cls.resolve_effective_device(device) if cls is not None else device
    ftype = synthetic_benchmark.family_type(family)
    path = os.path.join(out_dir, synthetic_cell_filename(algo, family, eff))
    if os.path.exists(path):     # cache-as-done
        return "skip"
    if in_process:
        try:
            if cls is None:
                raise RuntimeError(f"algorithm {algo} is not installed")
            _write_synthetic_cell(out_dir, algo, family, ftype, eff,
                            synthetic_benchmark.run_synthetic_cell(cls, family, device))
            return None
        except Exception as e:
            _write_synthetic_cell(out_dir, algo, family, ftype, eff, {}, crashed=True, error=str(e))
            return str(e)
    # ood needs no data paths, and its stimuli are seed-frozen by design (item_rng with fixed
    # family ids), the child's --seed is unused, so any value works here.
    return _spawn_cell(cell, datasets=None, conditions={}, out_dir=out_dir, device=device,
                       seed=42, cap={}, expected=path,
                       on_crash=lambda k, _a=algo, _f=family, _t=ftype, _e=eff:
                           _write_synthetic_cell(out_dir, _a, _f, _t, _e, {},
                                           crashed=True, error=k))


def _run_synthetic_cells(cells, *, out_dir, device, in_process, cls_map):
    if not cells:
        return
    counts = {"done": 0, "skip": 0, "crash": 0}
    bar = tqdm(cells, desc="[synthetic] cells", unit="cell")
    for cell in bar:
        kind = _run_one_synthetic_cell(cell, out_dir=out_dir, device=device,
                                 in_process=in_process, cls_map=cls_map)
        if kind == "skip":
            counts["skip"] += 1
        elif kind:
            counts["crash"] += 1
            bar.write(f"[synthetic] CRASH {cell.algo}/{cell.subject} ({kind})")
        else:
            counts["done"] += 1
        bar.set_postfix(**counts)


def _run_speed_cells(cells, *, out_dir, device, cls_map, sample_rate=22050, hop_length=256,
                     signal_length_sec=1.0, n_runs=10):
    """Serial + in-process, and ALWAYS overwrites: timing depends on machine state."""
    if not cells:
        return
    devices = ["cpu", device] if device in ("cuda", "mps") else ["cpu"]
    timestamp = datetime.now(timezone.utc).isoformat()
    for a in cells:
        cls = cls_map[a.algo]
        if cls is None:
            print(f"[speed] skip {a.algo} (not installed)")
            continue
        results = speed_benchmark.run_speed_cell(
            cls, devices=devices, sample_rate=sample_rate,
            hop_length=hop_length, signal_length_sec=signal_length_sec, n_runs=n_runs)
        write_cell(
            os.path.join(out_dir, speed_cell_filename(
                a.algo, sample_rate, hop_length, signal_length_sec, n_runs)),
            suite=Suite.SPEED,
            metadata={
                "algorithm_name": a.algo,
                "timestamp_utc": timestamp, "devices_tested": devices,
            },
            parameters={
                "sample_rate": sample_rate, "hop_length": hop_length,
                "signal_length_seconds": signal_length_sec, "n_runs": n_runs,
                "signal_type": "harmonic", "fundamental_frequency": 440,
                "harmonics": [1, 2, 3],
            },
            results=results,
        )
        print(f"[speed] {a.algo} done")


def _spawn_cell(cell, *, datasets, conditions, out_dir, device, seed, cap, expected, on_crash):
    """ONE child process for one cell: a narrowed evaluate.py CLI invocation, threads pinned,
    tqdm silenced (child bars would garble the parent console; the [suite] result lines
    survive). If the child dies without writing its cell, the orchestrator records the crash.
    `conditions` is the name->Condition map, so a recorded cell's corpus dir travels on --data."""
    # CHILD_ENV marks the process as a spawned child: it runs its one cell in-process (no
    # further spawning) and skips the run summary (the parent owns it). An env var, not a
    # public run_cells argument -- it is orchestration plumbing, not a knob a caller sets.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "TQDM_DISABLE": "1", CHILD_ENV: "1"}
    cmd = [sys.executable, os.path.join(REPO, "evaluate.py"),
           "--algorithms", cell.algo, "--suites", cell.suite.value,
           "--output-dir", out_dir, "--device", device, "--seed", str(seed),
           "--max-clips", str(cap.get("max_clips") or 0),
           "--max-seconds", str(cap.get("max_seconds") or 0)]
    if cell.suite in (Suite.FRAME, Suite.NOTE):
        cmd += ["--datasets", cell.subject]
        # ONE --data flag, many NAME=DIR values (argparse nargs="+": a repeated flag would
        # silently keep only its last occurrence). A recorded cell's corpus rides here too, keyed
        # by condition name, and the child rebinds it onto the Condition object.
        specs = [f"{cell.subject}={_data_dir(cell.subject, datasets)}"]
        if cell.suite == Suite.FRAME:
            cond = conditions[cell.condition]
            if isinstance(cond, Recorded):
                specs.append(f"{cond.name}={cond.corpus_dir}")
            cmd += ["--conditions", cell.condition]
        cmd += ["--data", *specs]
    elif cell.suite == Suite.SYNTHETIC:
        cmd += ["--families", cell.subject]
    # Only ood cells get a timeout: their runtime is known and bounded (fixed 2-s synthetic
    # clips) AND their stimuli are what makes fragile C-extension trackers hang. Frame/note
    # runtimes are data x algorithm dependent; any fixed bound would kill honest work.
    timeout = 300 if cell.suite == Suite.SYNTHETIC else None
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
    if cell.suite == Suite.FRAME:
        return os.path.join(out_dir, frame_cell_filename(
            cell.subject, cell.algo, cell.condition, is_probe=bool(cap),
            max_clips=cap.get("max_clips"), max_seconds=cap.get("max_seconds"),
            device=eff, seed=seed))
    if cell.suite == Suite.NOTE:
        return os.path.join(out_dir, note_cell_filename(
            cell.algo, cell.subject, cell.condition, seed))
    return os.path.join(out_dir, synthetic_cell_filename(cell.algo, cell.subject, eff))


def _crash_writer(cell, out_dir, device, seed, cap, cls):
    """The crashed-cell record for a frame/note child that died without output."""
    def _write(kind):
        print(f"[{cell.suite.value}] CRASHED {cell.subject}/{cell.algo} ({kind})")
        if cell.suite == Suite.FRAME:
            write_cell(
                _expected_path(cell, out_dir, device, seed, cap, cls),
                suite=Suite.FRAME,
                metadata={"algorithm_name": cell.algo, "dataset_name": cell.subject,
                          "condition": cell.condition, "probe": bool(cap), "seed": seed,
                          "device": device, "crashed": True, "crash_kind": kind,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                parameters={"max_clips": cap.get("max_clips"),
                            "max_seconds": cap.get("max_seconds")},
                results=frame_benchmark._failure_dict(0))
        elif cell.suite == Suite.NOTE:
            write_cell(
                _expected_path(cell, out_dir, device, seed, cap, cls),
                suite=Suite.NOTE,
                metadata={"algorithm_name": cell.algo,
                          "dataset_name": cell.subject, "condition": cell.condition,
                          "seed": seed, "crashed": True, "crash_kind": kind,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                parameters={}, results={"conp": None, "conpoff": None})
    return _write


def run_cells(algos, *, datasets=None, suites=None,
              max_clips=None, max_seconds=None, out_dir="results",
              device="cpu", seed=42, workers=4):
    """Run every missing cell, then return load_cells(out_dir).

    `suites` is the measurement spec, a {suite: narrowing} map -- presence of a key runs that
    suite, the value narrows it: "frame" -> a list of Condition OBJECTS (or None = every
    procedural condition), "synthetic" -> a list of family names (or None = all), "note"/"speed"
    -> None. Default (None) runs every suite at full width. Because the narrowing is scoped by
    its key, a family name cannot be handed to frame.

    A frame condition is a typed object that owns its parameters; a RECORDED one (chime, demand)
    carries its noise corpus (`Recorded("chime", corpus_dir=DIR)`) and cannot run without it --
    so there is no separate location map for noise, and a recorded condition is unwriteable
    without its data. `None` frame narrowing = all PROCEDURAL conditions (the ones needing no
    corpus); recorded ones run only when you pass them (with a dir) explicitly.

    `datasets` is a {name: path | None} LOCATION map that is also the frame/note subject set:
    a string is the directory, None means the dataset's own default (only bundled corpora have
    one, else it raises); to drop a dataset, leave it out.

    The cap (max_clips, max_seconds) applies to EVERY frame cell, clean included; default is
    uncapped -- the full, certifiable measurement. Pass a cap to sample; a capped run cannot
    certify (assert_full rejects it). The affordable leaderboard is two runs, see the README.

    Isolation is per algorithm and uniform: a registry tracker always runs in a child process
    (crash -> crashed cell, result independent of `workers`); your own class always runs
    in-process (a child cannot rebuild it). `workers` is only how many children run at once --
    a concurrency cap on short-lived per-cell processes, not a persistent pool; it overlaps their
    startups and is bounded by cores / memory / GPU. Default 4 (a few concurrent children fit a
    typical laptop's RAM with room to spare; set 1 for contention-free serial)."""
    suites = suites if suites is not None else {s.value: None for s in Suite}
    _check_names("suites", suites, [s.value for s in Suite])
    # None = every procedural condition; an explicit list (incl. []) is taken verbatim -- [] means
    # "no procedural", exactly a recorded-only child spawn. Each item is a Condition object; a
    # bare string is rejected (a recorded condition needs its corpus, which only the object holds).
    frame_narrowing = suites.get("frame")
    conds = list(PROCEDURAL) if frame_narrowing is None else list(frame_narrowing)
    for c in conds:
        if not isinstance(c, Condition):
            raise ValueError(
                f"frame conditions must be Condition objects, got {c!r} -- use "
                f"evaluate.BY_NAME[name], or Recorded(name, corpus_dir=DIR) for chime/demand")
    conditions = {c.name: c for c in conds}                 # name -> object, for the executor
    max_clips = max_clips or None
    max_seconds = float(max_seconds) if max_seconds else None
    cap = ({"max_clips": max_clips, "max_seconds": max_seconds}
           if (max_clips or max_seconds) else {})
    cls_map, in_process = _algo_classes(algos)
    names = list(cls_map)
    dataset_names = list(datasets) if datasets else None
    cells = enumerate_cells(names, datasets=dataset_names, conditions=list(conditions),
                            suites=tuple(Suite(k) for k in suites),
                            families=suites.get("synthetic"))
    os.makedirs(out_dir, exist_ok=True)
    frame_cells = [c for c in cells if c.suite == Suite.FRAME]
    note_cells = [c for c in cells if c.suite == Suite.NOTE]
    synthetic_cells = [c for c in cells if c.suite == Suite.SYNTHETIC]
    speed_cells = [c for c in cells if c.suite == Suite.SPEED]
    # Isolation is decided PER ALGORITHM, uniformly: a registry name a child can rebuild ALWAYS
    # runs in a child, so a segfault becomes a crashed cell and its result never depends on the
    # worker count; a class we were handed cannot be rebuilt in a fresh interpreter, so it always
    # runs in-process. The branch is child-vs-parent, not workers: `workers` only sets how many
    # children the parent runs at once.
    def _local(cs):
        return [c for c in cs if c.algo in in_process]

    def _isolated(cs):
        return [c for c in cs if c.algo not in in_process]

    if is_child():
        # a spawned worker runs its ONE cell in-process and never re-spawns (anti-recursion)
        _run_frame_cells(frame_cells, datasets=datasets, conditions=conditions, out_dir=out_dir,
                         device=device, seed=seed, run_cap=cap, cls_map=cls_map)
        _run_note_cells(note_cells, datasets=datasets, out_dir=out_dir,
                        device=device, seed=seed, cls_map=cls_map)
        _run_synthetic_cells(synthetic_cells, out_dir=out_dir, device=device,
                             in_process=True, cls_map=cls_map)
    else:
        if in_process:
            print(f"evaluate: {sorted(in_process)} cannot be isolated (custom class) and run "
                  f"in-process; the rest run in child processes")
        from concurrent.futures import ThreadPoolExecutor

        def _child(cell):
            cell_cap = _cell_cap(cell, cap)
            expected = _expected_path(cell, out_dir, device, seed, cell_cap,
                                      cls_map[cell.algo])
            if cell.suite != Suite.SYNTHETIC and os.path.exists(expected):
                return
            if cell.suite == Suite.SYNTHETIC:
                kind = _run_one_synthetic_cell(cell, out_dir=out_dir, device=device,
                                         in_process=False, cls_map=cls_map)
                if kind and kind != "skip":
                    print(f"[synthetic] CRASH {cell.algo}/{cell.subject} ({kind})")
                return
            _spawn_cell(cell, datasets=datasets, conditions=conditions, out_dir=out_dir,
                        device=device, seed=seed, cap=cell_cap, expected=expected,
                        on_crash=_crash_writer(cell, out_dir, device, seed, cell_cap,
                                               cls_map[cell.algo]))
        # registry cells spawn (pool of `workers`; workers=1 is a serial pool of one); custom
        # cells run here in-process, sharing the per-group dataset build across algorithms.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_child, _isolated(frame_cells + note_cells + synthetic_cells)))
        _run_frame_cells(_local(frame_cells), datasets=datasets, conditions=conditions,
                         out_dir=out_dir, device=device, seed=seed, run_cap=cap, cls_map=cls_map)
        _run_note_cells(_local(note_cells), datasets=datasets, out_dir=out_dir,
                        device=device, seed=seed, cls_map=cls_map)
        _run_synthetic_cells(_local(synthetic_cells), out_dir=out_dir, device=device,
                             in_process=True, cls_map=cls_map)
    _run_speed_cells(speed_cells, out_dir=out_dir, device=device, cls_map=cls_map)
    return load_cells(out_dir, algos=names)


# ---------------------------------------------------------------------------- #
# Results: load, certify, compare
# ---------------------------------------------------------------------------- #
def cell_rank(d):
    """Which of two cells competing for one key wins: a completed cell beats a crashed
    one, then cpu (the reproducible reference) beats an accelerator. Device is in the
    filename, not the key, so cpu and mps variants of the same cell collide -- and the
    glob is alphabetical, which would otherwise let a crashed `_mps_` cell overwrite a
    good `_cpu_` one purely because 'm' sorts after 'c'."""
    m = d.get("metadata", {})
    return (not m.get("crashed"), m.get("device") == "cpu")


def load_cells(results_dir, algos=None):
    """{CellKey: cell_json}, routed by metadata content (filenames are never parsed)."""
    cells = {}

    def put(key, d):
        cur = cells.get(key)
        if cur is None or cell_rank(d) > cell_rank(cur):
            cells[key] = d

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
        metrics.check_format(d)
        suite = Suite(m["suite"])
        if suite is Suite.NOTE:
            key = CellKey(suite=suite, subject=m.get("dataset_name"),
                          condition=m.get("condition", "clean"), algo=algo)
        elif suite is Suite.SYNTHETIC:
            key = CellKey(suite=suite, subject=m.get("family"), condition=None, algo=algo)
        elif suite is Suite.SPEED:
            key = CellKey(suite=suite, subject=None, condition=None, algo=algo)
        else:
            key = CellKey(suite=suite, subject=m.get("dataset_name"),
                          condition=m.get("condition"), algo=algo, cap=cap_of(d))
        put(key, d)
    return cells


def assert_full(cells, algos, *, datasets=None, conditions=None,
                suites=tuple(s for s in Suite if s is not Suite.SPEED)):
    """Certify that `cells` IS the full benchmark for `algos`: every expected cell present and
    no probe-sized cell anywhere. A subset cannot masquerade as the full benchmark."""
    names = list(_algo_classes(algos)[0])
    missing, probed = [], []
    for key in enumerate_cells(names, datasets=datasets, conditions=conditions,
                               suites=suites):
        if key.suite is Suite.SPEED:     # timing is not a correctness claim
            continue
        # enumerate_cells yields cap=None keys, and only an UNCAPPED cell certifies: a capped
        # measurement of the same (dataset, condition, algo) lives under its own key, so it
        # simply is not this cell -- the honest reading of "not the full benchmark".
        if cells.get(key) is not None:
            continue
        capped = [k for k in cells
                  if k.cap is not None
                  and (k.suite, k.subject, k.condition, k.algo)
                      == (key.suite, key.subject, key.condition, key.algo)]
        (probed if capped else missing).append(key)
    if missing or probed:
        raise AssertionError(
            f"not the full benchmark: {len(missing)} cells never run, {len(probed)} run "
            f"only capped (re-run those uncapped). "
            f"Never run: {missing[:8]}{'...' if len(missing) > 8 else ''}; "
            f"capped only: {probed[:8]}{'...' if len(probed) > 8 else ''}"
        )


def compare(cells, algo_a, algo_b, metric="pitch_f1", *, theta="star", datasets=None,
            conditions=None):
    """Paired cluster-bootstrap comparison of two algorithms POOLED over every frame cell both
    completed (optionally narrowed), each read at its own operating point (theta="star" =
    each algorithm's frozen theta*; a float pins both to one shared threshold). Clusters are
    (dataset, condition, group), so correlated clips stay together and shared clip difficulty
    cancels. Returns a `Delta` of A - B, whose `.significant` is the interval excluding 0.

    Raises if either algorithm has no comparable cells, rather than returning a silent NaN
    triple: a caller that gets nothing back needs to know WHY (usually a stale cell store
    or a name that ran nothing)."""
    idxs = {}
    for algo in (algo_a, algo_b):
        if theta == "star":
            idxs[algo] = metrics.theta_star(cells, algo, datasets=datasets)["idx"]
        else:
            grid = list(metrics.DEFAULT_THRESHOLDS)
            idxs[algo] = int(np.argmin(np.abs(np.asarray(grid) - float(theta))))
        if idxs[algo] is None:
            raise ValueError(
                f"{algo} has no operating point: no readable clean frame cells. "
                "Run it first, or regenerate if the cell store predates the current "
                "metric definitions.")
    keyed = {algo_a: {}, algo_b: {}}
    conds = conditions or sorted({k.condition for k in cells if k.suite == Suite.FRAME})
    for algo in (algo_a, algo_b):
        for cond in conds:
            # metrics.frame_cells yields ONE cell per dataset, preferring the uncapped
            # measurement, so a capped and an uncapped run of the same thing are never
            # pooled together
            for ds, _cap, cell in metrics.frame_cells(cells, algo, cond, datasets):
                pc = cell.get("results", {}).get("per_clip")
                if not pc or not pc.get("stats"):
                    continue
                k, _n = metrics.frame_keyed(pc, idxs[algo])
                for g, sums in k.items():
                    keyed[algo][(ds, cond, g)] = sums
    common = set(keyed[algo_a]) & set(keyed[algo_b])
    if len(common) < 2:
        raise ValueError(
            f"{algo_a} and {algo_b} share {len(common)} comparable clusters; a paired "
            "interval needs at least 2. They may have been run on different datasets.")
    return metrics.Delta(*metrics.compare_keyed(keyed[algo_a], keyed[algo_b], metric))


def _suite_arg(s):
    """argparse type= for --suites: reuses the one did-you-mean helper and returns a Suite,
    so the CLI boundary produces the same type the programmatic entry does."""
    try:
        _check_names("suites", [s], SUITES)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return Suite(s)


def main():
    p = argparse.ArgumentParser(
        description="Run the benchmark (the one entry point for all suites).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-clips", type=int, default=0,
                   help="Clip cap for every frame cell (clean included); 0 = uncapped "
                        "(the default: the full, certifiable measurement). Pass a cap to "
                        "sample; see the standard leaderboard run in the README")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Per-clip duration cap for frame cells; 0 = uncapped")
    p.add_argument("--data", nargs="+", default=[], metavar="NAME=DIR",
                   help="Dataset locations, plus the corpus dir for a recorded condition "
                        "(chime/demand), e.g. --data 'PTDB=/x/my ptdb' KEELE=/y/keele chime=/z/chime")
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="Registry names (default: every installed algorithm)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=4,
                   help="Max concurrent CHILD PROCESSES (not a persistent pool): each registry "
                        "tracker cell runs in its own short-lived process, and workers>1 overlaps "
                        "their startups. Bounded by cores (each child is thread-pinned to 1), "
                        "memory (each neural child loads its own model), and the GPU (MPS children "
                        "contend). Default 4; set 1 for contention-free serial.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suites", nargs="+", default=list(SUITES), type=_suite_arg,
                   metavar="SUITE", help=f"default: {' '.join(s.value for s in SUITES)}")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Frame conditions to run. Procedural ones (pink, reverb, ...) run by "
                        "name; a recorded one (chime/demand) also needs its corpus via "
                        "--data chime=DIR. Default: every procedural condition.")
    p.add_argument("--families", nargs="+", default=None,
                   choices=synthetic_benchmark.ALL_FAMILIES)
    p.add_argument("--report", action="store_true",
                   help="Generate the markdown report from --output-dir after the run")
    args = p.parse_args()

    algos = args.algorithms or get_available_algorithms()
    if not algos:
        p.error("no algorithms installed (uv sync --all-extras)")
    # --data carries dataset paths AND recorded-condition corpus dirs (chime/demand), keyed the
    # same way. Partition by registry membership: a recorded-condition name -> its corpus dir
    # (bound onto a Recorded object below); everything else -> the whole-registry dataset map.
    known = set(list_pitch_datasets()) | set(RECORDED_NAMES)
    provided = {}
    for spec in args.data:
        name, sep, d = spec.partition("=")
        if not sep or name not in known:
            p.error(f"--data expects NAME=DIR with a registered NAME, got: {spec}")
        provided[name] = d
    corpora = {k: v for k, v in provided.items() if k in RECORDED_NAMES}
    dsmap = {name: provided.get(name) for name in list_pitch_datasets()}
    if args.datasets:
        unknown = set(args.datasets) - set(dsmap)
        if unknown:
            p.error(f"--datasets: unknown {sorted(unknown)}; choose from the dataset registry")
        dsmap = {k: dsmap[k] for k in args.datasets}
    # Resolve --conditions names to Condition OBJECTS; a recorded name is bound to its corpus
    # (from --data) here, else it is an error -- the object carries the dir, there is no map.
    def _recorded(name):
        if name not in corpora:
            p.error(f"condition {name} is recorded; add its corpus, e.g. --data {name}=DIR")
        return Recorded(name, corpus_dir=corpora[name])

    frame_narrowing = None
    if args.conditions:
        unknown = set(args.conditions) - set(BY_NAME)
        if unknown:
            p.error(f"--conditions: unknown {sorted(unknown)}; choose from {sorted(BY_NAME)}")
        frame_narrowing = [_recorded(n) if isinstance(BY_NAME[n], Recorded) else BY_NAME[n]
                           for n in args.conditions]
    elif corpora:
        # no explicit narrowing but corpora given -> the full axis: all procedural + those recorded
        frame_narrowing = list(PROCEDURAL) + [_recorded(n) for n in corpora]
    # assemble the {suite: narrowing} map from --suites + the frame/synthetic narrowings
    smap = {}
    for suite in args.suites:
        smap[suite.value] = (frame_narrowing if suite is Suite.FRAME
                             else args.families if suite is Suite.SYNTHETIC else None)
    cells = run_cells(
        algos, datasets=dsmap, suites=smap,
        max_clips=args.max_clips, max_seconds=args.max_seconds,
        out_dir=args.output_dir, device=args.device,
        seed=args.seed, workers=args.workers,
    )
    if not is_child():                         # children skip the summary (parent owns it)
        cap_desc = ("uncapped" if not (args.max_clips or args.max_seconds)
                    else f"capped n{args.max_clips or '-'}/t{args.max_seconds or '-'}")
        print(f"\n=== {len(cells)} result cells in {args.output_dir} "
              f"(robust cells {cap_desc}, algos={len(algos)}) ===")
    if args.report:
        # write the report NEXT TO the cells it renders: the repo's committed report
        # describes the full reference run, and a scratch run must not overwrite it
        out = os.path.join(args.output_dir, "benchmark_report.md")
        subprocess.run([sys.executable, os.path.join(REPO, "generate_report.py"),
                        "--results", args.output_dir, "--out", out], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
