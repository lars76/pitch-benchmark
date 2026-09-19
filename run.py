import argparse
import gc
import glob
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

import score
from algorithms import build_algorithm, get_algorithm, list_algorithms
from resampling import TimingProbe

_INTERRUPT_SIGNALS = frozenset({-signal.SIGINT, -signal.SIGTERM})
NOTHING_MEASURED = {"thresholds": [], "clips": [], "roles": [], "crashed": [],
                    "crash_kinds": [], "stats": []}

PITCH_BANDS = (("bass", 0.0, 80.0), ("low", 80.0, 260.0), ("mid", 260.0, 650.0),
               ("high", 650.0, 1050.0), ("vhigh", 1050.0, math.inf))

PIN_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")


@dataclass(frozen=True)
class Cell:
    dataset: str
    panel: str
    algo: str


def read_split(split_dir):
    with open(Path(split_dir) / "dataset.json") as f:
        return json.load(f)


class AudioDataset:

    def __init__(self, path, panel=None):
        self.dir = Path(path)
        self.panel = panel
        self.entries = []
        for role in ("valid", "test"):
            clips = self.audio_dir(role)
            if clips.is_dir():
                self.entries += [(role, i)
                                 for i in sorted(int(p.stem) for p in clips.glob("*.flac"))]
        if not self.entries:
            raise ValueError(f"{self.dir}: no clips for panel {self.panel!r}")
        split = read_split(self.dir.parent)
        self.hop_size = int(split["hop_size"])
        window = split["corpora"][self.dir.name]
        self.fmin, self.fmax = window["fmin"], window["fmax"]
        self.sample_rate = sf.info(str(self.audio_path(0))).samplerate

    def audio_dir(self, role_dir):
        clips = self.dir / role_dir / "audio"
        return clips if self.panel is None else clips / self.panel

    def audio_path(self, idx):
        role, index = self.entries[idx]
        return self.audio_dir(role) / f"{index:07d}.flac"

    def labels(self, idx):
        role, index = self.entries[idx]
        path = self.dir / role / "labels" / f"{index:07d}.npz"
        with np.load(path, allow_pickle=False) as labels:
            return labels["pitch"], labels["trusted"]

    def __len__(self):
        return len(self.entries)

    def key(self, idx):
        return f"{self.entries[idx][1]:07d}"

    def role(self, idx):
        return self.entries[idx][0]

    def __getitem__(self, idx):
        audio, _ = sf.read(str(self.audio_path(idx)), dtype="float32")
        pitch, trusted = self.labels(idx)
        return {"audio": audio, "pitch": pitch, "trusted": trusted}


def write_cell(path, *, dataset, panel, algo, results, crashed, crash_kind,
               elapsed_s=None):
    score.write_json(path, {
        "metadata": {"algorithm_name": algo, "dataset_name": dataset, "panel": panel,
                     "crashed": crashed, "crash_kind": crash_kind,
                     "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "elapsed_s": elapsed_s},
        "results": results})


def measure_cell(eval_dataset, algorithm_class, dataset_name):
    algo_name = algorithm_class.get_name()
    try:
        algo = build_algorithm(algorithm_class, eval_dataset.sample_rate,
                               eval_dataset.hop_size, eval_dataset.fmin, eval_dataset.fmax)
    except Exception as e:
        print(f"FATAL: {algo_name} failed to build ({e}). Recording as crashed.", file=sys.stderr)
        return dict(NOTHING_MEASURED), True, type(e).__name__

    clips, roles, crashed_flags, crash_kinds, stats = [], [], [], [], []
    for idx in range(len(eval_dataset)):
        sample = eval_dataset[idx]
        audio = sample["audio"]
        true_pitch = sample["pitch"]
        trusted = sample["trusted"]
        kind = None
        try:
            results = algo.extract_pitch(audio, thresholds=list(score.THRESHOLDS))
            if len(results) != len(score.THRESHOLDS):
                raise ValueError(f"expected {len(score.THRESHOLDS)} results, got {len(results)}")
            rows = score.sweep_clip_stats(results, true_pitch, trusted,
                                          eval_dataset.fmin, eval_dataset.fmax)
        except Exception as e:
            print(f"CLIP CRASH: {algo_name} on sample {idx}: {e}", file=sys.stderr)
            voiced, auditable, scope = score.label_masks(true_pitch, trusted,
                                                         eval_dataset.fmin, eval_dataset.fmax)
            rows = score.crashed_clip_stats(int(np.sum(voiced)),
                                            int(np.sum(auditable & scope)))
            kind = type(e).__name__
        clips.append(eval_dataset.key(idx))
        roles.append(eval_dataset.role(idx))
        crashed_flags.append(kind is not None)
        crash_kinds.append(kind)
        stats.append(rows)
        if (idx + 1) % 200 == 0:
            gc.collect()
    del algo

    results = {"thresholds": [float(t) for t in score.THRESHOLDS],
               "clips": clips, "roles": roles, "crashed": crashed_flags,
               "crash_kinds": crash_kinds, "stats": stats}
    if clips and all(crashed_flags):
        kind = Counter(k for k in crash_kinds if k).most_common(1)[0][0]
        print(f"FATAL: {algo_name} crashed on all {len(clips)} clips of {dataset_name} "
              f"({kind}). Recording the cell as crashed.", file=sys.stderr)
        return results, True, kind
    return results, False, None


def run_one_cell(cell, *, dataset_dir, out_dir):
    cls = get_algorithm(cell.algo, fail_silently=True)
    path = os.path.join(out_dir, cell.dataset, cell.panel, f"{cell.algo}.json")
    t0 = time.time()
    if cls is None:
        write_cell(path, dataset=cell.dataset, panel=cell.panel, algo=cell.algo,
                   results=dict(NOTHING_MEASURED), crashed=True, crash_kind="not installed")
        return
    eval_ds = AudioDataset(os.path.join(dataset_dir, cell.dataset), cell.panel)
    results, crashed, crash_kind = measure_cell(eval_ds, cls, cell.dataset)
    write_cell(path, dataset=cell.dataset, panel=cell.panel, algo=cell.algo, results=results,
               crashed=crashed, crash_kind=crash_kind, elapsed_s=round(time.time() - t0, 2))


def _log_crash(out_dir, cell, kind, text):
    try:
        with open(os.path.join(out_dir, "crashes.log"), "a") as f:
            f.write(f"\n=== {cell.algo} {cell.dataset}/{cell.panel} ({kind}) ===\n{text}\n")
    except OSError:
        pass


def spawn_cell(cell, *, dataset_dir, out_dir, timeout):
    expected = os.path.join(out_dir, cell.dataset, cell.panel, f"{cell.algo}.json")
    env = {**os.environ, "TQDM_DISABLE": "1", **dict.fromkeys(PIN_VARS, "1")}
    cmd = [sys.executable, __file__, "--cell", cell.dataset, cell.panel,
           cell.algo, "--dataset", dataset_dir, "--out", out_dir]
    kind, err, rc, console = None, "", None, ""
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=timeout or None)
        rc = r.returncode
        console = (r.stdout or "") + (r.stderr or "")
        if rc != 0:
            kind, err = f"exit {rc}", r.stderr or ""
    except subprocess.TimeoutExpired as e:
        kind = f"timeout > {timeout:.0f}s"
        err = e.stderr or ""
    except OSError as e:
        kind, err = "spawn failed", str(e)
    if os.path.exists(expected):
        try:
            with open(expected) as f:
                written = json.load(f)
        except (OSError, json.JSONDecodeError):
            written = {}
        m = written.get("metadata") or {}
        clip_kinds = [k for k in (written.get("results") or {}).get("crash_kinds") or [] if k]
        if clip_kinds and console:
            _log_crash(out_dir, cell, m.get("crash_kind") or f"{len(clip_kinds)} clips crashed",
                       console)
        return (m.get("crash_kind") or "crashed") if m.get("crashed") else None
    if rc in _INTERRUPT_SIGNALS:
        return "interrupted"
    kind = kind or "no output"
    _log_crash(out_dir, cell, kind, err or console)
    write_cell(expected, dataset=cell.dataset, panel=cell.panel, algo=cell.algo,
               results=dict(NOTHING_MEASURED), crashed=True, crash_kind=kind)
    return kind


def matrix_status(cells, out_dir):
    tally = Counter({"ok": 0, "failed": 0, "pending": 0})
    for c in cells:
        path = os.path.join(out_dir, c.dataset, c.panel, f"{c.algo}.json")
        if not os.path.exists(path):
            tally["pending"] += 1
            continue
        try:
            with open(path) as f:
                crashed = json.load(f).get("metadata", {}).get("crashed")
        except (json.JSONDecodeError, OSError):
            crashed = True
        tally["failed" if crashed else "ok"] += 1
    return tally


def _corpus_record(root, entry, panel):
    ds = AudioDataset(root, panel)
    seconds = 0.0
    frames = voiced = auditable = oow = clips = 0
    f0 = []
    for i in range(len(ds)):
        if ds.role(i) != "test":
            continue
        pitch, trusted = ds.labels(i)
        on = pitch > 0
        clips += 1
        seconds += sf.info(str(ds.audio_path(i))).frames / ds.sample_rate
        frames += pitch.size
        voiced += int(on.sum())
        auditable += int((on & trusted).sum())
        oow += int((on & trusted & ((pitch < ds.fmin) | (pitch > ds.fmax))).sum())
        if on.any():
            f0.append(pitch[on].astype(np.float64))
    allf0 = np.concatenate(f0) if f0 else np.array([np.nan])
    p5, p50, p95 = np.percentile(allf0, (5, 50, 95))
    nv = max(voiced, 1)
    return {**entry,
            "clips": clips,
            "cal_clips": len(glob.glob(os.path.join(root, "valid", "labels", "*.npz"))),
            "hours": seconds / 3600.0,
            "voiced_pct": 100.0 * voiced / max(frames, 1),
            "f0_p5": float(p5), "f0_p50": float(p50), "f0_p95": float(p95),
            "out_of_window_pct": 100.0 * oow / max(auditable, 1),
            "bands": {name: 100.0 * int(np.sum((allf0 >= lo) & (allf0 < hi))) / nv
                      for name, lo, hi in PITCH_BANDS}}


def probe_bands(windows):
    lo = min(float(w["fmin"]) for w in windows)
    hi = max(float(w["fmax"]) for w in windows)
    octaves = math.log2(hi / lo)
    n = max(1, math.ceil(octaves / 2.0))
    edges = lo * 2.0 ** (np.arange(n + 1) * octaves / n)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]


def warn_timestamps(algos, dataset_dir, datasets, panel, out_dir, probe_timeout=600.0):
    env = {**os.environ, "TQDM_DISABLE": "1"}
    for name in algos:
        cmd = [sys.executable, __file__, "--probe-timing", "--dataset", dataset_dir,
               "--out", out_dir, "--panels", panel, "--datasets", *datasets,
               "--algorithms", name]
        try:
            rc = subprocess.run(cmd, env=env, timeout=probe_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = f"timeout > {probe_timeout:.0f}s"
        except OSError as e:
            rc = str(e)
        if rc:
            print(f"warning: {name} could not be probed for timing ({rc}), so its frame "
                  f"timing was not checked", file=sys.stderr, flush=True)


def probe_timing(algos, doc, datasets, dataset_dir, panel, out_dir):
    sample_rate = AudioDataset(os.path.join(dataset_dir, datasets[0]), panel=panel).sample_rate
    probes = [TimingProbe(sample_rate, lo, hi)
              for lo, hi in probe_bands([doc["corpora"][d] for d in datasets])]
    hop_size = int(doc["hop_size"])
    span = f"{probes[0].f_lo:.0f}-{probes[-1].f_hi:.0f} Hz"
    for name in algos:
        if get_algorithm(name, fail_silently=True) is None:
            print(f"warning: {name} is not installed, so its frame timing was not checked",
                  file=sys.stderr, flush=True)
            continue
        seen = []
        for probe in probes:
            algo = None
            try:
                algo = build_algorithm(name, sample_rate, hop_size, probe.fmin, probe.fmax)
                seen.append(probe.offset_ms(algo))
            except Exception as e:
                print(f"warning: {name} raised on the {probe.f_lo:.0f}-{probe.f_hi:.0f} Hz "
                      f"probe ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
                seen.append(None)
            finally:
                del algo
                gc.collect()
        score.write_json(os.path.join(out_dir, f"probe_{name}.json"),
                         {"algorithm_name": name,
                          "bands": [{"f_lo": p.f_lo, "f_hi": p.f_hi, "offset_ms": ms}
                                    for p, ms in zip(probes, seen)]})
        blind = [p for p, ms in zip(probes, seen) if ms is None]
        got = [ms for ms in seen if ms is not None]
        if not got:
            print(f"warning: {name} did not follow the timing probe on any band ({span}), "
                  f"so its frame timing was not checked", file=sys.stderr, flush=True)
            continue
        exceeds_tolerance = max(abs(ms) for ms in got) > 2.0
        if not (exceeds_tolerance or blind):
            continue
        reach = (f"{min(got):+.1f} to {max(got):+.1f} ms" if len(got) > 1
                 else f"{got[0]:+.1f} ms")
        unchecked = ("" if not blind else "; unchecked at "
                     + ", ".join(f"{p.f_lo:.0f}-{p.f_hi:.0f}" for p in blind)
                     + " Hz, where it did not follow the probe")
        tail = ("; its frames may not line up with the labels it is scored against"
                if exceeds_tolerance else "")
        print(f"warning: {name} chirp alignment offset {reach} across {span}, positive where "
              f"a frame describes audio later than its own timestamp{unchecked}{tail}",
              file=sys.stderr, flush=True)


def run_cells(algos, *, dataset_dir, datasets=None, panels=None, out_dir="cells",
              workers=4, cell_timeout=3600.0):
    doc = read_split(dataset_dir)
    corpora = list(doc["corpora"])
    datasets = list(datasets or corpora)
    declared = doc["design"]
    panels = list(panels if panels is not None else declared["panels"])
    for kind, given, known in (("datasets", datasets, corpora),
                               ("panels", panels, list(declared["panels"])),
                               ("algorithms", algos, list_algorithms())):
        unknown = [g for g in given if g not in known]
        if unknown:
            raise ValueError(f"unknown {kind}: {unknown}\n"
                             f"  available: {', '.join(sorted(str(k) for k in known))}")
    cells = [Cell(d, p, a) for p in panels for d in datasets for a in algos]
    os.makedirs(out_dir, exist_ok=True)

    run_path = os.path.join(out_dir, "run.json")
    previous = {}
    if os.path.exists(run_path):
        with open(run_path) as f:
            previous = json.load(f)
    for name in algos:
        stale = Path(out_dir) / f"probe_{name}.json"
        if stale.exists():
            stale.unlink()
    warn_timestamps(algos, dataset_dir, datasets, panels[0], out_dir)
    probe = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "probe_*.json"))):
        with open(path) as f:
            record = json.load(f)
        probe[record["algorithm_name"]] = record["bands"]
    records = {c: _corpus_record(os.path.join(dataset_dir, c), doc["corpora"][c],
                                 declared["identity"]) for c in datasets}
    with open(run_path, "w") as f:
        json.dump({"dataset": os.path.abspath(dataset_dir), "probe": probe,
                   "design": declared,
                   "started_utc": (previous.get("started_utc")
                                   or datetime.now(timezone.utc).isoformat()),
                   "corpora": dict(previous.get("corpora") or {}, **records)},
                  f, indent=1)
    status = matrix_status(cells, out_dir)
    print(f"=== benchmark: {len(cells)} cells ({len(panels)} panels x {len(datasets)} datasets "
          f"x {len(algos)} algos), {status['ok']} ok, {status['failed']} failed, "
          f"{status['pending']} pending; dataset {dataset_dir}, "
          f"workers {workers}, output {out_dir}", file=sys.stderr, flush=True)

    pending = [c for c in cells
               if not os.path.exists(os.path.join(out_dir, c.dataset, c.panel,
                                                  f"{c.algo}.json"))]
    absent = {a for a in {c.algo for c in pending}
              if get_algorithm(a, fail_silently=True) is None}
    skipped = [c for c in pending if c.algo in absent]
    for cell in skipped:
        run_one_cell(cell, dataset_dir=dataset_dir, out_dir=out_dir)
    if absent:
        print(f"not installed: {', '.join(sorted(absent))}; recorded {len(skipped)} cells "
              f"without spawning a process each", file=sys.stderr, flush=True)
    pending = [c for c in pending if c.algo not in absent]
    bar = tqdm(total=len(pending), unit="cell", file=sys.stderr)
    failed = status["failed"] + len(skipped)

    def _child(cell):
        nonlocal failed
        kind = spawn_cell(cell, dataset_dir=dataset_dir, out_dir=out_dir,
                          timeout=cell_timeout)
        if kind not in (None, "interrupted"):
            failed += 1
            bar.write(f"FAILED {cell.algo} {cell.dataset}/{cell.panel} ({kind})", file=sys.stderr)
        bar.set_postfix(failed=failed)
        bar.update(1)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_child, pending))
    bar.close()
    final = matrix_status(cells, out_dir)
    print(f"=== done in {time.time() - t0:.0f}s: {len(cells)} cells, {final['ok']} ok, "
          f"{final['failed']} failed, {final['pending']} pending ===", file=sys.stderr, flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", metavar="DIR", required=True,
                   help="path to the downloaded eval/ split")
    p.add_argument("--algorithms", nargs="+", default=None,
                   help=f"default: the full registry ({len(list_algorithms())} trackers)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="default: every corpus the split declares")
    p.add_argument("--panels", nargs="+", default=None,
                   help="default: the full panel registry (identity + the factorial cells)")
    p.add_argument("--out", default="cells")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cell-timeout", type=float, default=3600.0,
                   help="wall-clock ceiling per cell in seconds (default 3600; "
                        "0 = none, for runs over whole corpora)")
    p.add_argument("--probe-timing", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cell", nargs=3, metavar=("DATASET", "PANEL", "ALGO"),
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.cell:
        run_one_cell(Cell(*args.cell), dataset_dir=args.dataset, out_dir=args.out)
        return
    if args.probe_timing:
        doc = read_split(args.dataset)
        datasets = list(args.datasets or doc["corpora"])
        probe_timing(args.algorithms or list_algorithms(), doc, datasets, args.dataset,
                     (args.panels or [doc["design"]["identity"]])[0], args.out)
        return
    run_cells(args.algorithms or list_algorithms(), dataset_dir=args.dataset,
              datasets=args.datasets, panels=args.panels, out_dir=args.out,
              workers=args.workers, cell_timeout=args.cell_timeout)


if __name__ == "__main__":
    main()
