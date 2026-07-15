#!/usr/bin/env bash
# THE single entry point: run the COMPLETE benchmark and build the report. Every registered dataset x
# all algorithms across BOTH tracks -- frame (clean leaderboard + robustness probe) AND note
# transcription -- plus speed and OOD. Run this to get all the numbers.
#
# PARALLEL: every (dataset, algorithm[, condition]) cell is an INDEPENDENT pitch_benchmark/note_benchmark
# invocation writing its own result JSON, so they fan out over a bounded worker pool (xargs -P). This only
# changes SCHEDULING, not output -- results are byte-identical to a serial run. Each cell is pinned to
# $THREADS intra-op threads so $WORKERS x $THREADS never oversubscribes the cores (measured: the heaviest
# algo, CREPE, does not scale past 1 cpu thread, so THREADS=1 + WORKERS~=physical-cores is the knee and
# cannot run slower than serial). Resumable: finished result JSONs are skipped, so re-running (after adding
# an algorithm, or resuming an interrupted run) is cheap. Continues past a failing cell (logs a warning),
# so one bad dataset/algorithm never aborts an unattended run.
#
# Knobs (env): WORKERS (default = physical cores), THREADS (1), DEVICE (cpu; mps/cuda for GPU trackers),
# SKIP_DATASETS (space list, e.g. "AVID" to drop the 33-min sparse-voiced long pole).
#   uv sync --all-extras   # a partial install benchmarks only the algorithms whose extras are present
set -uo pipefail
cd "$(dirname "$0")"

PYTHON="uv run python"
CHIME_DIR="../datasets/chime_home"
DEMAND_DIR="../datasets/DEMAND"
OUTPUT_DIR="results"   # frame, note (notes_*), and speed (speed_*) cells all land here, keyed by metadata
DEVICE="${DEVICE:-cpu}"

# --- one cell (self-reentrant: the pool invokes `bash run.sh __cell "<spec>"`) --------------------------
# spec = kind|dataset|datadir|degradation|algo   (kind: clean | robust | note)
if [ "${1:-}" = "__cell" ]; then
  IFS='|' read -r kind ds dir deg algo <<< "$2"
  export OMP_NUM_THREADS="${THREADS:-1}" MKL_NUM_THREADS="${THREADS:-1}" OPENBLAS_NUM_THREADS="${THREADS:-1}" \
         VECLIB_MAXIMUM_THREADS="${THREADS:-1}" NUMEXPR_NUM_THREADS="${THREADS:-1}"
  case "$kind" in
    clean)  $PYTHON pitch_benchmark.py --dataset "$ds" --data-dir "$dir" --algorithms "$algo" \
              --degradation clean --device "$DEVICE" --output-dir "$OUTPUT_DIR" \
              || echo "[run.sh] WARN: clean $ds/$algo failed (continuing)" ;;
    robust) $PYTHON pitch_benchmark.py --dataset "$ds" --data-dir "$dir" --algorithms "$algo" \
              --degradation "$deg" --chime-dir "$CHIME_DIR" --demand-dir "$DEMAND_DIR" \
              --max-samples "${MAX_SAMPLES:-30}" --max-seconds "${MAX_SECONDS:-10}" \
              --device "$DEVICE" --output-dir "$OUTPUT_DIR" \
              || echo "[run.sh] WARN: robust $deg $ds/$algo failed (continuing)" ;;
    note)   $PYTHON note_benchmark.py --dataset "$ds" --data-dir "$dir" --algorithms "$algo" \
              --device "$DEVICE" --output-dir "$OUTPUT_DIR" --no-isolate \
              || echo "[run.sh] WARN: notes $ds/$algo failed (continuing)" ;;
  esac
  exit 0
fi

# --- config -------------------------------------------------------------------------------------------
WORKERS="${WORKERS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
export THREADS="${THREADS:-1}"
export MAX_SAMPLES="${MAX_SAMPLES:-30}"
export MAX_SECONDS="${MAX_SECONDS:-10}"
SKIP_DATASETS="${SKIP_DATASETS:-}"
# Frame datasets ORDERED LONGEST-AUDIO FIRST so the wall-clock long poles (a slow algo on a long-clip set)
# start immediately and the pool drains evenly; "name|data-dir" (pipe = paths may contain spaces).
# M4Singer intentionally excluded from every leaderboard (score-grade GT; see datasets/m4singer.py).
FRAME_DATASETS=(
  "AVID|../datasets/avid"
  "PTDB|../datasets/SPEECH DATA"
  "MDBStemSynth|../datasets/MDB-stem-synth"
  "OSFGlottis|../datasets/osf_glottis"
  "MIR1K|../datasets/MIR-1K"
  "SVD|../datasets/svd_zenodo/healthy"
  "URMP|../datasets/URMP"
  "NSynth|../datasets/nsynth-test"
  "APLAWD|../datasets/aplawd/APLAWDW"
  "SpeechSynth|datasets/speechsynth.pt"
  "Bach10Synth|../datasets/Bach10Synth/Bach10-mf0-synth"
  "Vocadito|../datasets/vocadito"
  "FDA|../datasets/FDA"
  "KEELE|../datasets/KEELE/KEELE"
)
NOTE_DATASETS=(
  "Vocadito|../datasets/vocadito"
  "URMP|../datasets/URMP"
)
CONDITIONS="clean white pink chime demand telephone reverb room"
ALGOS=$($PYTHON -c "from algorithms import get_available_algorithms as g; print(' '.join(g()))")
[ -z "$ALGOS" ] && { echo "[run.sh] FATAL: no algorithms installed (uv sync --all-extras)"; exit 1; }

skipped() { case " $SKIP_DATASETS " in *" $1 "*) return 0;; *) return 1;; esac; }

# --- build the cell list (clean long-poles first, then capped robustness, then notes) ------------------
CELLS="$(mktemp)"; trap 'rm -f "$CELLS"' EXIT
for d in "${FRAME_DATASETS[@]}"; do
  name="${d%%|*}"; dir="${d#*|}"; skipped "$name" && continue
  for a in $ALGOS; do echo "clean|$name|$dir||$a"; done
done >> "$CELLS"
for d in "${FRAME_DATASETS[@]}"; do
  name="${d%%|*}"; dir="${d#*|}"; skipped "$name" && continue
  for c in $CONDITIONS; do for a in $ALGOS; do [ "$c" = clean ] && continue; echo "robust|$name|$dir|$c|$a"; done; done
done >> "$CELLS"
for d in "${NOTE_DATASETS[@]}"; do
  name="${d%%|*}"; dir="${d#*|}"; skipped "$name" && continue
  for a in $ALGOS; do echo "note|$name|$dir||$a"; done
done >> "$CELLS"

N=$(wc -l < "$CELLS" | tr -d ' ')
echo "=== PARALLEL BENCHMARK: $N cells | $WORKERS workers x $THREADS threads | device=$DEVICE ==="
echo "    algos: $ALGOS"
[ -n "$SKIP_DATASETS" ] && echo "    skipping datasets: $SKIP_DATASETS"
SECONDS=0
xargs -P "$WORKERS" -I{} bash "$0" __cell "{}" < "$CELLS"
echo "=== FRAME+NOTE cells done in ${SECONDS}s ($((SECONDS/3600))h$(((SECONDS%3600)/60))m) ==="

# --- serial tail: speed, OOD, report (each reads the cells written above) ------------------------------
echo "=== SPEED (synthetic timing) ==="
case "$DEVICE" in
  cuda|mps) $PYTHON speed_benchmark.py --output-dir "$OUTPUT_DIR" --devices cpu "$DEVICE" ;;
  *)        $PYTHON speed_benchmark.py --output-dir "$OUTPUT_DIR" --devices cpu ;;
esac
echo "=== OOD generalization (synthetic, exact labels) ==="
$PYTHON ood_benchmark.py --output-dir "$OUTPUT_DIR" --device "$DEVICE" || echo "[run.sh] WARN: OOD failed"
echo "=== REPORT (frame + notes) ==="
$PYTHON generate_report.py --results-dir "$OUTPUT_DIR"
