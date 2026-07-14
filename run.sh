#!/usr/bin/env bash
# THE single entry point: run the COMPLETE benchmark and build the report. Every registered dataset x
# all algorithms across BOTH tracks -- frame (clean leaderboard + robustness probe) AND note
# transcription -- plus speed and OOD. Run this to get all the numbers.
#
# Resumable: finished result JSONs are skipped, so re-running (e.g. after adding an algorithm) is cheap.
# Needs every backend installed: `uv sync --all-extras` (a partial install benchmarks only the
# algorithms whose extras are present). Continues past a failing dataset/algorithm, logging a warning,
# so one bad cell never aborts an unattended run.
set -uo pipefail
cd "$(dirname "$0")"

# --- paths (edit these to your machine) -------------------------------------
PYTHON="uv run python"
CHIME_DIR="../datasets/chime_home"
DEMAND_DIR="../datasets/DEMAND"
OUTPUT_DIR="results"
NOTES_DIR="results_notes"
# Pin the device so leaderboard numbers are reproducible (the runner defaults to auto interactively).
# cpu is the portable reference; DEVICE=cuda ./run.sh on a CUDA box; DEVICE=mps ./run.sh = faster
# Apple-GPU overview (neural-tracker numbers become mps-specific, so not the cpu reference board).
DEVICE="${DEVICE:-cpu}"

# Frame datasets, "name|data-dir" (pipe-separated so paths may contain spaces). Ordered small->large so
# the report fills in early; the giants (PTDB/NSynth/SVD/APLAWD) are the long pole. Covers the full
# held-out gate set + the two training-domain sets (PTDB, MDBStemSynth) kept as disclosed "home data".
# M4Singer is intentionally excluded from every leaderboard (score-grade GT; see datasets/m4singer.py).
FRAME_DATASETS=(
  "KEELE|../datasets/KEELE/KEELE"
  "FDA|../datasets/FDA"
  "Vocadito|../datasets/vocadito"
  "Bach10Synth|../datasets/Bach10Synth/Bach10-mf0-synth"
  "SpeechSynth|datasets/speechsynth.pt"
  "OSFGlottis|../datasets/osf_glottis"
  "AVID|../datasets/avid"
  "MIR1K|../datasets/MIR-1K"
  "URMP|../datasets/URMP"
  "MDBStemSynth|../datasets/MDB-stem-synth"
  "PTDB|../datasets/SPEECH DATA"
  "SVD|../datasets/svd_zenodo/healthy"
  "NSynth|../datasets/nsynth-test"
  "APLAWD|../datasets/aplawd/APLAWDW"
)
# Note-track datasets (== datasets.list_note_datasets() minus M4Singer, which policy excludes).
NOTE_DATASETS=(
  "Vocadito|../datasets/vocadito"
  "URMP|../datasets/URMP"
)

# Robustness probe: small capped + truncated sample, run across each degradation.
MAX_SAMPLES=30
MAX_SECONDS=10
CONDITIONS="clean white pink chime demand telephone reverb room"

frame_clean() {  # <name> <data-dir>  -- full-dataset clean leaderboard cell
  $PYTHON pitch_benchmark.py --dataset "$1" --data-dir "$2" \
    --degradation clean --device "$DEVICE" --output-dir "$OUTPUT_DIR" \
    || echo "[run.sh] WARN: frame clean $1 failed (continuing)"
}
frame_robust() {  # <name> <data-dir>  -- capped probe x each degradation
  for cond in $CONDITIONS; do
    $PYTHON pitch_benchmark.py --dataset "$1" --data-dir "$2" \
      --degradation "$cond" --chime-dir "$CHIME_DIR" --demand-dir "$DEMAND_DIR" \
      --max-samples "$MAX_SAMPLES" --max-seconds "$MAX_SECONDS" \
      --device "$DEVICE" --output-dir "$OUTPUT_DIR" \
      || echo "[run.sh] WARN: frame $cond $1 failed (continuing)"
  done
}
note_clean() {  # <name> <data-dir>  -- note-transcription track (per-algorithm theta x lambda sweep)
  $PYTHON note_benchmark.py --dataset "$1" --data-dir "$2" \
    --device "$DEVICE" --output-dir "$NOTES_DIR" \
    || echo "[run.sh] WARN: notes $1 failed (continuing)"
}

echo "=== FRAME: clean leaderboard (${#FRAME_DATASETS[@]} datasets, full) ==="
for d in "${FRAME_DATASETS[@]}"; do frame_clean "${d%%|*}" "${d#*|}"; done
echo "=== FRAME: robustness probe (capped x ${CONDITIONS}) ==="
for d in "${FRAME_DATASETS[@]}"; do frame_robust "${d%%|*}" "${d#*|}"; done
echo "=== NOTE TRACK (clean) ==="
for d in "${NOTE_DATASETS[@]}"; do note_clean "${d%%|*}" "${d#*|}"; done

echo "=== SPEED (synthetic timing) ==="
case "$DEVICE" in
  cuda|mps) $PYTHON speed_benchmark.py --output-dir "$OUTPUT_DIR" --devices cpu "$DEVICE" ;;
  *)        $PYTHON speed_benchmark.py --output-dir "$OUTPUT_DIR" --devices cpu ;;
esac

echo "=== OOD generalization (synthetic, exact labels) ==="
$PYTHON ood_benchmark.py --output-dir "$OUTPUT_DIR" --device "$DEVICE" || echo "[run.sh] WARN: OOD failed"

echo "=== REPORT (frame + notes) ==="
$PYTHON generate_report.py --results-dir "$OUTPUT_DIR" --notes-dir "$NOTES_DIR"
