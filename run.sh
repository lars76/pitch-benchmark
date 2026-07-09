#!/usr/bin/env bash
# Run the complete pitch benchmark: every dataset x all algorithms, then build the report.
# Resumable: finished result JSONs are skipped, so re-running after adding an algorithm is cheap.
# Needs every backend installed: run `uv sync --all-extras` first. With a partial install the
# runner benchmarks only the algorithms whose extras are present.
set -euo pipefail
cd "$(dirname "$0")"

# --- paths (edit these to your machine) -------------------------------------
PYTHON="uv run python"
CHIME_DIR="../datasets/chime_home"
DEMAND_DIR="../datasets/DEMAND"
OUTPUT_DIR="results"
# Pin the device so leaderboard numbers are reproducible (the runner itself defaults to auto for
# interactive use). cpu is the portable reference; on the CUDA box run `DEVICE=cuda ./run.sh`.
DEVICE="${DEVICE:-cpu}"

PTDB="../datasets/SPEECH DATA"
NSYNTH="../datasets/nsynth-test"
MDBSTEMSYNTH="../datasets/MDB-stem-synth"
# SpeechSynth is a LightSpeech TTS checkpoint (synthesizes speech; the conditioned f0 is vocoder-rendered, so labels are faithful to within ~tens of cents), not a folder.
SPEECHSYNTH="datasets/speechsynth.pt"
MIR1K="../datasets/MIR-1K"
VOCADITO="../datasets/vocadito"
BACH10SYNTH="../datasets/Bach10Synth/Bach10-mf0-synth"
# ---------------------------------------------------------------------------

# Robustness probe: small capped + truncated sample, run across each degradation.
MAX_SAMPLES=30
MAX_SECONDS=10
# Minimal-but-expressive set: additive family (white/pink + real provenances chime/demand) reported
# per-source, plus low-cut (telephone) and reverb/room.
CONDITIONS="clean white pink chime demand telephone reverb room"

run() {  # run <dataset-name> <data-dir>  -- clean leaderboard (full dataset)
  $PYTHON pitch_benchmark.py --dataset "$1" --data-dir "$2" \
    --degradation clean --device "$DEVICE" --output-dir "$OUTPUT_DIR"
}

robust() {  # robust <dataset-name> <data-dir>  -- probe x conditions
  for cond in $CONDITIONS; do
    $PYTHON pitch_benchmark.py --dataset "$1" --data-dir "$2" \
      --degradation "$cond" --chime-dir "$CHIME_DIR" --demand-dir "$DEMAND_DIR" \
      --max-samples "$MAX_SAMPLES" --max-seconds "$MAX_SECONDS" \
      --device "$DEVICE" --output-dir "$OUTPUT_DIR"
  done
}

# --- clean leaderboard (full datasets) ---
run Bach10Synth   "$BACH10SYNTH"
run MDBStemSynth  "$MDBSTEMSYNTH"
run MIR1K         "$MIR1K"
run NSynth        "$NSYNTH"
run PTDB          "$PTDB"
run SpeechSynth   "$SPEECHSYNTH"
run Vocadito      "$VOCADITO"

# --- robustness (probe x degradations) ---
robust Bach10Synth   "$BACH10SYNTH"
robust MDBStemSynth  "$MDBSTEMSYNTH"
robust MIR1K         "$MIR1K"
robust NSynth        "$NSYNTH"
robust PTDB          "$PTDB"
robust SpeechSynth   "$SPEECHSYNTH"
robust Vocadito      "$VOCADITO"

# --- speed (synthetic timing; no datasets needed) ---
# Forward the DEVICE knob (speed_benchmark has no "auto"; cpu is always timed as the reference).
case "$DEVICE" in
  cuda|mps) $PYTHON speed_benchmark.py --output-dir "$OUTPUT_DIR" --devices cpu "$DEVICE" ;;
  *)        $PYTHON speed_benchmark.py --output-dir "$OUTPUT_DIR" --devices cpu ;;
esac

# --- OOD generalization (synthetic signals with exact labels; each cell isolated in a subprocess) ---
$PYTHON ood_benchmark.py --output-dir "$OUTPUT_DIR" --device "$DEVICE"

# build the markdown report from all result JSONs
$PYTHON generate_report.py --results-dir "$OUTPUT_DIR"
