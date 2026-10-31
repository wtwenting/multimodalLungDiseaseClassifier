#!/usr/bin/env bash
# Reproducible overnight COUGHVID + HeAR experiment launcher.
# Run from the repository root: bash run_coughvid_hear_overnight.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/coughvid_hear_experiments}"
mkdir -p "$OUTPUT_DIR"

# On macOS, prevent idle sleep during the long H2 feature-extraction stage.
if command -v caffeinate >/dev/null 2>&1 && [[ "${COUGHVID_CAFFEINATED:-0}" != "1" ]]; then
    export COUGHVID_CAFFEINATED=1
    exec caffeinate -i bash "$0" "$@"
fi

# Save a full transcript while preserving the Python process exit code.
"$PYTHON_BIN" src/psupr_proj/run_coughvid_hear_h0_h2.py \
    --output-dir "$OUTPUT_DIR" \
    "$@" 2>&1 | tee -a "$OUTPUT_DIR/overnight.log"
