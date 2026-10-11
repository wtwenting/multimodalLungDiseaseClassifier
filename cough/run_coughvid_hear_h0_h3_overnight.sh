#!/usr/bin/env bash
# Run existing H0 baseline (resume-safe), then new H3 regularization experiments.
# Place at repository root, next to run_coughvid_hear_overnight.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p data/coughvid_hear_h3_regularization
if command -v caffeinate >/dev/null 2>&1 && [[ "${COUGHVID_CAFFEINATED:-0}" != "1" ]]; then
  export COUGHVID_CAFFEINATED=1
  exec caffeinate -i bash "$0" "$@"
fi
# Original H0 artifacts are preserved and skipped if already complete.
"$PYTHON_BIN" src/psupr_proj/run_coughvid_hear_h0_h2.py --stages h0 2>&1 | tee -a data/coughvid_hear_h3_regularization/overnight.log
# H3 writes to its own folder, never overwriting H0/H1/H2.
"$PYTHON_BIN" src/psupr_proj/run_coughvid_hear_h3.py "$@" 2>&1 | tee -a data/coughvid_hear_h3_regularization/overnight.log
