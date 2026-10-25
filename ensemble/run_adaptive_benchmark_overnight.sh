#!/bin/sh
set -u
cd "$(dirname "$0")"
mkdir -p output/xray_adaptive_benchmark/logs
LOG="output/xray_adaptive_benchmark/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"
PYTHON="../.venv/bin/python"
echo "Logging to: $LOG"
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i "$PYTHON" -u src/benchmark_adaptive_xray.py > "$LOG" 2>&1
else
  "$PYTHON" -u src/benchmark_adaptive_xray.py > "$LOG" 2>&1
fi
STATUS=$?
echo "Exit status: $STATUS | Log: $LOG"
