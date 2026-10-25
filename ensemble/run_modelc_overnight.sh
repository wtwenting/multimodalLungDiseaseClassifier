#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p output/xray_modelc_baseline/logs
stamp=$(date +%Y%m%d_%H%M%S)
log="output/xray_modelc_baseline/logs/modelc_${stamp}.log"
echo "Logging to: $log"
if [ -f output/xray_modelc_baseline/cleaning_decisions.csv ]; then
  set -- --cleaning output/xray_modelc_baseline/cleaning_decisions.csv
else
  set --
fi
if command -v caffeinate >/dev/null 2>&1; then
  nohup caffeinate -i ../.venv/bin/python -u src/benchmark_modelc_baseline.py "$@" > "$log" 2>&1 < /dev/null &
else
  nohup ../.venv/bin/python -u src/benchmark_modelc_baseline.py "$@" > "$log" 2>&1 < /dev/null &
fi
echo "PID: $!"
echo "Monitor: tail -f $log"
