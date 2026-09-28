#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-data/experiment_p0_p4}"

if [[ ! -f "${OUTPUT_DIR}/metadata_split.csv" ]]; then
    echo "Missing ${OUTPUT_DIR}/metadata_split.csv" >&2
    echo "Create one split first, then rerun this script." >&2
    exit 1
fi

for experiment in p0 p1 p2 p3 p4; do
    echo "=== Training ${experiment} ==="
    "${PYTHON_BIN}" -m psupr_proj \
        "--train-${experiment}" \
        --output-dir "${OUTPUT_DIR}"

    echo "=== Evaluating ${experiment} ==="
    "${PYTHON_BIN}" -m psupr_proj \
        --evaluate-model "${experiment}_balanced_sampling" \
        --output-dir "${OUTPUT_DIR}"
done

echo "All P0-P4 experiments completed in ${OUTPUT_DIR}."
