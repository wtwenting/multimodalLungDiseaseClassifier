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

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
rows = []

for experiment in ("p0", "p1", "p2", "p3", "p4"):
    report_path = output_dir / "models" / f"{experiment}_balanced_sampling.evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    covid = report["classification_report"]["COVID-19"]
    rows.append({
        "Experiment": experiment.upper(),
        "COVID Recall": covid["recall"],
        "COVID Precision": covid["precision"],
        "COVID F1": covid["f1-score"],
        "Macro F1": report["macro_f1"],
        "Accuracy": report["accuracy"],
        "Best epoch": report["best_epoch"],
        "Confusion matrix": json.dumps(report["confusion_matrix"]),
    })

with (output_dir / "p0_p4_summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
    writer = csv.DictWriter(summary_file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

(output_dir / "p0_p4_summary.json").write_text(
    json.dumps(rows, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Summary CSV saved to: {output_dir / 'p0_p4_summary.csv'}")
print(f"Summary JSON saved to: {output_dir / 'p0_p4_summary.json'}")
PY

echo "All P0-P4 experiments completed in ${OUTPUT_DIR}."
