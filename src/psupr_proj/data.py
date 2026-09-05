"""Utilities for reading CoughVID metadata."""

import csv
import shutil
from pathlib import Path


def default_metadata_path() -> Path:
    """Return the repository's default metadata file path."""
    return Path("data/coughvid_20211012/metadata_compiled.csv")


def summarize_metadata(path: Path) -> dict[str, int | float]:
    """Summarize metadata rows and cough-detection scores."""
    rows = 0
    scores: list[float] = []
    with path.open(newline="", encoding="utf-8") as metadata_file:
        for row in csv.DictReader(metadata_file):
            rows += 1
            score = row.get("cough_detected", "").strip()
            if score:
                scores.append(float(score))

    summary: dict[str, int | float] = {
        "records": rows,
        "scored_records": len(scores),
    }
    if scores:
        summary["average_cough_score"] = sum(scores) / len(scores)
    return summary


def filter_dataset(
    metadata_path: Path,
    source_dir: Path,
    output_dir: Path,
    threshold: float = 0.8,
) -> dict[str, int]:
    """Copy files whose metadata cough score meets ``threshold``.

    The filtered metadata is written to ``metadata_filtered.csv`` in the
    output directory. Files are matched by the metadata row's UUID, allowing
    audio files and JSON sidecars to be copied together.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, str]] = []
    copied_files = 0
    missing_files = 0

    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if not reader.fieldnames or "uuid" not in reader.fieldnames:
            raise ValueError("Metadata CSV must contain a 'uuid' column")
        if "cough_detected" not in reader.fieldnames:
            raise ValueError("Metadata CSV must contain a 'cough_detected' column")

        for row in reader:
            score = row.get("cough_detected", "").strip()
            if not score or float(score) < threshold:
                continue
            selected_rows.append(row)
            uuid = row["uuid"].strip()
            matching_files = list(source_dir.glob(f"{uuid}.*"))
            if not matching_files:
                missing_files += 1
            for source_file in matching_files:
                destination = output_dir / source_file.name
                if not destination.exists() or destination.stat().st_size != source_file.stat().st_size:
                    shutil.copy2(source_file, destination)
                    copied_files += 1

    filtered_metadata = output_dir / "metadata_filtered.csv"
    with filtered_metadata.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    return {
        "records": len(selected_rows),
        "files": copied_files,
        "missing_files": missing_files,
    }