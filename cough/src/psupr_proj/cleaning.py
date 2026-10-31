"""Metadata analysis and cleaning utilities for COUGHVID."""

import csv
from collections import Counter
from pathlib import Path


def analyze_metadata(path: Path) -> None:
    """Analyse filtered COUGHVID metadata before cleaning."""

    with path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)

        if not reader.fieldnames:
            raise ValueError("Metadata CSV has no header.")

        rows = list(reader)

    print("\n" + "=" * 60)
    print("COUGHVID METADATA ANALYSIS")
    print("=" * 60)

    print(f"\nTotal records: {len(rows):,}")
    print(f"Total columns: {len(reader.fieldnames)}")

    print("\n=== Columns ===")
    for column in reader.fieldnames:
        print(f"- {column}")

    analyze_missing_values(rows, reader.fieldnames)
    analyze_duplicates(rows)
    analyze_column(rows, "status")
    analyze_column(rows, "status_SSL")
    analyze_column(rows, "gender")
    analyze_column(rows, "respiratory_condition")
    analyze_numeric_column(rows, "age")
    analyze_numeric_column(rows, "cough_detected")


def analyze_missing_values(
    rows: list[dict[str, str]],
    columns: list[str],
) -> None:
    """Report missing values for every metadata column."""

    print("\n=== Missing Values ===")

    total = len(rows)

    missing_counts = {}

    for column in columns:
        missing = sum(
            1
            for row in rows
            if not row.get(column, "").strip()
        )

        missing_counts[column] = missing

    for column, missing in sorted(
        missing_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        percentage = (
            (missing / total * 100)
            if total
            else 0
        )

        print(
            f"{column:30} "
            f"{missing:8,} "
            f"({percentage:6.2f}%)"
        )


def analyze_duplicates(
    rows: list[dict[str, str]],
) -> None:
    """Check duplicate UUIDs."""

    print("\n=== UUID Analysis ===")

    uuids = [
        row.get("uuid", "").strip()
        for row in rows
        if row.get("uuid", "").strip()
    ]

    counts = Counter(uuids)

    duplicates = {
        uuid: count
        for uuid, count in counts.items()
        if count > 1
    }

    print(f"UUID values:       {len(uuids):,}")
    print(f"Unique UUIDs:      {len(counts):,}")
    print(f"Duplicate UUIDs:   {len(duplicates):,}")

    if duplicates:
        print("\nExample duplicates:")

        for uuid, count in list(duplicates.items())[:10]:
            print(f"  {uuid}: {count}")


def analyze_column(
    rows: list[dict[str, str]],
    column: str,
) -> None:
    """Display value distribution for a categorical column."""

    if not rows or column not in rows[0]:
        print(f"\n=== {column} ===")
        print("Column not present.")
        return

    values = [
        row.get(column, "").strip() or "<MISSING>"
        for row in rows
    ]

    counts = Counter(values)

    print(f"\n=== {column} Distribution ===")

    for value, count in counts.most_common():
        percentage = count / len(rows) * 100

        print(
            f"{value:30} "
            f"{count:8,} "
            f"({percentage:6.2f}%)"
        )


def analyze_numeric_column(
    rows: list[dict[str, str]],
    column: str,
) -> None:
    """Display simple statistics for a numeric metadata column."""

    if not rows or column not in rows[0]:
        print(f"\n=== {column} ===")
        print("Column not present.")
        return

    values = []

    invalid = 0

    for row in rows:
        raw_value = row.get(column, "").strip()

        if not raw_value:
            continue

        try:
            values.append(float(raw_value))
        except ValueError:
            invalid += 1

    print(f"\n=== {column} Statistics ===")

    if not values:
        print("No valid numeric values.")
        return

    print(f"Valid:       {len(values):,}")
    print(f"Invalid:     {invalid:,}")
    print(f"Minimum:     {min(values):.4f}")
    print(f"Maximum:     {max(values):.4f}")
    print(f"Average:     {sum(values) / len(values):.4f}")

def clean_metadata(
    input_path: Path,
    output_path: Path,
    threshold: float = 0.8,
) -> dict[str, int]:
    """Clean metadata using the experiment's cough-score threshold."""

    valid_statuses = {
        "healthy",
        # "symptomatic",
        "COVID-19",
    }

    with input_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        reader = csv.DictReader(metadata_file)

        if not reader.fieldnames:
            raise ValueError("Metadata CSV has no header.")

        # Remove empty / unnamed CSV columns.
        fieldnames = [
            column
            for column in reader.fieldnames
            if column.strip()
        ]

        cleaned_rows = []

        total = 0
        missing_uuid = 0
        missing_status = 0
        invalid_status = 0
        invalid_cough_score = 0

        for row in reader:
            total += 1

            uuid = row.get("uuid", "").strip()

            if not uuid:
                missing_uuid += 1
                continue

            status = row.get("status", "").strip()

            if not status:
                missing_status += 1
                continue

            if status not in valid_statuses:
                invalid_status += 1
                continue

            raw_score = row.get(
                "cough_detected",
                "",
            ).strip()

            try:
                cough_score = float(raw_score)
            except ValueError:
                invalid_cough_score += 1
                continue

            # Reapply the threshold for reproducibility if the input was
            # generated by a different preprocessing step.
            if cough_score < threshold:
                invalid_cough_score += 1
                continue

            cleaned_row = {
                key: value
                for key, value in row.items()
                if key and key.strip()
            }

            cleaned_rows.append(cleaned_row)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(cleaned_rows)

    return {
        "input_records": total,
        "clean_records": len(cleaned_rows),
        "missing_uuid": missing_uuid,
        "missing_status": missing_status,
        "invalid_status": invalid_status,
        "invalid_cough_score": invalid_cough_score,
    }