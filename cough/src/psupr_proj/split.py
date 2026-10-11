"""Dataset splitting utilities for COUGHVID."""

import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


def create_stratified_split(
    metadata_path: Path,
    audio_dir: Path,
    output_path: Path,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, object]:
    """Create a stratified train/validation/test split."""

    # Make sure percentages add up to 100%.
    total_size = train_size + validation_size + test_size

    if abs(total_size - 1.0) > 1e-9:
        raise ValueError(
            "train_size + validation_size + test_size "
            "must equal 1.0"
        )

    # ---------------------------------------------------------
    # 1. Read cleaned metadata
    # ---------------------------------------------------------

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        if not reader.fieldnames:
            raise ValueError("Metadata CSV has no header.")

        rows = list(reader)

    # ---------------------------------------------------------
    # 2. Verify standardized WAV exists
    # ---------------------------------------------------------

    valid_rows = []
    missing_audio = []

    for row in rows:

        uuid = row["uuid"].strip()
        status = row["status"].strip()

        if status not in {
            "healthy",
            # "symptomatic",
            "COVID-19",
        }:
            continue

        wav_path = audio_dir / f"{uuid}.wav"

        if not wav_path.exists():
            missing_audio.append(uuid)
            continue

        # Add WAV location to metadata.
        row["audio_path"] = str(wav_path)
        row["status"] = status

        valid_rows.append(row)

    if not valid_rows:
        raise ValueError(
            "No records with standardized audio were found."
        )

    # ---------------------------------------------------------
    # 3. First split:
    #
    # 70% Train
    # 30% Temporary
    # ---------------------------------------------------------

    labels = [
        row["status"]
        for row in valid_rows
    ]

    train_rows, temp_rows = train_test_split(
        valid_rows,
        test_size=(validation_size + test_size),
        random_state=random_state,
        stratify=labels,
    )

    # ---------------------------------------------------------
    # 4. Split temporary set:
    #
    # 15% Validation
    # 15% Test
    # ---------------------------------------------------------

    temp_labels = [
        row["status"]
        for row in temp_rows
    ]

    relative_test_size = (
        test_size /
        (validation_size + test_size)
    )

    validation_rows, test_rows = train_test_split(
        temp_rows,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_labels,
    )

    # ---------------------------------------------------------
    # 5. Add permanent split labels
    # ---------------------------------------------------------

    for row in train_rows:
        row["split"] = "train"

    for row in validation_rows:
        row["split"] = "validation"

    for row in test_rows:
        row["split"] = "test"

    all_rows = (
        train_rows
        + validation_rows
        + test_rows
    )

    # ---------------------------------------------------------
    # 6. Write metadata_split.csv
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(reader.fieldnames)

    if "audio_path" not in fieldnames:
        fieldnames.append("audio_path")

    if "split" not in fieldnames:
        fieldnames.append("split")

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
        writer.writerows(all_rows)

    # ---------------------------------------------------------
    # 7. Generate report
    # ---------------------------------------------------------

    def class_counts(
        subset: list[dict[str, str]],
    ) -> dict[str, int]:

        return dict(
            Counter(
                row["status"]
                for row in subset
            )
        )

    return {
        "total": len(valid_rows),
        "missing_audio": len(missing_audio),

        "train": len(train_rows),
        "validation": len(validation_rows),
        "test": len(test_rows),

        "train_classes": class_counts(train_rows),
        "validation_classes": class_counts(validation_rows),
        "test_classes": class_counts(test_rows),

        "random_state": random_state,
    }


def create_coswara_stratified_split(
    metadata_path: Path,
    output_path: Path,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, object]:
    """Create a participant-level stratified split for prepared Coswara metadata.

    Unlike the COUGHVID splitter above, Coswara metadata already contains the
    correct audio_path for each participant. This function preserves that path
    rather than rebuilding it from uuid.
    """

    total_size = train_size + validation_size + test_size

    if abs(total_size - 1.0) > 1e-9:
        raise ValueError(
            "train_size + validation_size + test_size "
            "must equal 1.0"
        )

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        reader = csv.DictReader(metadata_file)

        if not reader.fieldnames:
            raise ValueError("Metadata CSV has no header.")

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required_columns = {
        "uuid",
        "status",
        "audio_path",
    }

    missing_columns = required_columns - set(fieldnames)

    if missing_columns:
        raise ValueError(
            "Coswara metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    valid_rows = []
    missing_audio = []
    invalid_status = 0
    seen_uuids = set()
    duplicate_uuids = []

    for row in rows:
        uuid = row.get("uuid", "").strip()
        status = row.get("status", "").strip()
        audio_path_raw = row.get("audio_path", "").strip()

        if status not in {
            "healthy",
            "COVID-19",
        }:
            invalid_status += 1
            continue

        if not uuid:
            continue

        # Protect participant-level independence. For the current heavy-only
        # experiment each participant should occur exactly once.
        if uuid in seen_uuids:
            duplicate_uuids.append(uuid)
            continue

        seen_uuids.add(uuid)

        audio_path = Path(audio_path_raw)

        if not audio_path.exists():
            missing_audio.append(uuid)
            continue

        row["uuid"] = uuid
        row["status"] = status
        row["audio_path"] = str(audio_path)

        valid_rows.append(row)

    if duplicate_uuids:
        raise ValueError(
            "Duplicate Coswara participant IDs found. "
            "Participant-level splitting requires one row per participant. "
            f"Examples: {duplicate_uuids[:5]}"
        )

    if not valid_rows:
        raise ValueError(
            "No valid Coswara records with audio were found."
        )

    labels = [
        row["status"]
        for row in valid_rows
    ]

    train_rows, temp_rows = train_test_split(
        valid_rows,
        test_size=(validation_size + test_size),
        random_state=random_state,
        stratify=labels,
    )

    temp_labels = [
        row["status"]
        for row in temp_rows
    ]

    relative_test_size = (
        test_size
        / (validation_size + test_size)
    )

    validation_rows, test_rows = train_test_split(
        temp_rows,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_labels,
    )

    for row in train_rows:
        row["split"] = "train"

    for row in validation_rows:
        row["split"] = "validation"

    for row in test_rows:
        row["split"] = "test"

    all_rows = (
        train_rows
        + validation_rows
        + test_rows
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if "split" not in fieldnames:
        fieldnames.append("split")

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
        writer.writerows(all_rows)

    def class_counts(
        subset: list[dict[str, str]],
    ) -> dict[str, int]:
        return dict(
            Counter(
                row["status"]
                for row in subset
            )
        )

    report = {
        "total": len(valid_rows),
        "missing_audio": len(missing_audio),
        "invalid_status": invalid_status,
        "train": len(train_rows),
        "validation": len(validation_rows),
        "test": len(test_rows),
        "train_classes": class_counts(train_rows),
        "validation_classes": class_counts(validation_rows),
        "test_classes": class_counts(test_rows),
        "random_state": random_state,
    }

    print("\n" + "=" * 60)
    print("COSWARA STRATIFIED SPLIT")
    print("=" * 60)
    print(f"\nInput:       {metadata_path}")
    print(f"Output:      {output_path}")
    print(f"Total:       {report['total']:,}")
    print(f"Train:       {report['train']:,}")
    print(f"Validation:  {report['validation']:,}")
    print(f"Test:        {report['test']:,}")
    print(f"\nTrain classes:      {report['train_classes']}")
    print(f"Validation classes: {report['validation_classes']}")
    print(f"Test classes:       {report['test_classes']}")
    print(f"\nMissing audio: {report['missing_audio']:,}")
    print(f"Invalid status: {report['invalid_status']:,}")
    print(f"Random state: {random_state}")

    return report
