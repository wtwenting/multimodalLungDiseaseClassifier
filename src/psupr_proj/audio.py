"""Audio validation utilities for COUGHVID."""

import csv
from collections import Counter
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".webm",
    ".ogg",
    ".wav",
    ".mp3",
    ".m4a",
}


def analyze_audio_files(
    metadata_path: Path,
    audio_dir: Path,
) -> dict[str, object]:
    """Check whether cleaned metadata records have audio files."""

    total_records = 0
    found_records = 0
    missing_records = 0
    zero_byte_files = 0

    extension_counts: Counter[str] = Counter()

    missing_uuids: list[str] = []
    zero_byte_uuids: list[str] = []

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        for row in reader:
            total_records += 1

            uuid = row["uuid"].strip()

            matches = [
                path
                for path in audio_dir.glob(f"{uuid}.*")
                if path.suffix.lower() in AUDIO_EXTENSIONS
            ]

            if not matches:
                missing_records += 1
                missing_uuids.append(uuid)
                continue

            found_records += 1

            for audio_path in matches:
                extension_counts[
                    audio_path.suffix.lower()
                ] += 1

                if audio_path.stat().st_size == 0:
                    zero_byte_files += 1
                    zero_byte_uuids.append(uuid)

    return {
        "total_records": total_records,
        "found_records": found_records,
        "missing_records": missing_records,
        "zero_byte_files": zero_byte_files,
        "extension_counts": dict(extension_counts),
        "missing_uuids": missing_uuids,
        "zero_byte_uuids": zero_byte_uuids,
    }