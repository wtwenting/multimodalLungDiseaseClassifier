"""Audio validation utilities for COUGHVID."""

import csv
from collections import Counter
from pathlib import Path
import json
import subprocess
from statistics import mean


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

def probe_audio_file(audio_path: Path) -> dict[str, object]:
    """Inspect one audio file using FFprobe."""

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json",
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    streams = data.get("streams", [])

    if not streams:
        raise ValueError("No audio stream found.")

    stream = streams[0]
    format_info = data.get("format", {})

    return {
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "duration": float(format_info.get("duration", 0)),
    }

def analyze_audio_content(
    metadata_path: Path,
    audio_dir: Path,
    limit: int | None = 100,
) -> dict[str, object]:
    """Inspect audio properties using FFprobe."""

    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    codecs: Counter[str] = Counter()

    durations: list[float] = []

    successful = 0
    failed = 0

    failures: list[tuple[str, str]] = []

    audio_paths: list[Path] = []

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        for row in reader:
            uuid = row["uuid"].strip()

            matches = [
                path
                for path in audio_dir.glob(f"{uuid}.*")
                if path.suffix.lower() in AUDIO_EXTENSIONS
            ]

            if matches:
                audio_paths.append(matches[0])

    if limit is not None:
        audio_paths = audio_paths[:limit]

    for audio_path in audio_paths:

        try:
            info = probe_audio_file(audio_path)

            successful += 1

            sample_rates[info["sample_rate"]] += 1
            channels[info["channels"]] += 1
            codecs[info["codec"]] += 1

            if info["duration"] > 0:
                durations.append(info["duration"])

        except (
            subprocess.CalledProcessError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:

            failed += 1
            failures.append(
                (audio_path.name, str(exc))
            )

    duration_stats = {}

    if durations:
        duration_stats = {
            "minimum": min(durations),
            "maximum": max(durations),
            "average": mean(durations),
        }

    return {
        "files_checked": len(audio_paths),
        "successful": successful,
        "failed": failed,
        "sample_rates": dict(sample_rates),
        "channels": dict(channels),
        "codecs": dict(codecs),
        "duration": duration_stats,
        "failures": failures,
    }

def standardize_audio_file(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    ) -> None:
    """Convert an audio file to standardized mono PCM WAV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-v", "error",
        "-y",
        "-i", str(input_path),

        # Ignore any video stream.
        "-vn",

        # Convert to mono.
        "-ac", "1",

        # Resample.
        "-ar", str(sample_rate),

        # 16-bit PCM.
        "-acodec", "pcm_s16le",

        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

def standardize_audio_dataset(
    metadata_path: Path,
    audio_dir: Path,
    output_dir: Path,
    sample_rate: int = 16000,
    limit: int | None = None,
    ) -> dict[str, object]:
    """Standardize cleaned COUGHVID audio to mono PCM WAV."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    converted = 0
    skipped = 0
    failed = 0

    failures: list[tuple[str, str]] = []

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        for row in reader:

            if limit is not None and total >= limit:
                break

            total += 1

            uuid = row["uuid"].strip()

            matches = [
                path
                for path in audio_dir.glob(f"{uuid}.*")
                if path.suffix.lower() in AUDIO_EXTENSIONS
            ]

            if not matches:
                failed += 1
                failures.append(
                    (uuid, "Source audio not found")
                )
                continue

            input_path = matches[0]
            output_path = output_dir / f"{uuid}.wav"

            # Allows safe re-running.
            if output_path.exists() and output_path.stat().st_size > 0:
                skipped += 1
                continue

            try:
                standardize_audio_file(
                    input_path=input_path,
                    output_path=output_path,
                    sample_rate=sample_rate,
                )

                converted += 1

            except subprocess.CalledProcessError as exc:
                failed += 1

                failures.append(
                    (
                        input_path.name,
                        exc.stderr.strip(),
                    )
                )

    return {
        "total": total,
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "sample_rate": sample_rate,
        "failures": failures,
    }