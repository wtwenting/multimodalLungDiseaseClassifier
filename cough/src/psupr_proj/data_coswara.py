"""Coswara dataset loading and preparation utilities."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------
# Coswara label mapping for the binary classification task
# ---------------------------------------------------------

STATUS_MAP = {
    "healthy": "healthy",
    "positive_asymp": "COVID-19",
    "positive_mild": "COVID-19",
    "positive_moderate": "COVID-19",
}

COUGH_FILENAME = "cough-heavy.wav"


def default_coswara_metadata_path() -> Path:
    """Return the default Coswara combined metadata path."""

    return Path(
        "/Users/wenting/Projects/Coswara-Data/combined_data.csv"
    )


def default_coswara_audio_root() -> Path:
    """Return the root directory containing extracted Coswara audio."""

    return Path(
        "/Users/wenting/Projects/Coswara-Data/Extracted_data"
    )


def find_participant_audio(
    audio_root: Path,
    participant_id: str,
    cough_filename: str = COUGH_FILENAME,
) -> Path | None:
    """Locate a participant's cough recording across Coswara date folders."""

    matches = list(
        audio_root.glob(
            f"*/{participant_id}/{cough_filename}"
        )
    )

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Multiple audio files found for participant "
            f"{participant_id}: {matches}"
        )

    return matches[0]


def prepare_coswara_metadata(
    metadata_path: Path,
    audio_root: Path,
    output_path: Path,
    cough_filename: str = COUGH_FILENAME,
) -> dict[str, int]:
    """
    Prepare Coswara metadata for the existing cough model pipeline.

    Included:
        healthy -> healthy

        positive_asymp -> COVID-19
        positive_mild -> COVID-19
        positive_moderate -> COVID-19

    Other Coswara statuses are excluded.

    The output contains:
        uuid
        status
        audio_path
        source_status
        dataset
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Coswara metadata not found: {metadata_path}"
        )

    if not audio_root.exists():
        raise FileNotFoundError(
            f"Coswara audio root not found: {audio_root}"
        )

    prepared_rows = []

    total = 0
    excluded_status = 0
    missing_audio = 0
    missing_id = 0

    source_status_counts = Counter()
    final_class_counts = Counter()

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        required_columns = {
            "id",
            "covid_status",
        }

        if not reader.fieldnames:
            raise ValueError(
                "Coswara metadata CSV has no header."
            )

        missing_columns = (
            required_columns - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Coswara metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            total += 1

            participant_id = row.get(
                "id",
                "",
            ).strip()

            source_status = row.get(
                "covid_status",
                "",
            ).strip()

            source_status_counts[source_status] += 1

            if not participant_id:
                missing_id += 1
                continue

            # Exclude statuses outside our clean binary task.
            if source_status not in STATUS_MAP:
                excluded_status += 1
                continue

            final_status = STATUS_MAP[source_status]

            audio_path = find_participant_audio(
                audio_root=audio_root,
                participant_id=participant_id,
                cough_filename=cough_filename,
            )

            if audio_path is None:
                missing_audio += 1
                continue

            prepared_rows.append(
                {
                    # Keep the same naming convention used
                    # by the existing pipeline.
                    "uuid": participant_id,
                    "status": final_status,
                    "audio_path": str(
                        audio_path.resolve()
                    ),
                    "source_status": source_status,
                    "dataset": "coswara",
                }
            )

            final_class_counts[final_status] += 1

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "uuid",
        "status",
        "audio_path",
        "source_status",
        "dataset",
    ]

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
        writer.writerows(prepared_rows)

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("COSWARA DATASET PREPARATION")
    print("=" * 60)

    print(f"\nMetadata:     {metadata_path}")
    print(f"Audio root:   {audio_root}")
    print(f"Cough file:   {cough_filename}")
    print(f"Output:       {output_path}")

    print("\nOriginal status counts:")

    for status, count in sorted(
        source_status_counts.items()
    ):
        print(
            f"  {status:<30} {count:>6,}"
        )

    print("\nFinal binary class counts:")

    for status in [
        "healthy",
        "COVID-19",
    ]:
        print(
            f"  {status:<15} "
            f"{final_class_counts[status]:>6,}"
        )

    print("\nFiltering:")

    print(
        f"  Total participants:       {total:,}"
    )

    print(
        f"  Excluded status:          "
        f"{excluded_status:,}"
    )

    print(
        f"  Missing participant ID:   "
        f"{missing_id:,}"
    )

    print(
        f"  Missing cough audio:      "
        f"{missing_audio:,}"
    )

    print(
        f"  Final usable samples:     "
        f"{len(prepared_rows):,}"
    )

    return {
        "total": total,
        "excluded_status": excluded_status,
        "missing_id": missing_id,
        "missing_audio": missing_audio,
        "usable": len(prepared_rows),
        "healthy": final_class_counts["healthy"],
        "COVID-19": final_class_counts["COVID-19"],
    }

def _summary(values: list[float]) -> dict[str, float]:
    """Return simple descriptive statistics."""

    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "max": 0.0,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def _print_comparison(
    title: str,
    healthy_values: list[float],
    covid_values: list[float],
    unit: str = "",
) -> None:
    """Print descriptive statistics by binary class."""

    healthy = _summary(healthy_values)
    covid = _summary(covid_values)

    print(f"\n=== {title} ===")

    print(
        f"{'Statistic':<12}"
        f"{'Healthy':>15}"
        f"{'COVID-19':>15}"
    )

    print("-" * 42)

    for key, label in [
        ("count", "Count"),
        ("mean", "Mean"),
        ("median", "Median"),
        ("std", "Std"),
        ("min", "Min"),
        ("q25", "25%"),
        ("q75", "75%"),
        ("max", "Max"),
    ]:
        if key == "count":
            print(
                f"{label:<12}"
                f"{healthy[key]:>15.0f}"
                f"{covid[key]:>15.0f}"
            )
        else:
            print(
                f"{label:<12}"
                f"{healthy[key]:>14.3f}{unit}"
                f"{covid[key]:>14.3f}{unit}"
            )


def analyze_coswara_audio(
    metadata_path: Path,
    top_db_values: tuple[int, ...] = (20, 30, 40, 50, 60),
) -> None:
    """
    Perform EDA on prepared Coswara cough recordings.

    Analyses:
        - class distribution
        - sample-rate distribution
        - channel distribution
        - recording duration
        - active duration after librosa trim
        - percentage removed by trimming
        - spectral energy below 8 kHz
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared Coswara metadata not found: {metadata_path}"
        )

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        reader = csv.DictReader(metadata_file)
        records = list(reader)

    if not records:
        raise ValueError(
            "Prepared Coswara metadata contains no records."
        )

    print("\n" + "=" * 60)
    print("COSWARA COUGH AUDIO EDA")
    print("=" * 60)

    print(f"\nMetadata: {metadata_path}")
    print(f"Samples:  {len(records):,}")
    print(
        "Trim thresholds tested: "
        + ", ".join(str(value) for value in top_db_values)
        + " dB"
    )

    class_counts = Counter()
    sample_rate_counts = Counter()
    channel_counts = Counter()

    durations = {
        "healthy": [],
        "COVID-19": [],
    }

    active_durations = {
        top_db: {
            "healthy": [],
            "COVID-19": [],
        }
        for top_db in top_db_values
    }

    removed_percentages = {
        top_db: {
            "healthy": [],
            "COVID-19": [],
        }
        for top_db in top_db_values
    }

    energy_below_8k = {
        "healthy": [],
        "COVID-19": [],
    }

    missing_audio = 0
    invalid_audio = 0

    total_records = len(records)
    trim_analysis_rows = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        status = record["status"].strip()
        audio_path = Path(
            record["audio_path"]
        )

        class_counts[status] += 1

        if not audio_path.exists():
            missing_audio += 1
            continue

        try:
            # ---------------------------------------------
            # Read file metadata without resampling
            # ---------------------------------------------
            info = sf.info(str(audio_path))

            original_sr = int(info.samplerate)
            channels = int(info.channels)
            duration = float(info.duration)

            sample_rate_counts[original_sr] += 1
            channel_counts[channels] += 1

            durations[status].append(duration)

            # ---------------------------------------------
            # Load original audio
            #
            # sr=None is important:
            # EDA must inspect the original recording
            # before deciding whether 16 kHz is suitable.
            # ---------------------------------------------
            waveform, sr = librosa.load(
                audio_path,
                sr=None,
                mono=True,
            )

            if waveform.size == 0:
                invalid_audio += 1
                continue

            record_trim_results = {
                "uuid": record["uuid"],
                "status": status,
                "audio_path": str(audio_path),
                "original_duration": duration,
            }

            # ---------------------------------------------
            # Trimming threshold comparison
            # ---------------------------------------------
            for top_db in top_db_values:
                trimmed, _ = librosa.effects.trim(
                    waveform,
                    top_db=top_db,
                )

                active_duration = len(trimmed) / sr
                active_durations[top_db][status].append(
                    active_duration
                )

                if duration > 0:
                    removed_percentage = (
                        1.0 - (active_duration / duration)
                    ) * 100.0
                else:
                    removed_percentage = 0.0

                removed_percentage = min(
                    100.0,
                    max(0.0, removed_percentage),
                )

                removed_percentages[top_db][status].append(
                    removed_percentage
                )

                record_trim_results[
                    f"active_duration_db{top_db}"
                ] = active_duration
                record_trim_results[
                    f"removed_percent_db{top_db}"
                ] = removed_percentage

            if 40 in top_db_values and 60 in top_db_values:
                record_trim_results["difference_db40_db60"] = (
                    record_trim_results["active_duration_db60"]
                    - record_trim_results["active_duration_db40"]
                )
            else:
                record_trim_results["difference_db40_db60"] = ""

            trim_analysis_rows.append(record_trim_results)

            # ---------------------------------------------
            # Frequency-energy analysis
            #
            # 16 kHz sampling has an 8 kHz Nyquist
            # frequency. Measure the proportion of
            # spectral energy already below 8 kHz.
            # ---------------------------------------------
            spectrum = np.fft.rfft(
                waveform.astype(
                    np.float64
                )
            )

            power = (
                np.abs(spectrum) ** 2
            )

            frequencies = np.fft.rfftfreq(
                len(waveform),
                d=1.0 / sr,
            )

            total_energy = float(
                np.sum(power)
            )

            if total_energy > 0:
                below_8k = float(
                    np.sum(
                        power[
                            frequencies
                            <= 8000.0
                        ]
                    )
                    / total_energy
                    * 100.0
                )

                energy_below_8k[
                    status
                ].append(
                    below_8k
                )

        except Exception as exc:
            invalid_audio += 1

            print(
                "\nWARNING: Could not analyse "
                f"{audio_path}: {exc}"
            )

        if (
            index % 100 == 0
            or index == total_records
        ):
            print(
                f"\rAnalysed "
                f"{index:,}/{total_records:,}",
                end="",
                flush=True,
            )

    print()

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print("\n=== Class Distribution ===")

    for status in [
        "healthy",
        "COVID-19",
    ]:
        count = class_counts[status]

        percentage = (
            count / total_records * 100
        )

        print(
            f"{status:<15}"
            f"{count:>6,}"
            f"  ({percentage:5.1f}%)"
        )

    # -----------------------------------------------------
    # Sample-rate distribution
    # -----------------------------------------------------

    print("\n=== Original Sample Rates ===")

    for sr, count in sorted(
        sample_rate_counts.items()
    ):
        percentage = (
            count
            / sum(sample_rate_counts.values())
            * 100
        )

        print(
            f"{sr:>6,} Hz:"
            f" {count:>6,}"
            f"  ({percentage:5.1f}%)"
        )

    # -----------------------------------------------------
    # Channel distribution
    # -----------------------------------------------------

    print("\n=== Audio Channels ===")

    for channels, count in sorted(
        channel_counts.items()
    ):
        label = (
            "mono"
            if channels == 1
            else (
                "stereo"
                if channels == 2
                else f"{channels} channels"
            )
        )

        print(
            f"{label:<15}"
            f"{count:>6,}"
        )

    # -----------------------------------------------------
    # Duration analysis
    # -----------------------------------------------------

    _print_comparison(
        title="Original Duration",
        healthy_values=durations[
            "healthy"
        ],
        covid_values=durations[
            "COVID-19"
        ],
        unit="s",
    )

    # -----------------------------------------------------
    # Fixed-duration coverage using ORIGINAL untrimmed audio
    # -----------------------------------------------------

    duration_thresholds = (4, 5, 6, 7, 8, 10)

    print("\n=== Fixed Duration Coverage (Untrimmed Audio) ===")
    print(
        f"{'Duration':<12}"
        f"{'Healthy':>22}"
        f"{'COVID-19':>22}"
    )
    print("-" * 56)

    healthy_durations = durations["healthy"]
    covid_durations = durations["COVID-19"]

    for threshold in duration_thresholds:
        healthy_count = sum(
            value <= threshold
            for value in healthy_durations
        )
        covid_count = sum(
            value <= threshold
            for value in covid_durations
        )

        healthy_pct = (
            healthy_count / len(healthy_durations) * 100
            if healthy_durations
            else 0.0
        )
        covid_pct = (
            covid_count / len(covid_durations) * 100
            if covid_durations
            else 0.0
        )

        print(
            f"<= {threshold:<7}s"
            f"{healthy_count:>8,} ({healthy_pct:5.1f}%)"
            f"{covid_count:>8,} ({covid_pct:5.1f}%)"
        )

    healthy_over_10 = sum(
        value > 10
        for value in healthy_durations
    )
    covid_over_10 = sum(
        value > 10
        for value in covid_durations
    )

    healthy_over_10_pct = (
        healthy_over_10 / len(healthy_durations) * 100
        if healthy_durations
        else 0.0
    )
    covid_over_10_pct = (
        covid_over_10 / len(covid_durations) * 100
        if covid_durations
        else 0.0
    )

    print(
        f"> 10      s"
        f"{healthy_over_10:>8,} ({healthy_over_10_pct:5.1f}%)"
        f"{covid_over_10:>8,} ({covid_over_10_pct:5.1f}%)"
    )

    # -----------------------------------------------------
    # Fixed-duration coverage after conservative db60 trim
    # -----------------------------------------------------

    if 60 in active_durations:
        print("\n=== Fixed Duration Coverage (After trim top_db=60) ===")
        print(
            f"{'Duration':<12}"
            f"{'Healthy':>22}"
            f"{'COVID-19':>22}"
        )
        print("-" * 56)

        healthy_db60 = active_durations[60]["healthy"]
        covid_db60 = active_durations[60]["COVID-19"]

        for threshold in duration_thresholds:
            healthy_count = sum(
                value <= threshold
                for value in healthy_db60
            )
            covid_count = sum(
                value <= threshold
                for value in covid_db60
            )

            healthy_pct = (
                healthy_count / len(healthy_db60) * 100
                if healthy_db60
                else 0.0
            )
            covid_pct = (
                covid_count / len(covid_db60) * 100
                if covid_db60
                else 0.0
            )

            print(
                f"<= {threshold:<7}s"
                f"{healthy_count:>8,} ({healthy_pct:5.1f}%)"
                f"{covid_count:>8,} ({covid_pct:5.1f}%)"
            )

        healthy_over_10 = sum(
            value > 10 for value in healthy_db60
        )
        covid_over_10 = sum(
            value > 10 for value in covid_db60
        )

        healthy_over_10_pct = (
            healthy_over_10 / len(healthy_db60) * 100
            if healthy_db60
            else 0.0
        )
        covid_over_10_pct = (
            covid_over_10 / len(covid_db60) * 100
            if covid_db60
            else 0.0
        )

        print(
            f"> 10      s"
            f"{healthy_over_10:>8,} ({healthy_over_10_pct:5.1f}%)"
            f"{covid_over_10:>8,} ({covid_over_10_pct:5.1f}%)"
        )

    # -----------------------------------------------------
    # Trimming threshold comparison
    # -----------------------------------------------------

    print("\n=== Trimming Threshold Comparison ===")

    print(
        f"{'top_db':<10}"
        f"{'Healthy active':>16}"
        f"{'Healthy removed':>18}"
        f"{'COVID active':>16}"
        f"{'COVID removed':>18}"
    )
    print("-" * 78)

    for top_db in top_db_values:
        healthy_active = _summary(
            active_durations[top_db]["healthy"]
        )
        healthy_removed = _summary(
            removed_percentages[top_db]["healthy"]
        )
        covid_active = _summary(
            active_durations[top_db]["COVID-19"]
        )
        covid_removed = _summary(
            removed_percentages[top_db]["COVID-19"]
        )

        print(
            f"{top_db:<10}"
            f"{healthy_active['median']:>15.3f}s"
            f"{healthy_removed['median']:>17.2f}%"
            f"{covid_active['median']:>15.3f}s"
            f"{covid_removed['median']:>17.2f}%"
        )

    # -----------------------------------------------------
    # 16 kHz investigation
    # -----------------------------------------------------

    _print_comparison(
        title="Spectral Energy <= 8 kHz",
        healthy_values=energy_below_8k[
            "healthy"
        ],
        covid_values=energy_below_8k[
            "COVID-19"
        ],
        unit="%",
    )

    print("\n=== Data Quality ===")
    print(
        f"Missing audio:  {missing_audio:,}"
    )
    print(
        f"Invalid audio:  {invalid_audio:,}"
    )

    # -----------------------------------------------------
    # Export per-recording trimming analysis
    # -----------------------------------------------------
    trim_output_path = metadata_path.parent / (
        metadata_path.stem + "_trim_analysis.csv"
    )

    fieldnames = [
        "uuid",
        "status",
        "audio_path",
        "original_duration",
    ]

    for top_db in top_db_values:
        fieldnames.extend(
            [
                f"active_duration_db{top_db}",
                f"removed_percent_db{top_db}",
            ]
        )

    fieldnames.append("difference_db40_db60")

    with trim_output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(trim_analysis_rows)

    print("\nPer-recording trimming analysis:")
    print(trim_output_path)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


def export_listening_samples(
    trim_analysis_path: Path,
    output_dir: Path,
    samples_per_class: int = 2,
) -> None:
    """Export original, db40 and db60 WAVs for useful A/B listening tests."""

    if not trim_analysis_path.exists():
        raise FileNotFoundError(
            f"Trim analysis CSV not found: {trim_analysis_path}. Run --analyze first."
        )

    with trim_analysis_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    selected = []
    for status in ("healthy", "COVID-19"):
        class_rows = [row for row in rows if row.get("status", "").strip() == status]
        class_rows.sort(
            key=lambda row: float(row.get("difference_db40_db60", 0.0) or 0.0),
            reverse=True,
        )
        selected.extend(class_rows[:samples_per_class])

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("COSWARA LISTENING SAMPLE EXPORT")
    print("=" * 60)

    for row in selected:
        participant_id = row["uuid"].strip()
        status = row["status"].strip()
        source_path = Path(row["audio_path"])

        if not source_path.exists():
            print(f"WARNING: Missing source audio: {source_path}")
            continue

        waveform, sr = librosa.load(source_path, sr=None, mono=True)
        db40, _ = librosa.effects.trim(waveform, top_db=40)
        db60, _ = librosa.effects.trim(waveform, top_db=60)

        sample_dir = output_dir / f"{status}_{participant_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sf.write(sample_dir / "original.wav", waveform, sr)
        sf.write(sample_dir / "db40.wav", db40, sr)
        sf.write(sample_dir / "db60.wav", db60, sr)

        print(
            f"{status:<10} {participant_id}  "
            f"db40-db60 difference={float(row['difference_db40_db60']):.3f}s"
        )
        print(f"  {sample_dir}")

    print(f"\nExported listening samples to: {output_dir}")


def validate_coswara_audio(
    metadata_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Validate prepared Coswara audio and write valid records to a new CSV."""

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared Coswara metadata not found: {metadata_path}"
        )

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        reader = csv.DictReader(metadata_file)

        if not reader.fieldnames:
            raise ValueError("Prepared Coswara metadata CSV has no header.")

        fieldnames = list(reader.fieldnames)
        records = list(reader)

    valid_rows = []
    invalid_rows = []

    reason_counts = Counter()

    for index, record in enumerate(records, start=1):
        audio_path = Path(record.get("audio_path", "").strip())
        reason = None

        if not audio_path.exists():
            reason = "missing_file"
        else:
            try:
                info = sf.info(str(audio_path))

                if info.frames <= 0 or info.duration <= 0:
                    reason = "zero_duration"
                else:
                    waveform, _ = librosa.load(
                        audio_path,
                        sr=None,
                        mono=True,
                    )

                    if waveform.size == 0:
                        reason = "empty_waveform"
                    elif not np.isfinite(waveform).all():
                        reason = "non_finite"
                    else:
                        energy = float(
                            np.sum(
                                waveform.astype(np.float64) ** 2
                            )
                        )

                        if energy <= 0.0:
                            reason = "zero_energy"

            except Exception:
                reason = "decode_error"

        if reason is None:
            valid_rows.append(record)
        else:
            invalid_record = dict(record)
            invalid_record["invalid_reason"] = reason
            invalid_rows.append(invalid_record)
            reason_counts[reason] += 1

        if index % 100 == 0 or index == len(records):
            print(
                f"\rValidated {index:,}/{len(records):,}",
                end="",
                flush=True,
            )

    print()

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
        writer.writerows(valid_rows)

    invalid_output_path = output_path.with_name(
        output_path.stem.replace("_valid_metadata", "_invalid_audio")
        + ".csv"
    )

    invalid_fieldnames = fieldnames + ["invalid_reason"]

    with invalid_output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=invalid_fieldnames,
        )
        writer.writeheader()
        writer.writerows(invalid_rows)

    print("\n" + "=" * 60)
    print("COSWARA AUDIO VALIDATION")
    print("=" * 60)
    print(f"\nInput metadata:   {metadata_path}")
    print(f"Valid metadata:   {output_path}")
    print(f"Invalid report:   {invalid_output_path}")
    print(f"\nInput records:    {len(records):,}")
    print(f"Valid records:    {len(valid_rows):,}")
    print(f"Invalid records:  {len(invalid_rows):,}")

    if reason_counts:
        print("\nInvalid reasons:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason:<20} {count:>6,}")

    return {
        "input_records": len(records),
        "valid_records": len(valid_rows),
        "invalid_records": len(invalid_rows),
    }

def main() -> None:
    """Prepare or analyse the Coswara dataset."""

    parser = argparse.ArgumentParser(
        description="Coswara dataset utilities."
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare Coswara metadata.",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run EDA on prepared Coswara cough audio.",
    )

    parser.add_argument(
        "--export-listening-samples",
        action="store_true",
        help="Export original/db40/db60 WAVs for selected listening comparisons.",
    )

    parser.add_argument(
        "--validate-audio",
        action="store_true",
        help="Validate prepared Coswara audio and write valid-only metadata.",
    )

    parser.add_argument(
        "--cough-type",
        choices=["heavy", "shallow"],
        default="heavy",
        help=(
            "Coswara cough recording to use: "
            "heavy or shallow (default: heavy)."
        ),
    )

    args = parser.parse_args()

    # Convert CLI option to the actual Coswara filename.
    cough_filename = f"cough-{args.cough_type}.wav"

    # Keep heavy and shallow experiments separate.
    output_path = Path(
        f"data/coswara/"
        f"coswara_{args.cough_type}_metadata.csv"
    )

    valid_output_path = Path(
        f"data/coswara/"
        f"coswara_{args.cough_type}_valid_metadata.csv"
    )

    if args.prepare:
        prepare_coswara_metadata(
            metadata_path=default_coswara_metadata_path(),
            audio_root=default_coswara_audio_root(),
            output_path=output_path,
            cough_filename=cough_filename,
        )

    if args.validate_audio:
        validate_coswara_audio(
            metadata_path=output_path,
            output_path=valid_output_path,
        )

    if args.analyze:
        analyze_coswara_audio(
            metadata_path=output_path,
        )

    if args.export_listening_samples:
        trim_analysis_path = output_path.parent / (
            output_path.stem + "_trim_analysis.csv"
        )
        export_listening_samples(
            trim_analysis_path=trim_analysis_path,
            output_dir=Path(
                f"data/coswara/listening_samples/{args.cough_type}"
            ),
        )

    if not args.prepare and not args.analyze and not args.validate_audio and not args.export_listening_samples:
        parser.print_help()

if __name__ == "__main__":
    main()