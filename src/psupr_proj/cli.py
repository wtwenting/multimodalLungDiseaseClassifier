"""Command-line entry point for PsuprProj."""

import argparse
from pathlib import Path

from .data import default_metadata_path, filter_dataset, summarize_metadata
from .cleaning import analyze_metadata, clean_metadata
from .download import download_dataset


def main() -> None:
    """Print a summary of the CoughVID metadata dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=default_metadata_path(),
        help="Path to a metadata CSV file.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Copy records meeting the cough score threshold to the output folder.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum cough_detected score for --filter (default: 0.8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/coughvid_20211012_processed"),
        help="Destination folder for --filter.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the COUGHVID dataset from Zenodo.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data"),
        help="Destination folder for --download (default: data).",
    )

    parser.add_argument(
    "--analyze",
    action="store_true",
    help="Analyse filtered COUGHVID metadata before cleaning.", 
    )

    parser.add_argument(
    "--clean",
    action="store_true",
    help="Clean filtered COUGHVID metadata for supervised training.",
    )

    args = parser.parse_args()
    if args.download:
        result = download_dataset(args.download_dir)
        print(f"Archive: {result['archive']}")
        print(f"Extracted to: {result['extracted_to']}")
        return

    if args.filter:
        result = filter_dataset(
            args.data,
            args.data.parent,
            args.output_dir,
            args.threshold,
        )
        print(f"Filtered records: {result['records']}")
        print(f"Copied files: {result['files']}")
        print(f"Records without files: {result['missing_files']}")
        print(f"Output: {args.output_dir}")
        return

    if args.analyze:
        filtered_metadata = (
            args.output_dir / "metadata_filtered.csv"
        )

        if not filtered_metadata.exists():
            parser.error(
                "Filtered metadata does not exist. "
                "Run with --filter first."
            )

        analyze_metadata(filtered_metadata)
        return

    if args.clean:
        filtered_metadata = (
            args.output_dir / "metadata_filtered.csv"
        )

        clean_metadata_path = (
            args.output_dir / "metadata_clean.csv"
        )

        if not filtered_metadata.exists():
            parser.error(
                "Filtered metadata does not exist. "
                "Run with --filter first."
            )

        result = clean_metadata(
            filtered_metadata,
            clean_metadata_path,
        )

        print("\n" + "=" * 60)
        print("COUGHVID METADATA CLEANING")
        print("=" * 60)

        print(
            f"Input records:        "
            f"{result['input_records']:,}"
        )

        print(
            f"Missing UUID:         "
            f"{result['missing_uuid']:,}"
        )

        print(
            f"Missing status:       "
            f"{result['missing_status']:,}"
        )

        print(
            f"Invalid status:       "
            f"{result['invalid_status']:,}"
        )

        print(
            f"Invalid cough score:  "
            f"{result['invalid_cough_score']:,}"
        )

        print("-" * 40)

        print(
            f"Clean records:        "
            f"{result['clean_records']:,}"
        )

        print(
            f"\nOutput: {clean_metadata_path}"
        )

        return

    summary = summarize_metadata(args.data)
    print(f"Dataset: {args.data}")
    print(f"Records: {summary['records']}")
    print(f"Scored records: {summary['scored_records']}")
    average = summary.get("average_cough_score")
    if average is not None:
        print(f"Average cough score: {average:.4f}")


if __name__ == "__main__":
    main()
