"""Command-line entry point for PsuprProj."""

import argparse
from pathlib import Path

from .data import (
    analyze_cough_thresholds,
    default_metadata_path,
    filter_dataset,
    print_cough_threshold_analysis,
    summarize_metadata,
)
from .cleaning import analyze_metadata, clean_metadata
from .download import download_dataset
from .audio import (
    analyze_audio_content,
    analyze_audio_files,
    standardize_audio_dataset,
)
from .split import create_stratified_split
from .features import (
    analyze_acoustic_features_by_class,
    analyze_duration_by_class,
    analyze_active_audio_by_class,
    explore_training_examples,
    test_fixed_log_mel,
    test_segmentation,
)
from .models import test_baseline_2d_cnn
from .model_data import (
    test_balanced_tf_dataset,
    test_dataset_loader,
    test_tf_dataset,
)
from .evaluate import evaluate_baseline_2d_cnn

from .train import (
    train_baseline_2d_cnn,
    train_sampled_2d_cnn,
)

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
        "--analyze-thresholds",
        action="store_true",
        help="Count healthy/COVID-19 records at several thresholds without copying files.",
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

    parser.add_argument(
    "--analyze-audio",
    action="store_true",
    help="Analyse audio files associated with cleaned metadata.",
    )

    parser.add_argument(
    "--probe-audio",
    action="store_true",
    help="Inspect audio properties using FFprobe.",
    )

    parser.add_argument(
    "--standardize-audio",
    action="store_true",
    help="Convert cleaned COUGHVID audio to standardized WAV.",
    )

    parser.add_argument(
    "--split",
    action="store_true",
    help="Create stratified train/validation/test splits.",
    )

    parser.add_argument(
    "--explore-features",
    action="store_true",
    help="Visualise waveform and Log-Mel examples.",
    )

    parser.add_argument(
    "--test-segmentation",
    action="store_true",
    help="Test silence trimming and cough segmentation.",
    )

    parser.add_argument(
    "--test-logmel",
    action="store_true",
    help="Test fixed-duration Log-Mel generation.",
    )

    parser.add_argument(
        "--analyze-duration",
        action="store_true",
        help="Analyse training audio duration by class.",
    )

    parser.add_argument(
        "--analyze-acoustic-features",
        action="store_true",
        help="Summarize training acoustic features by class.",
    )
    
    parser.add_argument(
        "--test-dataset",
        action="store_true",
        help="Test model-ready Log-Mel data loading.",
    )

    parser.add_argument(
    "--test-model",
    action="store_true",
    help="Build and inspect the baseline 2D CNN.",
    )

    parser.add_argument(
    "--test-tf-dataset",
    action="store_true",
    help="Test streaming TensorFlow data loading.",
    )

    parser.add_argument(
    "--train-baseline",
    action="store_true",
    help="Train the baseline Log-Mel 2D CNN.",
    )

    parser.add_argument(
    "--evaluate-baseline",
    action="store_true",
    help="Evaluate the baseline 2D CNN on validation data.",
    )

    parser.add_argument(
    "--evaluate-model",
    type=str,
    metavar="MODEL_NAME",
    help="Evaluate a saved model on validation data.",
    )

    parser.add_argument(
    "--train-weighted",
    action="store_true",
    help="Train the class-weighted baseline 2D CNN.",
    )

    parser.add_argument(
    "--test-balanced-dataset",
    action="store_true",
    help="Test balanced class sampling.",
    )

    parser.add_argument(
    "--train-balanced",
    action="store_true",
    help="Train V3 using balanced class sampling.",
    )

    parser.add_argument(
    "--train-v4",
    action="store_true",
    help="Train V4 using balanced healthy/COVID-19 class sampling.",
    )

    parser.add_argument(
        "--train-p0",
        action="store_true",
        help="Train P0 baseline without waveform preprocessing.",
    )
    parser.add_argument(
        "--train-p1",
        action="store_true",
        help="Train P1 with silence trimming.",
    )
    parser.add_argument(
        "--train-p2",
        action="store_true",
        help="Train P2 with active-audio extraction.",
    )
    parser.add_argument(
        "--train-p3",
        action="store_true",
        help="Train P3 with active extraction and segmentation.",
    )
    parser.add_argument(
        "--train-p4",
        action="store_true",
        help="Train P4 with active extraction, segmentation, and augmentation.",
    )

    parser.add_argument(
        "--analyze-active-audio",
        action="store_true",
        help="Analyse non-silent training audio by class.",
    )
    parser.add_argument(
        "--active-top-db",
        type=float,
        default=40.0,
        help="Silence threshold for --analyze-active-audio (default: 40).",
    )

    args = parser.parse_args()
    if args.analyze_thresholds:
        results = analyze_cough_thresholds(args.data)
        print_cough_threshold_analysis(results)
        return

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
            threshold=args.threshold,
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

    if args.analyze_audio:
        clean_metadata_path = (
            args.output_dir / "metadata_clean.csv"
        )

        if not clean_metadata_path.exists():
            parser.error(
                "metadata_clean.csv does not exist. "
                "Run --clean first."
            )

        result = analyze_audio_files(
            metadata_path=clean_metadata_path,
            audio_dir=args.output_dir,
        )

        print("\n" + "=" * 60)
        print("COUGHVID AUDIO FILE ANALYSIS")
        print("=" * 60)

        print(
            f"Metadata records:       "
            f"{result['total_records']:,}"
        )
        print(
            f"Audio found:            "
            f"{result['found_records']:,}"
        )
        print(
            f"Audio missing:          "
            f"{result['missing_records']:,}"
        )
        print(
            f"Zero-byte audio files:  "
            f"{result['zero_byte_files']:,}"
        )

        print("\n=== Audio Formats ===")

        for extension, count in sorted(
            result["extension_counts"].items()
        ):
            print(f"{extension:10} {count:8,}")

        if result["missing_uuids"]:
            print("\n=== Example Missing UUIDs ===")

            for uuid in result["missing_uuids"][:10]:
                print(f"- {uuid}")

        return

    if args.probe_audio:

        clean_metadata_path = (
            args.output_dir / "metadata_clean.csv"
        )

        if not clean_metadata_path.exists():
            parser.error(
                "metadata_clean.csv does not exist. "
                "Run --clean first."
            )

        result = analyze_audio_content(
            metadata_path=clean_metadata_path,
            audio_dir=args.output_dir,
            limit=None,
        )

        print("\n" + "=" * 60)
        print("COUGHVID AUDIO CONTENT ANALYSIS")
        print("=" * 60)

        print(f"Files checked:       {result['files_checked']:,}")
        print(f"Successfully probed: {result['successful']:,}")
        print(f"Failed:              {result['failed']:,}")

        print("\n=== Sample Rates ===")
        for rate, count in sorted(result["sample_rates"].items()):
            print(f"{rate:8} Hz : {count:,}")

        print("\n=== Channels ===")
        for channel_count, count in sorted(
            result["channels"].items()
        ):
            print(f"{channel_count:8} : {count:,}")

        print("\n=== Codecs ===")
        for codec, count in sorted(result["codecs"].items()):
            print(f"{codec:12} : {count:,}")

        duration = result["duration"]

        if duration:
            print("\n=== Duration ===")
            print(f"Minimum: {duration['minimum']:.2f} sec")
            print(f"Maximum: {duration['maximum']:.2f} sec")
            print(f"Average: {duration['average']:.2f} sec")

        if result["failures"]:
            print("\n=== Example Failures ===")

            for filename, error in result["failures"][:10]:
                print(f"{filename}: {error}")

        return

    if args.standardize_audio:

        clean_metadata_path = (
            args.output_dir / "metadata_clean.csv"
        )

        standardized_dir = (
            args.output_dir / "audio_standardized"
        )

        if not clean_metadata_path.exists():
            parser.error(
                "metadata_clean.csv does not exist. "
                "Run --clean first."
            )

        result = standardize_audio_dataset(
            metadata_path=clean_metadata_path,
            audio_dir=args.output_dir,
            output_dir=standardized_dir,
            sample_rate=16000,
            limit=None,
        )

        print("\n" + "=" * 60)
        print("COUGHVID AUDIO STANDARDISATION")
        print("=" * 60)

        print(f"Input records:       {result['total']:,}")
        print(f"Converted:           {result['converted']:,}")
        print(f"Already existing:    {result['skipped']:,}")
        print(f"Failed:              {result['failed']:,}")

        print("\n=== Output Format ===")
        print(f"Sample rate:         {result['sample_rate']:,} Hz")
        print("Channels:            1 (mono)")
        print("Codec:               PCM signed 16-bit")
        print("Container:           WAV")

        print(f"\nOutput: {standardized_dir}")

        if result["failures"]:
            print("\n=== Example Failures ===")

            for filename, error in result["failures"][:10]:
                print(f"{filename}: {error}")

        return

    if args.split:

        clean_metadata_path = (
            args.output_dir / "metadata_clean.csv"
        )

        standardized_audio_dir = (
            args.output_dir / "audio_standardized"
        )

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not clean_metadata_path.exists():
            parser.error(
                "metadata_clean.csv does not exist. "
                "Run --clean first."
            )

        if not standardized_audio_dir.exists():
            parser.error(
                "Standardized audio directory does not exist. "
                "Run --standardize-audio first."
            )

        result = create_stratified_split(
            metadata_path=clean_metadata_path,
            audio_dir=standardized_audio_dir,
            output_path=split_metadata_path,
            train_size=0.70,
            validation_size=0.15,
            test_size=0.15,
            random_state=42,
        )

        print("\n" + "=" * 60)
        print("COUGHVID STRATIFIED DATASET SPLIT")
        print("=" * 60)

        print(f"Total records:       {result['total']:,}")
        print(f"Missing audio:       {result['missing_audio']:,}")

        print("\n=== Dataset Split ===")

        print(f"Train:               {result['train']:,}")
        print(f"Validation:          {result['validation']:,}")
        print(f"Test:                {result['test']:,}")

        print("\n=== Train Classes ===")
        for label, count in sorted(
            result["train_classes"].items()
        ):
            print(f"{label:20} {count:6,}")

        print("\n=== Validation Classes ===")
        for label, count in sorted(
            result["validation_classes"].items()
        ):
            print(f"{label:20} {count:6,}")

        print("\n=== Test Classes ===")
        for label, count in sorted(
            result["test_classes"].items()
        ):
            print(f"{label:20} {count:6,}")

        print(
            f"\nRandom state:        "
            f"{result['random_state']}"
        )

        print(
            f"\nOutput: {split_metadata_path}"
        )

        return

    if args.explore_features:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        explore_training_examples(
            split_metadata_path
        )

        return

    if args.test_segmentation:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        test_segmentation(
            metadata_path=split_metadata_path,
            top_db=30.0,
        )

        return

    if args.test_logmel:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        test_fixed_log_mel(
            metadata_path=split_metadata_path
        )

        return

    if args.analyze_duration:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        analyze_duration_by_class(
            metadata_path=split_metadata_path
        )

        return

    if args.analyze_acoustic_features:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        analyze_acoustic_features_by_class(
            metadata_path=split_metadata_path
        )

        return

    if args.test_dataset:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        test_dataset_loader(
            metadata_path=split_metadata_path,
            batch_size=8,
        )

        return

    if args.test_model:

        test_baseline_2d_cnn()

        return

    if args.test_tf_dataset:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        test_tf_dataset(
            metadata_path=split_metadata_path,
            batch_size=8,
        )

        return

    if args.train_baseline:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        train_baseline_2d_cnn(
            metadata_path=split_metadata_path,
            output_dir=args.output_dir,
            model_name="baseline_v1b_10epochs",
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
        )

        return
    
    if args.evaluate_baseline:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        model_path = (
            args.output_dir
            / "models"
            / "baseline_2d_cnn.keras"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist."
            )

        if not model_path.exists():
            parser.error(
                "Baseline model does not exist. "
                "Run --train-baseline first."
            )

        evaluate_baseline_2d_cnn(
            metadata_path=split_metadata_path,
            model_path=model_path,
            batch_size=32,
        )

        return

    if args.evaluate_model:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        model_path = (
            args.output_dir
            / "models"
            / f"{args.evaluate_model}.keras"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist."
            )

        if not model_path.exists():
            parser.error(
                f"Model does not exist: {model_path}"
            )

        evaluate_baseline_2d_cnn(
            metadata_path=split_metadata_path,
            model_path=model_path,
            batch_size=32,
            use_trim=args.evaluate_model.startswith("p1_"),
            use_active_extraction=args.evaluate_model.startswith(
                ("p2_", "p3_", "p4_")
            ),
            use_segmentation=args.evaluate_model.startswith(
                ("p3_", "p4_")
            ),
        )

        return

    if args.train_weighted:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist."
            )

        train_baseline_2d_cnn(
            metadata_path=split_metadata_path,
            output_dir=args.output_dir,
            model_name="class_weighted_v2",
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
            use_class_weights=True,
        )

        return
    
    if args.test_balanced_dataset:

        split_metadata_path = (
            args.output_dir
            / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist."
            )

        test_balanced_tf_dataset(
            metadata_path=split_metadata_path,
            sample_batches=30,
            batch_size=32,
        )

        return

    if args.train_balanced:

        split_metadata_path = (
            args.output_dir
            / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist."
            )

        train_sampled_2d_cnn(
            metadata_path=split_metadata_path,
            output_dir=args.output_dir,
            model_name="balanced_sampling_v3",
            sampling_weights=(
                1 / 2,
                1 / 2,
            ),
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
        )

        return

    experiment_flags = {
        "p0": args.train_p0,
        "p1": args.train_p1,
        "p2": args.train_p2,
        "p3": args.train_p3,
        "p4": args.train_p4,
    }

    selected_experiments = [
        name for name, selected in experiment_flags.items() if selected
    ]

    if args.train_v4 and selected_experiments:
        parser.error("Select either --train-v4 or a --train-p[0-4] preset.")

    if args.train_v4:
        selected_experiments = ["legacy_v4"]

    if len(selected_experiments) > 1:
        parser.error("Select only one training experiment preset.")

    if selected_experiments:

        experiment = selected_experiments[0]
        model_name = (
            "moderate_sampling_v4"
            if experiment == "legacy_v4"
            else f"{experiment}_balanced_sampling"
        )
        preprocessing = {
            "p0": {
                "use_trim": False,
                "use_active_extraction": False,
                "use_segmentation": False,
                "use_augmentation": False,
            },
            "p1": {
                "use_trim": True,
                "use_active_extraction": False,
                "use_segmentation": False,
                "use_augmentation": False,
            },
            "p2": {
                "use_trim": False,
                "use_active_extraction": True,
                "use_segmentation": False,
                "use_augmentation": False,
            },
            "p3": {
                "use_trim": False,
                "use_active_extraction": True,
                "use_segmentation": True,
                "use_augmentation": False,
            },
            "p4": {
                "use_trim": False,
                "use_active_extraction": True,
                "use_segmentation": True,
                "use_augmentation": True,
            },
        }

        if experiment == "legacy_v4":
            preprocessing = preprocessing["p0"]
        else:
            preprocessing = preprocessing[experiment]

        split_metadata_path = (
            args.output_dir
            / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist."
            )

        train_sampled_2d_cnn(
            metadata_path=split_metadata_path,
            output_dir=args.output_dir,
            model_name=model_name,
            sampling_weights=(
                0.50,
                0.50,
            ),
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            **preprocessing,
        )

        return

    
    if args.analyze_active_audio:

        split_metadata_path = (
            args.output_dir / "metadata_split.csv"
        )

        if not split_metadata_path.exists():
            parser.error(
                "metadata_split.csv does not exist. "
                "Run --split first."
            )

        analyze_active_audio_by_class(
            metadata_path=split_metadata_path,
            top_db=args.active_top_db,
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
