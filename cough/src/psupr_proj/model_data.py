"""Prepare model-ready COUGHVID data.

This module connects the processed COUGHVID dataset to the machine
learning models.

Responsibilities:
- Read train / validation / test records from metadata_split.csv
- Convert disease labels to numeric class indices
- Load standardized WAV files
- Generate fixed-size Log-Mel spectrograms
- Prepare model-ready NumPy tensors
"""

import csv
from pathlib import Path

import numpy as np

from .features import (
    create_fixed_log_mel,
    load_waveform,
)
import tensorflow as tf

# =========================================================
# Configuration
# =========================================================

SAMPLE_RATE = 16000
TARGET_DURATION = 11.0

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# =========================================================
# Class labels
# =========================================================

CLASS_NAMES = [
    "healthy",
    "symptomatic",
    "COVID-19",
]

CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: label
    for label, index in CLASS_TO_INDEX.items()
}


# =========================================================
# Metadata loading
# =========================================================

def load_split_records(
    metadata_path: Path,
    split: str,
) -> list[dict[str, str]]:
    """Load records belonging to one dataset split.

    Parameters
    ----------
    metadata_path:
        Path to metadata_split.csv.

    split:
        Dataset split to load:
        "train", "validation", or "test".

    Returns
    -------
    list
        Metadata rows belonging to the requested split.
    """

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of {sorted(valid_splits)}."
        )

    records: list[dict[str, str]] = []

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        if not reader.fieldnames:
            raise ValueError(
                "Metadata CSV has no header."
            )

        required_columns = {
            "uuid",
            "status",
            "audio_path",
            "split",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            if row["split"].strip() == split:
                records.append(row)

    return records


# =========================================================
# Load one Log-Mel sample
# =========================================================

def load_logmel_sample(
    record: dict[str, str],
) -> tuple[np.ndarray, int]:
    """Convert one COUGHVID recording into model-ready data.

    Returns
    -------
    x:
        Log-Mel spectrogram with shape approximately
        (64, 688, 1).

    y:
        Numeric class label.

        healthy      -> 0
        symptomatic  -> 1
        COVID-19     -> 2
    """

    status = record["status"].strip()

    if status not in CLASS_TO_INDEX:
        raise ValueError(
            f"Unknown class label: {status}"
        )

    audio_path = Path(
        record["audio_path"]
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # -----------------------------------------------------
    # Load standardized 16 kHz WAV
    # -----------------------------------------------------

    waveform, sample_rate = load_waveform(
        audio_path
    )

    # -----------------------------------------------------
    # Whole-recording baseline:
    #
    # NO cough trimming.
    #
    # Audio is padded/truncated to 11 seconds inside
    # create_fixed_log_mel().
    # -----------------------------------------------------

    log_mel = create_fixed_log_mel(
        waveform=waveform,
        sample_rate=sample_rate,
        target_duration=TARGET_DURATION,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # -----------------------------------------------------
    # Add CNN channel dimension.
    #
    # Before:
    #
    #     (64, 688)
    #
    # After:
    #
    #     (64, 688, 1)
    #
    # -----------------------------------------------------

    log_mel = np.expand_dims(
        log_mel,
        axis=-1,
    )

    # TensorFlow normally works with float32.
    x = log_mel.astype(
        np.float32
    )

    # Convert textual disease status to numeric label.
    y = CLASS_TO_INDEX[
        status
    ]

    return x, y


# =========================================================
# Technical loader test
# =========================================================

def test_dataset_loader(
    metadata_path: Path,
    batch_size: int = 8,
) -> None:
    """Test model-data preparation on a small training batch."""

    records = load_split_records(
        metadata_path=metadata_path,
        split="train",
    )

    print("\n" + "=" * 60)
    print("COUGHVID MODEL DATA TEST")
    print("=" * 60)

    print(
        f"Training records: "
        f"{len(records):,}"
    )

    print("\n=== Class Mapping ===")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:15} "
            f"-> {CLASS_TO_INDEX[class_name]}"
        )

    if not records:
        raise ValueError(
            "No training records were found."
        )

    actual_batch_size = min(
        batch_size,
        len(records),
    )

    samples: list[np.ndarray] = []
    labels: list[int] = []

    print("\n=== Individual Samples ===")

    for record in records[:actual_batch_size]:

        x, y = load_logmel_sample(
            record
        )

        samples.append(x)
        labels.append(y)

        print(
            f"\nUUID:   "
            f"{record['uuid']}"
        )

        print(
            f"Status: "
            f"{record['status']}"
        )

        print(
            f"Label:  "
            f"{y}"
        )

        print(
            f"Shape:  "
            f"{x.shape}"
        )

        print(
            f"dtype:  "
            f"{x.dtype}"
        )

    # -----------------------------------------------------
    # Combine samples into one model batch.
    # -----------------------------------------------------

    x_batch = np.stack(
        samples,
        axis=0,
    )

    y_batch = np.asarray(
        labels,
        dtype=np.int64,
    )

    print("\n=== Batch ===")

    print(
        f"X shape: "
        f"{x_batch.shape}"
    )

    print(
        f"y shape: "
        f"{y_batch.shape}"
    )

    print(
        f"X dtype: "
        f"{x_batch.dtype}"
    )

    print(
        f"y dtype: "
        f"{y_batch.dtype}"
    )

    print(
        f"Labels:  "
        f"{y_batch}"
    )

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    expected_shape = (
        actual_batch_size,
        N_MELS,
        x_batch.shape[2],
        1,
    )

    if x_batch.shape != expected_shape:
        raise ValueError(
            "Unexpected batch shape: "
            f"{x_batch.shape}"
        )

    if len(x_batch) != len(y_batch):
        raise ValueError(
            "Number of samples and labels do not match."
        )

    if not np.isfinite(x_batch).all():
        raise ValueError(
            "Input contains NaN or infinite values."
        )

    print("\nDataset loader test PASSED.")

def create_tf_dataset(
    metadata_path: Path,
    split: str,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a streaming TensorFlow dataset."""

    records = load_split_records(
        metadata_path=metadata_path,
        split=split,
    )

    audio_paths = [
        record["audio_path"]
        for record in records
    ]

    labels = [
        CLASS_TO_INDEX[record["status"]]
        for record in records
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            audio_paths,
            labels,
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def process_sample(
        audio_path: tf.Tensor,
        label: tf.Tensor,
    ):
        """Bridge TensorFlow to our existing Librosa pipeline."""

        def python_load(path, y):
            path_string = path.numpy().decode("utf-8")

            record = {
                "audio_path": path_string,
                "status": INDEX_TO_CLASS[int(y.numpy())],
            }

            x, numeric_label = load_logmel_sample(
                record
            )

            return (
                x,
                np.int64(numeric_label),
            )

        x, y = tf.py_function(
            func=python_load,
            inp=[audio_path, label],
            Tout=[
                tf.float32,
                tf.int64,
            ],
        )

        # tf.py_function loses static shape information,
        # so restore it explicitly.
        x.set_shape(
            (64, 688, 1)
        )

        y.set_shape(())

        return x, y

    dataset = dataset.map(
        process_sample,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

def test_tf_dataset(
    metadata_path: Path,
    batch_size: int = 8,
) -> None:
    """Test the streaming TensorFlow dataset."""

    dataset = create_tf_dataset(
        metadata_path=metadata_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("COUGHVID TENSORFLOW DATASET TEST")
    print("=" * 60)

    for x_batch, y_batch in dataset.take(1):

        print(
            f"X shape: {x_batch.shape}"
        )

        print(
            f"y shape: {y_batch.shape}"
        )

        print(
            f"X dtype: {x_batch.dtype}"
        )

        print(
            f"y dtype: {y_batch.dtype}"
        )

        print(
            f"Labels: {y_batch.numpy()}"
        )

    print(
        "\nTensorFlow dataset test PASSED."
    )

'''
def create_balanced_tf_dataset(
    metadata_path: Path,
    batch_size: int = 32,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a balanced training dataset using oversampling.

    Each class is sampled with equal probability:
        healthy      -> 1/3
        symptomatic  -> 1/3
        COVID-19     -> 1/3

    The original dataset is NOT modified.
    """

    records = load_split_records(
        metadata_path=metadata_path,
        split="train",
    )

    # Separate training records by class.
    class_records = {
        class_name: []
        for class_name in CLASS_NAMES
    }

    for record in records:
        class_records[
            record["status"]
        ].append(record)

    print("\n=== Balanced Sampling Pools ===")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:15} "
            f"{len(class_records[class_name]):,}"
        )

    class_datasets = []

    for class_name in CLASS_NAMES:

        records_for_class = (
            class_records[class_name]
        )

        audio_paths = [
            record["audio_path"]
            for record in records_for_class
        ]

        label = CLASS_TO_INDEX[
            class_name
        ]

        labels = [
            label
            for _ in records_for_class
        ]

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    audio_paths,
                    labels,
                )
            )
        )

        # Shuffle within this class.
        dataset = dataset.shuffle(
            buffer_size=len(
                records_for_class
            ),
            seed=seed,
            reshuffle_each_iteration=True,
        )

        # Repeat indefinitely so minority classes
        # can be oversampled.
        dataset = dataset.repeat()

        class_datasets.append(
            dataset
        )

    # -----------------------------------------------------
    # Sample equally from the three class datasets.
    # -----------------------------------------------------

    dataset = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[
            1 / 3,
            1 / 3,
            1 / 3,
        ],
        seed=seed,
    )

    # -----------------------------------------------------
    # Convert WAV -> Log-Mel
    # -----------------------------------------------------

    def process_sample(
        audio_path: tf.Tensor,
        label: tf.Tensor,
    ):
        """Convert one WAV into model-ready Log-Mel data."""

        def python_load(path, y):

            path_string = (
                path.numpy()
                .decode("utf-8")
            )

            numeric_label = int(
                y.numpy()
            )

            record = {
                "audio_path": path_string,
                "status": INDEX_TO_CLASS[
                    numeric_label
                ],
            }

            x, numeric_label = (
                load_logmel_sample(
                    record
                )
            )

            return (
                x,
                np.int64(
                    numeric_label
                ),
            )

        x, y = tf.py_function(
            func=python_load,
            inp=[
                audio_path,
                label,
            ],
            Tout=[
                tf.float32,
                tf.int64,
            ],
        )

        x.set_shape(
            (64, 688, 1)
        )

        y.set_shape(())

        return x, y

    dataset = dataset.map(
        process_sample,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset
'''

def create_balanced_tf_dataset(
    metadata_path: Path,
    batch_size: int = 32,
    seed: int = 42,
    sampling_weights: tuple[float, float, float] = (
        1 / 3,
        1 / 3,
        1 / 3,
    ),
) -> tf.data.Dataset:
    """Create a training dataset using configurable class sampling."""

    records = load_split_records(
        metadata_path=metadata_path,
        split="train",
    )

    class_records = {
        class_name: []
        for class_name in CLASS_NAMES
    }

    for record in records:
        class_records[
            record["status"]
        ].append(record)

    print("\n=== Sampling Pools ===")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:15} "
            f"{len(class_records[class_name]):,}"
        )

    print("\n=== Sampling Weights ===")

    for class_name, weight in zip(
        CLASS_NAMES,
        sampling_weights,
    ):
        print(
            f"{class_name:15} "
            f"{weight * 100:.1f}%"
        )

    class_datasets = []

    for class_name in CLASS_NAMES:

        records_for_class = (
            class_records[class_name]
        )

        audio_paths = [
            record["audio_path"]
            for record in records_for_class
        ]

        label = CLASS_TO_INDEX[
            class_name
        ]

        labels = [
            label
            for _ in records_for_class
        ]

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    audio_paths,
                    labels,
                )
            )
        )

        dataset = dataset.shuffle(
            buffer_size=len(
                records_for_class
            ),
            seed=seed,
            reshuffle_each_iteration=True,
        )

        dataset = dataset.repeat()

        class_datasets.append(
            dataset
        )

    dataset = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=list(sampling_weights),
        seed=seed,
    )

    def process_sample(
        audio_path: tf.Tensor,
        label: tf.Tensor,
    ):

        def python_load(path, y):

            path_string = (
                path.numpy()
                .decode("utf-8")
            )

            numeric_label = int(
                y.numpy()
            )

            record = {
                "audio_path": path_string,
                "status": INDEX_TO_CLASS[
                    numeric_label
                ],
            }

            x, numeric_label = (
                load_logmel_sample(
                    record
                )
            )

            return (
                x,
                np.int64(
                    numeric_label
                ),
            )

        x, y = tf.py_function(
            func=python_load,
            inp=[
                audio_path,
                label,
            ],
            Tout=[
                tf.float32,
                tf.int64,
            ],
        )

        x.set_shape(
            (64, 688, 1)
        )

        y.set_shape(())

        return x, y

    dataset = dataset.map(
        process_sample,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def test_balanced_tf_dataset(
    metadata_path: Path,
    sample_batches: int = 30,
    batch_size: int = 32,
) -> None:
    """Check the class distribution produced by balanced sampling."""

    dataset = create_balanced_tf_dataset(
        metadata_path=metadata_path,
        batch_size=batch_size,
        seed=42,
    )

    counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    total = 0

    for _, y_batch in dataset.take(
        sample_batches
    ):

        labels = y_batch.numpy()

        for label in labels:
            counts[int(label)] += 1
            total += 1

    print("\n" + "=" * 60)
    print("BALANCED SAMPLING TEST")
    print("=" * 60)

    for class_name in CLASS_NAMES:

        index = CLASS_TO_INDEX[
            class_name
        ]

        count = counts[index]

        percentage = (
            count / total * 100
        )

        print(
            f"{class_name:15} "
            f"{count:5,} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"\nSamples checked: {total:,}"
    )