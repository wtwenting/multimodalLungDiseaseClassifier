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
import itertools
import zlib
from pathlib import Path

import numpy as np
import librosa

from .features import (
    create_fixed_log_mel,
    detect_active_intervals,
    load_waveform,
    segment_waveform,
    trim_silence,
    create_coswara_log_mel,
    preprocess_coswara_waveform,
)
import tensorflow as tf

# =========================================================
# Configuration
# =========================================================

SAMPLE_RATE = 16000
TARGET_DURATION = 11.0
SEGMENT_DURATION = 4.0

N_MELS = 64
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 256


def get_model_input_shape(use_segmentation: bool = False) -> tuple[int, int, int]:
    """Return the fixed CNN input shape for an experiment pipeline."""

    target_duration = SEGMENT_DURATION if use_segmentation else TARGET_DURATION
    frame_count = int(
        target_duration * SAMPLE_RATE // HOP_LENGTH
    ) + 1
    return (N_MELS, frame_count, 1)


def get_waveform_input_shape(use_segmentation: bool = False) -> tuple[int, int]:
    """Return the fixed raw-waveform input shape for a 1D CNN."""

    target_duration = SEGMENT_DURATION if use_segmentation else TARGET_DURATION
    return (int(target_duration * SAMPLE_RATE), 1)


def get_mfcc_input_shape(use_segmentation: bool = False) -> tuple[int, int]:
    """Return (time_steps, MFCC coefficients) for the LSTM input."""

    target_duration = SEGMENT_DURATION if use_segmentation else TARGET_DURATION
    target_samples = int(target_duration * SAMPLE_RATE)
    frame_count = 1 + (target_samples // HOP_LENGTH)
    return (frame_count, N_MFCC)


def _fix_waveform_length(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad or truncate a waveform to an exact number of samples."""

    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) >= target_samples:
        return waveform[:target_samples]
    return np.pad(waveform, (0, target_samples - len(waveform)))


def apply_audio_preprocessing(
    waveform: np.ndarray,
    sample_rate: int,
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    use_augmentation: bool = False,
    augmentation_mode: str = "none",
    seed: int = 42,
) -> np.ndarray:
    """Apply preprocessing plus optional training-only waveform augmentation."""

    processed = waveform

    if use_trim:
        processed = trim_silence(
            processed,
            top_db=40.0,
        )

    if use_active_extraction:
        # Keep top_db fixed at 40 across preprocessing experiments. This
        # controls silence detection consistently rather than assuming this
        # threshold is optimal.
        intervals = detect_active_intervals(
            processed,
            sample_rate,
            top_db=40.0,
        )
        if intervals:
            processed = np.concatenate(
                [processed[start:end] for start, end in intervals]
            )

    if use_segmentation:
        segments = segment_waveform(
            processed,
            sample_rate=sample_rate,
            segment_duration=4.0,
            overlap=0.5,
        )
        # Keep one deterministic segment per recording so the split remains
        # record-level and every experiment has the same sample count.
        processed = segments[0]

    if use_augmentation:
        rng = np.random.default_rng(seed)
        mode = augmentation_mode if augmentation_mode != "none" else "gain_noise"

        # FT-Aug1 base: mild gain + Gaussian noise.
        gain = rng.uniform(0.9, 1.1)
        noise_scale = max(float(np.max(np.abs(processed))) * 0.005, 1e-5)
        processed = (processed * gain) + rng.normal(
            0.0, noise_scale, size=processed.shape
        ).astype(np.float32)

        # FT-Aug2: add a conservative random pitch shift (+/- 1 semitone).
        if mode == "gain_noise_pitch":
            n_steps = float(rng.uniform(-1.0, 1.0))
            processed = librosa.effects.pitch_shift(
                processed.astype(np.float32), sr=sample_rate, n_steps=n_steps
            ).astype(np.float32)

        # FT-Aug3: add a time shift up to +/-10% of the waveform length.
        elif mode == "gain_noise_time_shift":
            max_shift = max(1, int(0.10 * len(processed)))
            shift = int(rng.integers(-max_shift, max_shift + 1))
            shifted = np.zeros_like(processed)
            if shift > 0:
                shifted[shift:] = processed[:-shift]
            elif shift < 0:
                shifted[:shift] = processed[-shift:]
            else:
                shifted = processed
            processed = shifted

        elif mode not in {"gain_noise", "specaugment"}:
            raise ValueError(f"Unknown augmentation mode: {mode}")

    return processed.astype(np.float32, copy=False)


# =========================================================
# Class labels
# =========================================================

CLASS_NAMES = [
    "healthy",
    # "symptomatic",
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

            if (
                row["split"].strip() == split
                and row["status"].strip() in CLASS_TO_INDEX
            ):
                records.append(row)

    return records


# =========================================================
# Load one Log-Mel sample
# =========================================================

def load_logmel_sample(
    record: dict[str, str],
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    use_augmentation: bool = False,
    augmentation_mode: str = "none",
    seed: int = 42,
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
        COVID-19     -> 1
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

    waveform = apply_audio_preprocessing(
        waveform,
        sample_rate,
        use_trim=use_trim,
        use_active_extraction=use_active_extraction,
        use_segmentation=use_segmentation,
        use_augmentation=use_augmentation,
        augmentation_mode=augmentation_mode,
        seed=seed,
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
        target_duration=(
            SEGMENT_DURATION
            if use_segmentation
            else TARGET_DURATION
        ),
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # FT-Aug4: SpecAugment on Log-Mel only (training dataset passes this mode).
    if use_augmentation and augmentation_mode == "specaugment":
        rng = np.random.default_rng(seed + 17)
        log_mel = log_mel.copy()
        fill = float(np.min(log_mel))
        freq_width = int(rng.integers(1, max(2, min(9, log_mel.shape[0] // 8 + 1))))
        freq_start = int(rng.integers(0, max(1, log_mel.shape[0] - freq_width + 1)))
        log_mel[freq_start:freq_start + freq_width, :] = fill
        time_width = int(rng.integers(1, max(2, min(26, log_mel.shape[1] // 8 + 1))))
        time_start = int(rng.integers(0, max(1, log_mel.shape[1] - time_width + 1)))
        log_mel[:, time_start:time_start + time_width] = fill

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


def load_waveform_sample(
    record: dict[str, str],
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    use_augmentation: bool = False,
    augmentation_mode: str = "none",
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Convert one recording into a fixed raw waveform for a 1D CNN."""

    status = record["status"].strip()
    if status not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown class label: {status}")

    audio_path = Path(record["audio_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sample_rate = load_waveform(audio_path)
    waveform = apply_audio_preprocessing(
        waveform,
        sample_rate,
        use_trim=use_trim,
        use_active_extraction=use_active_extraction,
        use_segmentation=use_segmentation,
        use_augmentation=use_augmentation,
        augmentation_mode=augmentation_mode,
        seed=seed,
    )

    target_duration = SEGMENT_DURATION if use_segmentation else TARGET_DURATION
    target_samples = int(target_duration * sample_rate)
    waveform = _fix_waveform_length(waveform, target_samples)
    x = np.expand_dims(waveform, axis=-1).astype(np.float32)
    y = CLASS_TO_INDEX[status]
    return x, y


def load_mfcc_sample(
    record: dict[str, str],
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    use_augmentation: bool = False,
    augmentation_mode: str = "none",
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Convert one recording into a fixed MFCC sequence for an LSTM."""

    status = record["status"].strip()
    if status not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown class label: {status}")

    audio_path = Path(record["audio_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sample_rate = load_waveform(audio_path)
    waveform = apply_audio_preprocessing(
        waveform, sample_rate,
        use_trim=use_trim,
        use_active_extraction=use_active_extraction,
        use_segmentation=use_segmentation,
        use_augmentation=use_augmentation,
        augmentation_mode=augmentation_mode,
        seed=seed,
    )

    target_duration = SEGMENT_DURATION if use_segmentation else TARGET_DURATION
    target_samples = int(target_duration * sample_rate)
    waveform = _fix_waveform_length(waveform, target_samples)

    # librosa returns (n_mfcc, time); LSTM expects (time, features).
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    ).T
    x = mfcc.astype(np.float32)
    y = CLASS_TO_INDEX[status]
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
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    use_augmentation: bool = False,
    augmentation_mode: str = "none",
    representation: str = "logmel",
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

    presentation_counter = itertools.count()

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

            sample_loader = (
                (load_waveform_sample if representation == "waveform" else load_mfcc_sample if representation == "mfcc" else load_logmel_sample)
            )
            x, numeric_label = sample_loader(
                record,
                use_trim=use_trim,
                use_active_extraction=use_active_extraction,
                use_segmentation=use_segmentation,
                use_augmentation=use_augmentation,
                seed=(
                    seed
                    + zlib.crc32(path_string.encode("utf-8"))
                    + next(presentation_counter)
                ),
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
            get_waveform_input_shape(use_segmentation)
            if representation == "waveform"
            else get_mfcc_input_shape(use_segmentation)
            if representation == "mfcc"
            else get_model_input_shape(use_segmentation)
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
        healthy      -> 1/2
        COVID-19     -> 1/2

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

        x.set_shape(get_model_input_shape(use_segmentation))

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
    sampling_weights: tuple[float, float] = (
        1 / 2,
        1 / 2,
    ),
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    use_augmentation: bool = False,
    augmentation_mode: str = "none",
    representation: str = "logmel",
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

    presentation_counter = itertools.count()

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
                ((load_waveform_sample if representation == "waveform" else load_mfcc_sample if representation == "mfcc" else load_logmel_sample))(
                    record,
                    use_trim=use_trim,
                    use_active_extraction=use_active_extraction,
                    use_segmentation=use_segmentation,
                    use_augmentation=use_augmentation,
                    augmentation_mode=augmentation_mode,
                    seed=(
                        seed
                        + zlib.crc32(path_string.encode("utf-8"))
                        + next(presentation_counter)
                    ),
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
            get_waveform_input_shape(use_segmentation)
            if representation == "waveform"
            else get_mfcc_input_shape(use_segmentation)
            if representation == "mfcc"
            else get_model_input_shape(use_segmentation)
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


# =========================================================
# Coswara baseline model-data path
# =========================================================

COSWARA_SAMPLE_RATE = 16000
COSWARA_TARGET_DURATION = 10.0
COSWARA_TRIM_TOP_DB = 60.0


def get_coswara_model_input_shape() -> tuple[int, int, int]:
    """Return the fixed Log-Mel CNN shape for the Coswara baseline."""

    target_samples = int(
        COSWARA_TARGET_DURATION * COSWARA_SAMPLE_RATE
    )
    frame_count = 1 + (
        target_samples // HOP_LENGTH
    )
    return (
        N_MELS,
        frame_count,
        1,
    )


def load_coswara_logmel_sample(
    record: dict[str, str],
) -> tuple[np.ndarray, int]:
    """Convert one Coswara split record into model-ready Log-Mel data."""

    status = record["status"].strip()

    if status not in CLASS_TO_INDEX:
        raise ValueError(
            f"Unknown class label: {status}"
        )

    audio_path = Path(
        record["audio_path"]
    )

    log_mel, _ = create_coswara_log_mel(
        audio_path=audio_path,
        sample_rate=COSWARA_SAMPLE_RATE,
        target_duration=COSWARA_TARGET_DURATION,
        top_db=COSWARA_TRIM_TOP_DB,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    x = np.expand_dims(
        log_mel,
        axis=-1,
    ).astype(np.float32)

    expected_shape = get_coswara_model_input_shape()

    if x.shape != expected_shape:
        raise ValueError(
            "Unexpected Coswara Log-Mel shape: "
            f"{x.shape}; expected {expected_shape}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"Coswara Log-Mel contains NaN/Inf: {audio_path}"
        )

    y = CLASS_TO_INDEX[status]

    return x, y


def test_coswara_preprocessing(
    metadata_path: Path,
    samples_per_class: int = 2,
) -> None:
    """Smoke-test the Coswara baseline preprocessing on training records."""

    records = load_split_records(
        metadata_path=metadata_path,
        split="train",
    )

    selected = []
    counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    for record in records:
        status = record["status"].strip()

        if counts[status] >= samples_per_class:
            continue

        selected.append(record)
        counts[status] += 1

        if all(
            count >= samples_per_class
            for count in counts.values()
        ):
            break

    if not selected:
        raise ValueError(
            "No Coswara training records were found."
        )

    print("\n" + "=" * 60)
    print("COSWARA PREPROCESSING SMOKE TEST")
    print("=" * 60)
    print(f"\nMetadata: {metadata_path}")
    print(f"Train records: {len(records):,}")
    print(f"Sample rate: {COSWARA_SAMPLE_RATE:,} Hz")
    print(f"Trim: top_db={COSWARA_TRIM_TOP_DB:g}")
    print(f"Fixed duration: {COSWARA_TARGET_DURATION:g} sec")
    print(
        f"Target waveform samples: "
        f"{int(COSWARA_SAMPLE_RATE * COSWARA_TARGET_DURATION):,}"
    )
    print(
        f"Expected Log-Mel shape: "
        f"{get_coswara_model_input_shape()}"
    )

    shapes = set()

    for record in selected:
        audio_path = Path(
            record["audio_path"]
        )

        fixed_waveform, details = preprocess_coswara_waveform(
            audio_path=audio_path,
            sample_rate=COSWARA_SAMPLE_RATE,
            target_duration=COSWARA_TARGET_DURATION,
            top_db=COSWARA_TRIM_TOP_DB,
        )

        x, y = load_coswara_logmel_sample(
            record
        )

        shapes.add(x.shape)

        print("\n" + "-" * 60)
        print(f"UUID: {record['uuid']}")
        print(f"Status: {record['status']}")
        print(f"Numeric label: {y}")
        print(f"Audio: {audio_path}")
        print(
            f"Original sample rate: "
            f"{details['original_sample_rate']:,} Hz"
        )
        print(
            f"Original duration: "
            f"{details['original_duration']:.3f} sec"
        )
        print(
            f"After db60 trim: "
            f"{details['trimmed_duration']:.3f} sec"
        )
        print(
            f"After resample: "
            f"{details['samples_after_resample']:,} samples"
        )
        print(
            f"Duration action: "
            f"{details['duration_action']}"
        )
        print(
            f"Final waveform: "
            f"{fixed_waveform.shape} "
            f"({len(fixed_waveform):,} samples)"
        )
        print(
            f"Log-Mel shape: {x.shape}"
        )
        print(
            f"Finite waveform: "
            f"{bool(np.isfinite(fixed_waveform).all())}"
        )
        print(
            f"Finite Log-Mel: "
            f"{bool(np.isfinite(x).all())}"
        )

    if len(shapes) != 1:
        raise ValueError(
            f"Inconsistent Coswara Log-Mel shapes: {shapes}"
        )

    print("\nCOSWARA PREPROCESSING TEST PASSED.")



def create_coswara_tf_dataset(
    metadata_path: Path,
    split: str,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a streaming TensorFlow dataset for the Coswara Log-Mel baseline."""

    records = load_split_records(
        metadata_path=metadata_path,
        split=split,
    )

    if not records:
        raise ValueError(
            f"No Coswara records found for split: {split}"
        )

    audio_paths = [
        record["audio_path"]
        for record in records
    ]

    labels = [
        CLASS_TO_INDEX[
            record["status"].strip()
        ]
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
        """Bridge TensorFlow to the Coswara Librosa preprocessing path."""

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
                load_coswara_logmel_sample(
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
            get_coswara_model_input_shape()
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


def test_coswara_tf_dataset(
    metadata_path: Path,
    batch_size: int = 8,
) -> None:
    """Smoke-test one Coswara TensorFlow batch without training."""

    dataset = create_coswara_tf_dataset(
        metadata_path=metadata_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    expected_sample_shape = (
        get_coswara_model_input_shape()
    )

    print("\n" + "=" * 60)
    print("COSWARA TENSORFLOW DATASET TEST")
    print("=" * 60)
    print(f"\nMetadata: {metadata_path}")
    print(f"Batch size: {batch_size}")
    print(
        f"Expected sample shape: "
        f"{expected_sample_shape}"
    )

    found_batch = False

    for x_batch, y_batch in dataset.take(1):
        found_batch = True

        print(
            f"\nX shape: {x_batch.shape}"
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

        expected_batch_shape = (
            x_batch.shape[0],
            *expected_sample_shape,
        )

        if tuple(x_batch.shape) != expected_batch_shape:
            raise ValueError(
                "Unexpected Coswara TensorFlow batch shape: "
                f"{tuple(x_batch.shape)}; "
                f"expected {expected_batch_shape}"
            )

        if not bool(
            tf.reduce_all(
                tf.math.is_finite(x_batch)
            ).numpy()
        ):
            raise ValueError(
                "Coswara TensorFlow batch contains NaN/Inf."
            )

        label_values = set(
            int(value)
            for value in y_batch.numpy()
        )

        valid_label_values = set(
            CLASS_TO_INDEX.values()
        )

        if not label_values.issubset(
            valid_label_values
        ):
            raise ValueError(
                "Coswara TensorFlow batch contains "
                f"unexpected labels: {label_values}"
            )

    if not found_batch:
        raise ValueError(
            "Coswara TensorFlow dataset produced no batch."
        )

    print(
        "\nCOSWARA TENSORFLOW DATASET TEST PASSED."
    )
