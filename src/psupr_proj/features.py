"""Audio feature extraction and visualisation utilities."""

import csv
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_waveform(
    audio_path: Path,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """Load a standardized WAV file."""

    waveform, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
    )

    return waveform, sr


def create_log_mel(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Convert a waveform into a log-Mel spectrogram."""

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    return log_mel

def get_example_training_samples(
    metadata_path: Path,
) -> dict[str, dict[str, str]]:
    """Get one training example from each class."""

    wanted_classes = {
        "healthy",
        "symptomatic",
        "COVID-19",
    }

    examples = {}

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        for row in reader:

            if row["split"] != "train":
                continue

            status = row["status"]

            if (
                status in wanted_classes
                and status not in examples
            ):
                examples[status] = row

            if len(examples) == len(wanted_classes):
                break

    return examples

def plot_audio_example(
    audio_path: Path,
    label: str,
) -> None:
    """Plot waveform and Log-Mel representation."""

    waveform, sr = load_waveform(audio_path)

    log_mel = create_log_mel(
        waveform,
        sr,
    )

    duration = len(waveform) / sr

    print(
        f"{label}: "
        f"{duration:.2f}s, "
        f"{len(waveform):,} samples, "
        f"Log-Mel shape={log_mel.shape}"
    )

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(
        waveform,
        sr=sr,
    )

    plt.title(f"{label} - Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        log_mel,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        f"{label} - Log-Mel Spectrogram"
    )

    plt.tight_layout()
    plt.show()

def explore_training_examples(
    metadata_path: Path,
) -> None:
    """Visualise one training sample from each class."""

    examples = get_example_training_samples(
        metadata_path
    )

    print("\n" + "=" * 60)
    print("COUGHVID FEATURE EXPLORATION")
    print("=" * 60)

    for label in [
        "healthy",
        "symptomatic",
        "COVID-19",
    ]:

        row = examples.get(label)

        if row is None:
            print(f"No training example found: {label}")
            continue

        audio_path = Path(row["audio_path"])

        print(f"\nLabel: {label}")
        print(f"UUID:  {row['uuid']}")
        print(f"Audio: {audio_path}")

        plot_audio_example(
            audio_path,
            label,
        )

def trim_silence(
    waveform: np.ndarray,
    top_db: float = 30.0,
) -> np.ndarray:
    """Remove leading and trailing low-energy audio."""

    trimmed, _ = librosa.effects.trim(
        waveform,
        top_db=top_db,
    )

    return trimmed


def segment_waveform(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    segment_duration: float = 4.0,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """Split waveform into fixed-size overlapping segments."""

    segment_samples = int(
        sample_rate * segment_duration
    )

    hop_samples = int(
        segment_samples * (1 - overlap)
    )

    # Short recording:
    # pad with zeros until exactly 4 seconds.
    if len(waveform) <= segment_samples:

        padded = np.zeros(
            segment_samples,
            dtype=waveform.dtype,
        )

        padded[:len(waveform)] = waveform

        return [padded]

    segments = []

    start = 0

    while start < len(waveform):

        end = start + segment_samples
        segment = waveform[start:end]

        # Final segment shorter than 4 seconds:
        # pad rather than discard it.
        if len(segment) < segment_samples:

            padded = np.zeros(
                segment_samples,
                dtype=waveform.dtype,
            )

            padded[:len(segment)] = segment
            segment = padded

        segments.append(segment)

        # Stop once this window reached the end.
        if end >= len(waveform):
            break

        start += hop_samples

    return segments

def test_segmentation(
    metadata_path: Path,
    top_db: float = 30.0,
) -> None:
    """Test cough trimming and segmentation on example samples."""

    examples = get_example_training_samples(
        metadata_path
    )

    print("\n" + "=" * 60)
    print("COUGHVID COUGH SEGMENTATION TEST")
    print("=" * 60)

    for label in [
        "healthy",
        "symptomatic",
        "COVID-19",
    ]:

        row = examples.get(label)

        if row is None:
            continue

        audio_path = Path(row["audio_path"])

        waveform, sr = load_waveform(
            audio_path
        )

        original_duration = (
            len(waveform) / sr
        )

        trimmed = trim_silence(
            waveform,
            top_db=top_db,
        )

        plot_trimming_comparison(
            waveform=waveform,
            trimmed=trimmed,
            sample_rate=sr,
            label=label,
        )

        trimmed_duration = (
            len(trimmed) / sr
        )

        segments = segment_waveform(
            trimmed,
            sample_rate=sr,
            segment_duration=4.0,
            overlap=0.5,
        )

        print(f"\nLabel: {label}")
        print(f"UUID: {row['uuid']}")

        print(
            f"Original duration: "
            f"{original_duration:.2f} sec"
        )

        print("\nTrim threshold comparison:")
        compare_trim_thresholds(
            waveform,
            sr,
        )

        print("\nActive interval analysis:")
        compare_active_intervals(
            waveform,
            sr,
        )

        print(
            f"After trimming:    "
            f"{trimmed_duration:.2f} sec"
        )

        print(
            f"Removed:           "
            f"{original_duration - trimmed_duration:.2f} sec"
        )

        print(
            f"Segments:          "
            f"{len(segments)}"
        )

        for i, segment in enumerate(
            segments,
            start=1,
        ):

            print(
                f"  Segment {i}: "
                f"{len(segment):,} samples "
                f"({len(segment) / sr:.2f} sec)"
            )

def plot_trimming_comparison(
    waveform: np.ndarray,
    trimmed: np.ndarray,
    sample_rate: int,
    label: str,
) -> None:
    """Compare original and silence-trimmed waveforms."""

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(
        waveform,
        sr=sample_rate,
    )

    plt.title(f"{label} - Original Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(
        trimmed,
        sr=sample_rate,
    )

    plt.title(f"{label} - Trimmed Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def detect_active_intervals(
    waveform: np.ndarray,
    sample_rate: int,
    top_db: float = 40.0,
    ) -> list[tuple[int, int]]:
    """Detect multiple active audio regions."""

    intervals = librosa.effects.split(
        waveform,
        top_db=top_db,
    )

    return [
        (int(start), int(end))
        for start, end in intervals
    ]

def compare_trim_thresholds(
    waveform: np.ndarray,
    sample_rate: int,
) -> None:
    """Compare silence-trimming thresholds."""

    original_duration = len(waveform) / sample_rate

    for top_db in [20, 30, 40, 50, 60]:

        trimmed = trim_silence(
            waveform,
            top_db=top_db,
        )

        trimmed_duration = len(trimmed) / sample_rate

        removed_percent = (
            (original_duration - trimmed_duration)
            / original_duration
            * 100
        )

        print(
            f"  top_db={top_db:2d}: "
            f"{trimmed_duration:5.2f}s retained, "
            f"{removed_percent:5.1f}% removed"
        )

def compare_active_intervals(
    waveform: np.ndarray,
    sample_rate: int,
) -> None:
    """Compare active regions at several thresholds."""

    original_duration = len(waveform) / sample_rate

    print(
        f"Original duration: "
        f"{original_duration:.2f}s"
    )

    for top_db in [30, 40, 50, 60]:

        intervals = detect_active_intervals(
            waveform,
            sample_rate,
            top_db=top_db,
        )

        total_active_samples = sum(
            end - start
            for start, end in intervals
        )

        active_duration = (
            total_active_samples / sample_rate
        )

        print(
            f"\n  top_db={top_db}"
        )

        print(
            f"  Active intervals: "
            f"{len(intervals)}"
        )

        print(
            f"  Total active audio: "
            f"{active_duration:.2f}s"
        )

        for i, (start, end) in enumerate(
            intervals,
            start=1,
        ):

            start_sec = start / sample_rate
            end_sec = end / sample_rate

            print(
                f"    {i}: "
                f"{start_sec:.2f}s "
                f"→ {end_sec:.2f}s "
                f"({end_sec - start_sec:.2f}s)"
            )

def pad_or_truncate_waveform(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    target_duration: float = 11.0,
) -> np.ndarray:
    """Pad or truncate waveform to a fixed duration."""

    target_samples = int(
        sample_rate * target_duration
    )

    # Recording is too long.
    if len(waveform) > target_samples:
        return waveform[:target_samples]

    # Recording already has correct length.
    if len(waveform) == target_samples:
        return waveform

    # Recording is shorter.
    padded = np.zeros(
        target_samples,
        dtype=waveform.dtype,
    )

    padded[:len(waveform)] = waveform

    return padded

def create_fixed_log_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    target_duration: float = 11.0,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Create a fixed-size Log-Mel spectrogram."""

    fixed_waveform = pad_or_truncate_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        target_duration=target_duration,
    )

    log_mel = create_log_mel(
        waveform=fixed_waveform,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    return log_mel

def test_fixed_log_mel(
    metadata_path: Path,
) -> None:
    """Test fixed-duration Log-Mel generation."""

    examples = get_example_training_samples(
        metadata_path
    )

    print("\n" + "=" * 60)
    print("COUGHVID FIXED LOG-MEL TEST")
    print("=" * 60)

    shapes = []

    for label in [
        "healthy",
        "symptomatic",
        "COVID-19",
    ]:

        row = examples.get(label)

        if row is None:
            continue

        audio_path = Path(row["audio_path"])

        waveform, sr = load_waveform(
            audio_path
        )

        original_duration = len(waveform) / sr

        fixed_waveform = pad_or_truncate_waveform(
            waveform=waveform,
            sample_rate=sr,
            target_duration=11.0,
        )

        log_mel = create_fixed_log_mel(
            waveform=waveform,
            sample_rate=sr,
            target_duration=11.0,
            n_mels=64,
            n_fft=1024,
            hop_length=256,
        )

        shapes.append(log_mel.shape)

        print(f"\nLabel: {label}")
        print(f"UUID: {row['uuid']}")

        print(
            f"Original duration: "
            f"{original_duration:.2f} sec"
        )

        print(
            f"Original samples:  "
            f"{len(waveform):,}"
        )

        print(
            f"Fixed samples:     "
            f"{len(fixed_waveform):,}"
        )

        print(
            f"Log-Mel shape:     "
            f"{log_mel.shape}"
        )

    print("\n=== Shape Check ===")

    unique_shapes = set(shapes)

    print(
        f"Unique Log-Mel shapes: "
        f"{len(unique_shapes)}"
    )

    for shape in sorted(unique_shapes):
        print(shape)

def analyze_duration_by_class(
    metadata_path: Path,
) -> None:
    """Analyse training audio duration by class."""

    durations = defaultdict(list)

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:

        reader = csv.DictReader(metadata_file)

        for row in reader:

            # Only use training data when making
            # preprocessing decisions.
            if row["split"] != "train":
                continue

            label = row["status"]
            audio_path = Path(row["audio_path"])

            waveform, sr = load_waveform(
                audio_path
            )

            duration = len(waveform) / sr

            durations[label].append(duration)

    print("\n" + "=" * 60)
    print("COUGHVID TRAINING DURATION ANALYSIS")
    print("=" * 60)

    for label in [
        "healthy",
        "symptomatic",
        "COVID-19",
    ]:

        values = np.array(
            durations[label]
        )

        # --------------------------------------
        # Calculate recordings under 4 seconds
        # --------------------------------------

        under_4 = np.sum(values < 4.0)

        under_4_percentage = (
            under_4 / len(values) * 100
        )

        # --------------------------------------
        # Calculate recordings over 11 seconds
        # --------------------------------------

        over_11 = np.sum(values > 11.0)

        over_11_percentage = (
            over_11 / len(values) * 100
        )

        # --------------------------------------
        # Print statistics
        # --------------------------------------
        print(f"\n=== {label} ===")

        print(
            f"Recordings: "
            f"{len(values):,}"
        )

        print(
            f"Minimum:    "
            f"{np.min(values):.2f} sec"
        )

        print(
            f"25th pct:   "
            f"{np.percentile(values, 25):.2f} sec"
        )

        print(
            f"Median:     "
            f"{np.median(values):.2f} sec"
        )

        print(
            f"Mean:       "
            f"{np.mean(values):.2f} sec"
        )

        print(
            f"75th pct:   "
            f"{np.percentile(values, 75):.2f} sec"
        )

        print(
            f"Maximum:    "
            f"{np.max(values):.2f} sec"
        )

        print(
            f"Under 4 sec: "
            f"{under_4:,} "
            f"({under_4_percentage:.2f}%)"
        )

        print(
            f"Over 11 sec: "
            f"{over_11:,} "
            f"({over_11_percentage:.2f}%)"
        )