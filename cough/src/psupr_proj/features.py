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
        # "symptomatic",
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
        # "symptomatic",
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
        # "symptomatic",
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
        # "symptomatic",
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
        # "symptomatic",
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


def _extract_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int,
) -> dict[str, np.ndarray]:
    """Extract frame-level acoustic features from one recording."""

    return {
        "RMS energy": librosa.feature.rms(
            y=waveform,
            frame_length=1024,
            hop_length=256,
        )[0],
        "Zero-crossing rate": librosa.feature.zero_crossing_rate(
            waveform,
            hop_length=256,
        )[0],
        "Spectral centroid": librosa.feature.spectral_centroid(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
        )[0],
        "Spectral bandwidth": librosa.feature.spectral_bandwidth(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
        )[0],
        "Spectral rolloff": librosa.feature.spectral_rolloff(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
        )[0],
        "MFCC": librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mfcc=13,
        ),
    }


def _print_feature_summary(
    feature_name: str,
    values_by_class: dict[str, list[float]],
    unit: str = "",
) -> None:
    """Print robust summary statistics for one scalar feature."""

    labels = ["healthy", "COVID-19"]
    print(f"\n=== {feature_name}{unit} ===")
    print(f"{'':22}{labels[0]:>14}{labels[1]:>16}{labels[2]:>14}")

    for statistic_name, percentile in [
        ("Mean", None),
        ("Median", 50),
        ("25th percentile", 25),
        ("75th percentile", 75),
    ]:
        row = []
        for label in labels:
            values = np.asarray(values_by_class[label], dtype=float)
            if percentile is None:
                value = np.mean(values)
            else:
                value = np.percentile(values, percentile)
            row.append(f"{value:14.4f}")
        print(f"{statistic_name:22}{''.join(row)}")


def analyze_acoustic_features_by_class(
    metadata_path: Path,
) -> None:
    """Summarize per-recording acoustic features for training data."""

    labels = ["healthy", "COVID-19"]
    scalar_features = {
        "RMS energy": {label: [] for label in labels},
        "Zero-crossing rate": {label: [] for label in labels},
        "Spectral centroid": {label: [] for label in labels},
        "Spectral bandwidth": {label: [] for label in labels},
        "Spectral rolloff": {label: [] for label in labels},
    }
    mfcc_values = {
        coefficient: {label: [] for label in labels}
        for coefficient in range(13)
    }

    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        for row in reader:
            if row["split"] != "train" or row["status"] not in labels:
                continue

            waveform, sample_rate = load_waveform(Path(row["audio_path"]))
            features = _extract_acoustic_features(waveform, sample_rate)
            label = row["status"]

            for feature_name in scalar_features:
                scalar_features[feature_name][label].append(
                    float(np.mean(features[feature_name]))
                )

            for coefficient in range(13):
                mfcc_values[coefficient][label].append(
                    float(np.mean(features["MFCC"][coefficient]))
                )

    print("\n" + "=" * 60)
    print("COUGHVID TRAINING ACOUSTIC FEATURE ANALYSIS")
    print("=" * 60)

    _print_feature_summary("RMS ENERGY", scalar_features["RMS energy"])
    _print_feature_summary(
        "ZERO CROSSING RATE",
        scalar_features["Zero-crossing rate"],
    )
    _print_feature_summary(
        "SPECTRAL CENTROID",
        scalar_features["Spectral centroid"],
        unit=" (Hz)",
    )
    _print_feature_summary(
        "SPECTRAL BANDWIDTH",
        scalar_features["Spectral bandwidth"],
        unit=" (Hz)",
    )
    _print_feature_summary(
        "SPECTRAL ROLLOFF",
        scalar_features["Spectral rolloff"],
        unit=" (Hz)",
    )

    print("\n=== MFCC ===")
    print(f"{'':22}{labels[0]:>14}{labels[1]:>16}{labels[2]:>14}")
    for coefficient in range(13):
        values_by_class = mfcc_values[coefficient]
        row = []
        for label in labels:
            values = np.asarray(values_by_class[label], dtype=float)
            row.append(f"{np.mean(values):14.4f}")
        print(f"MFCC {coefficient + 1} mean{'':10}{''.join(row)}")

def analyze_active_audio_by_class(
    metadata_path: Path,
    top_db: float = 40.0,
) -> None:
    """Analyse non-silent (active) training audio by class.

    Active audio is detected with librosa.effects.split. It should not be
    interpreted as cough-only duration because speech/noise can also be active.
    """

    stats = defaultdict(lambda: {
        "duration": [],
        "active_duration": [],
        "active_ratio": [],
        "intervals": [],
    })

    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        for row in reader:
            if row["split"] != "train":
                continue

            label = row["status"]
            waveform, sr = load_waveform(Path(row["audio_path"]))
            duration = len(waveform) / sr

            intervals = detect_active_intervals(
                waveform, sr, top_db=top_db
            )
            active_samples = sum(end - start for start, end in intervals)
            active_duration = active_samples / sr
            active_ratio = active_duration / duration if duration > 0 else 0.0

            stats[label]["duration"].append(duration)
            stats[label]["active_duration"].append(active_duration)
            stats[label]["active_ratio"].append(active_ratio)
            stats[label]["intervals"].append(len(intervals))

    print("\n" + "=" * 60)
    print("COUGHVID TRAINING ACTIVE AUDIO ANALYSIS")
    print("=" * 60)
    print(f"Threshold: top_db={top_db:g}")
    print("Note: active audio means non-silent audio, not cough-only audio.")

    for label in ["healthy", "COVID-19"]:
        if not stats[label]["duration"]:
            continue

        active = np.asarray(stats[label]["active_duration"], dtype=float)
        ratio = np.asarray(stats[label]["active_ratio"], dtype=float)
        intervals = np.asarray(stats[label]["intervals"], dtype=float)

        print(f"\n=== {label} ===")
        print(f"Recordings:              {len(active):,}")
        print(f"Mean active duration:    {np.mean(active):.2f} sec")
        print(f"Median active duration:  {np.median(active):.2f} sec")
        print(f"Mean active ratio:       {np.mean(ratio) * 100:.2f}%")
        print(f"25th pct active ratio:   {np.percentile(ratio, 25) * 100:.2f}%")
        print(f"Median active ratio:     {np.median(ratio) * 100:.2f}%")
        print(f"75th pct active ratio:   {np.percentile(ratio, 75) * 100:.2f}%")
        print(f"Mean active intervals:   {np.mean(intervals):.2f}")
        print(f"Median active intervals: {np.median(intervals):.0f}")


# =========================================================
# Coswara baseline preprocessing
# =========================================================

COSWARA_SAMPLE_RATE = 16000
COSWARA_TARGET_DURATION = 10.0
COSWARA_TRIM_TOP_DB = 60.0


def preprocess_coswara_waveform(
    audio_path: Path,
    sample_rate: int = COSWARA_SAMPLE_RATE,
    target_duration: float = COSWARA_TARGET_DURATION,
    top_db: float = COSWARA_TRIM_TOP_DB,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Prepare one Coswara recording using the EDA-selected baseline.

    Steps:
    1. Load mono audio at its original sample rate.
    2. Conservatively trim leading/trailing low-energy audio at top_db=60.
    3. Resample to 16 kHz.
    4. Zero-pad or truncate to 10 seconds.

    The waveform is not repeated or time-stretched.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Coswara audio file not found: {audio_path}"
        )

    waveform, original_sr = librosa.load(
        audio_path,
        sr=None,
        mono=True,
    )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.size == 0:
        raise ValueError(
            f"Coswara audio is empty: {audio_path}"
        )

    if not np.isfinite(waveform).all():
        raise ValueError(
            f"Coswara audio contains NaN/Inf: {audio_path}"
        )

    original_samples = len(waveform)
    original_duration = original_samples / original_sr

    trimmed = trim_silence(
        waveform,
        top_db=top_db,
    )

    if trimmed.size == 0:
        raise ValueError(
            f"Coswara audio became empty after trimming: {audio_path}"
        )

    trimmed_duration = len(trimmed) / original_sr

    if original_sr != sample_rate:
        processed = librosa.resample(
            trimmed,
            orig_sr=original_sr,
            target_sr=sample_rate,
        )
    else:
        processed = trimmed

    processed = np.asarray(
        processed,
        dtype=np.float32,
    )

    resampled_samples = len(processed)
    target_samples = int(
        sample_rate * target_duration
    )

    if resampled_samples < target_samples:
        duration_action = "zero_pad"
    elif resampled_samples > target_samples:
        duration_action = "truncate"
    else:
        duration_action = "none"

    fixed = pad_or_truncate_waveform(
        waveform=processed,
        sample_rate=sample_rate,
        target_duration=target_duration,
    ).astype(np.float32, copy=False)

    details = {
        "audio_path": str(audio_path),
        "original_sample_rate": int(original_sr),
        "target_sample_rate": int(sample_rate),
        "original_duration": float(original_duration),
        "trimmed_duration": float(trimmed_duration),
        "trim_top_db": float(top_db),
        "samples_after_resample": int(resampled_samples),
        "target_samples": int(target_samples),
        "duration_action": duration_action,
    }

    return fixed, details


def create_coswara_log_mel(
    audio_path: Path,
    sample_rate: int = COSWARA_SAMPLE_RATE,
    target_duration: float = COSWARA_TARGET_DURATION,
    top_db: float = COSWARA_TRIM_TOP_DB,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Create a fixed Log-Mel spectrogram for one Coswara recording."""

    waveform, details = preprocess_coswara_waveform(
        audio_path=audio_path,
        sample_rate=sample_rate,
        target_duration=target_duration,
        top_db=top_db,
    )

    log_mel = create_log_mel(
        waveform=waveform,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    return log_mel, details
