"""Run all three Coswara cough baselines sequentially.

Models:
1. Log-Mel -> 2D CNN
2. Raw waveform -> 1D CNN
3. MFCC sequence -> LSTM

All models use the same fixed Coswara train/validation/test split and the
same waveform preprocessing:
- conservative trim: top_db=60
- resample: 16 kHz
- fixed duration: 10 seconds
- shorter recordings: zero-pad
- longer recordings: truncate

Baseline run:
- no augmentation
- no oversampling
- no class weights
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

from psupr_proj.features import (
    create_log_mel,
    preprocess_coswara_waveform,
)
from psupr_proj.models import (
    build_baseline_1d_cnn,
    build_baseline_2d_cnn,
    build_baseline_lstm,
)


CLASS_NAMES = ["healthy", "COVID-19"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

SAMPLE_RATE = 16000
TARGET_DURATION = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)
TOP_DB = 60.0

N_MELS = 64
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 256

LOGMEL_SHAPE = (
    N_MELS,
    1 + TARGET_SAMPLES // HOP_LENGTH,
    1,
)
WAVEFORM_SHAPE = (
    TARGET_SAMPLES,
    1,
)
MFCC_SHAPE = (
    1 + TARGET_SAMPLES // HOP_LENGTH,
    N_MFCC,
)

REPRESENTATIONS = {
    "2d_cnn": "logmel",
    "1d_cnn": "waveform",
    "lstm": "mfcc",
}


def load_records(
    metadata_path: Path,
    split: str,
) -> list[dict[str, str]]:
    """Load one fixed split from the Coswara metadata CSV."""

    records = []

    with metadata_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "uuid",
            "status",
            "audio_path",
            "split",
        }

        if not reader.fieldnames:
            raise ValueError("Metadata CSV has no header.")

        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Missing metadata columns: {sorted(missing)}"
            )

        for row in reader:
            if (
                row["split"].strip() == split
                and row["status"].strip() in CLASS_TO_INDEX
            ):
                records.append(row)

    if not records:
        raise ValueError(
            f"No records found for split '{split}'."
        )

    return records


def load_representation(
    audio_path: str,
    representation: str,
) -> np.ndarray:
    """Load one WAV and create the requested model representation."""

    waveform, _ = preprocess_coswara_waveform(
        audio_path=Path(audio_path),
        sample_rate=SAMPLE_RATE,
        target_duration=TARGET_DURATION,
        top_db=TOP_DB,
    )

    if representation == "waveform":
        return np.expand_dims(
            waveform,
            axis=-1,
        ).astype(np.float32)

    if representation == "logmel":
        log_mel = create_log_mel(
            waveform=waveform,
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        return np.expand_dims(
            log_mel,
            axis=-1,
        ).astype(np.float32)

    if representation == "mfcc":
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        ).T

        return mfcc.astype(np.float32)

    raise ValueError(
        f"Unknown representation: {representation}"
    )


def representation_shape(
    representation: str,
) -> tuple[int, ...]:
    """Return static TensorFlow shape for one representation."""

    if representation == "logmel":
        return LOGMEL_SHAPE

    if representation == "waveform":
        return WAVEFORM_SHAPE

    if representation == "mfcc":
        return MFCC_SHAPE

    raise ValueError(
        f"Unknown representation: {representation}"
    )


def create_dataset(
    metadata_path: Path,
    split: str,
    representation: str,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a streaming TensorFlow dataset."""

    records = load_records(
        metadata_path,
        split,
    )

    paths = [
        row["audio_path"]
        for row in records
    ]

    labels = [
        CLASS_TO_INDEX[
            row["status"].strip()
        ]
        for row in records
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    shape = representation_shape(
        representation
    )

    def process(
        audio_path: tf.Tensor,
        label: tf.Tensor,
    ):
        def python_load(path, y):
            path_string = (
                path.numpy().decode("utf-8")
            )

            x = load_representation(
                path_string,
                representation,
            )

            return (
                x,
                np.int64(y.numpy()),
            )

        x, y = tf.py_function(
            python_load,
            [audio_path, label],
            [tf.float32, tf.int64],
        )

        x.set_shape(shape)
        y.set_shape(())

        return x, y

    dataset = dataset.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size
    )

    return dataset.prefetch(
        tf.data.AUTOTUNE
    )


def build_model(
    model_key: str,
) -> keras.Model:
    """Build the matching existing cough-model architecture."""

    if model_key == "2d_cnn":
        return build_baseline_2d_cnn(
            input_shape=LOGMEL_SHAPE,
            num_classes=2,
        )

    if model_key == "1d_cnn":
        return build_baseline_1d_cnn(
            input_shape=WAVEFORM_SHAPE,
            num_classes=2,
        )

    if model_key == "lstm":
        return build_baseline_lstm(
            input_shape=MFCC_SHAPE,
            num_classes=2,
        )

    raise ValueError(
        f"Unknown model: {model_key}"
    )


def evaluate_model(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> dict[str, object]:
    """Evaluate one trained model on the untouched test split."""

    y_true = []
    y_pred = []

    for x_batch, y_batch in dataset:
        probabilities = model.predict(
            x_batch,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        y_true.extend(
            y_batch.numpy().tolist()
        )

        y_pred.extend(
            predictions.tolist()
        )

    y_true_array = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    matrix = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
    )

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true_array,
                y_pred_array,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "classification_report": report,
    }


def run_one_model(
    model_key: str,
    metadata_path: Path,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> dict[str, object]:
    """Train, save, and test one baseline model."""

    representation = REPRESENTATIONS[
        model_key
    ]

    print("\n" + "=" * 70)
    print(
        f"COSWARA BASELINE: {model_key.upper()}"
    )
    print("=" * 70)
    print(
        f"Representation: {representation}"
    )

    train_dataset = create_dataset(
        metadata_path=metadata_path,
        split="train",
        representation=representation,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_dataset = create_dataset(
        metadata_path=metadata_path,
        split="validation",
        representation=representation,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_model(
        model_key
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            )
        ],
    )

    model.summary()

    model_dir = (
        output_dir / "models"
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / f"coswara_{model_key}_baseline.keras"
    )

    best_model_path = (
        model_dir
        / f"coswara_{model_key}_baseline.best.keras"
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    start = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - start

    model.save(
        model_path
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    best_epoch = (
        int(
            np.argmin(
                history_data["val_loss"]
            )
        )
        + 1
    )

    # Evaluate the restored best weights on the test split only now.
    test_dataset = create_dataset(
        metadata_path=metadata_path,
        split="test",
        representation=representation,
        batch_size=batch_size,
        shuffle=False,
    )

    evaluation = evaluate_model(
        model,
        test_dataset,
    )

    result = {
        "model_key": model_key,
        "representation": representation,
        "model_path": str(model_path),
        "best_model_path": str(best_model_path),
        "epochs_requested": epochs,
        "epochs_run": len(
            history_data["val_loss"]
        ),
        "best_epoch": best_epoch,
        "best_val_loss": float(
            history_data["val_loss"][
                best_epoch - 1
            ]
        ),
        "training_seconds": float(
            elapsed
        ),
        "history": history_data,
        "test": evaluation,
    }

    result_path = (
        model_dir
        / f"coswara_{model_key}_baseline.results.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== TEST RESULT ===")
    print(
        f"Accuracy: {evaluation['accuracy']:.4f}"
    )
    print(
        f"Macro F1: {evaluation['macro_f1']:.4f}"
    )
    print(
        "Confusion matrix: "
        f"{evaluation['confusion_matrix']}"
    )
    print(
        f"Saved: {result_path}"
    )

    # Release memory before the next architecture.
    keras.backend.clear_session()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train all three Coswara baseline cough models sequentially."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/coswara/coswara_heavy_split.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/coswara/baseline_runs"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "2d_cnn",
            "1d_cnn",
            "lstm",
        ],
        default=[
            "2d_cnn",
            "1d_cnn",
            "lstm",
        ],
    )

    args = parser.parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Split metadata not found: {args.metadata}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("COSWARA OVERNIGHT BASELINE RUN")
    print("=" * 70)
    print(f"Metadata:       {args.metadata}")
    print(f"Output:         {args.output_dir}")
    print(f"Models:         {', '.join(args.models)}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Max epochs:     {args.epochs}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Early stopping: patience={args.patience}")
    print("\nPreprocessing:")
    print(f"  trim top_db:  {TOP_DB:g}")
    print(f"  sample rate:  {SAMPLE_RATE:,} Hz")
    print(f"  duration:     {TARGET_DURATION:g} sec")
    print("  short audio:  zero-pad")
    print("  long audio:   truncate")
    print("\nBaseline controls:")
    print("  augmentation: OFF")
    print("  oversampling: OFF")
    print("  class weights: OFF")

    all_results = {}

    for model_key in args.models:
        try:
            result = run_one_model(
                model_key=model_key,
                metadata_path=args.metadata,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
            )
            all_results[
                model_key
            ] = result

        except Exception as exc:
            # Record the failure, then continue to the next model so one
            # architecture does not waste the whole overnight run.
            print(
                f"\nERROR while running {model_key}: "
                f"{type(exc).__name__}: {exc}"
            )

            all_results[
                model_key
            ] = {
                "error_type": (
                    type(exc).__name__
                ),
                "error": str(exc),
            }

            keras.backend.clear_session()

    summary = {
        "metadata": str(args.metadata),
        "preprocessing": {
            "top_db": TOP_DB,
            "sample_rate": SAMPLE_RATE,
            "target_duration": TARGET_DURATION,
            "padding": "zero_pad",
            "truncation": "end",
        },
        "baseline": {
            "augmentation": False,
            "oversampling": False,
            "class_weights": False,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "early_stopping_patience": args.patience,
        },
        "results": all_results,
    }

    summary_path = (
        args.output_dir
        / "coswara_baseline_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("OVERNIGHT RUN COMPLETE")
    print("=" * 70)
    print(
        f"Summary: {summary_path}"
    )

    for model_key, result in all_results.items():
        if "test" in result:
            print(
                f"{model_key:8}  "
                f"accuracy={result['test']['accuracy']:.4f}  "
                f"macro_f1={result['test']['macro_f1']:.4f}"
            )
        else:
            print(
                f"{model_key:8}  FAILED: "
                f"{result.get('error', 'unknown error')}"
            )


if __name__ == "__main__":
    main()
