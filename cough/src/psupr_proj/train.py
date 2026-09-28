"""Training utilities for COUGHVID models."""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from .models import build_baseline_2d_cnn

from collections import Counter

from .model_data import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    create_balanced_tf_dataset,
    create_tf_dataset,
    get_model_input_shape,
    load_split_records,
)


def train_baseline_2d_cnn(
    metadata_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    use_class_weights: bool = False,
) -> keras.Model:
    """Train the baseline Log-Mel 2D CNN."""

    print("\n" + "=" * 60)
    print("COUGHVID BASELINE 2D CNN TRAINING")
    print("=" * 60)

    print(f"Experiment:       {model_name}")
    print(f"Batch size:       {batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"Learning rate:    {learning_rate}")

    # -----------------------------------------------------
    # Training dataset
    # -----------------------------------------------------

    train_dataset = create_tf_dataset(
        metadata_path=metadata_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    # -----------------------------------------------------
    # Validation dataset
    #
    # Do NOT shuffle validation data.
    # -----------------------------------------------------

    validation_dataset = create_tf_dataset(
        metadata_path=metadata_path,
        split="validation",
        batch_size=batch_size,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = build_baseline_2d_cnn()

    # -----------------------------------------------------
    # Compile
    # -----------------------------------------------------

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),

        loss=keras.losses.SparseCategoricalCrossentropy(),

        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            ),
        ],
    )

    print("\n=== Model ===")

    model.summary()

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    model_dir = output_dir / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
    model_dir /
    f"{model_name}.keras"
    )



    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    class_weights = None

    if use_class_weights:

        class_weights = calculate_class_weights(
            metadata_path
        )

        print("\n=== Class Weights ===")

        for class_name in CLASS_NAMES:

            class_index = CLASS_TO_INDEX[
                class_name
            ]

            print(
                f"{class_name:15} "
                f"{class_weights[class_index]:.4f}"
            )
    print("\n=== Training ===")
    model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    class_weight=class_weights,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model.save(
        model_path
    )

    print(
        f"\nModel saved to: "
        f"{model_path}"
    )

    return model

def calculate_class_weights(
    metadata_path: Path,
) -> dict[int, float]:
    """Calculate balanced class weights from training data."""

    records = load_split_records(
        metadata_path=metadata_path,
        split="train",
    )

    counts = Counter(
        record["status"]
        for record in records
    )

    total = len(records)
    num_classes = len(CLASS_NAMES)

    class_weights = {}

    for class_name in CLASS_NAMES:

        class_index = CLASS_TO_INDEX[class_name]
        class_count = counts[class_name]

        weight = (
            total
            / (num_classes * class_count)
        )

        class_weights[class_index] = weight

    return class_weights
'''
def train_balanced_2d_cnn(
    metadata_path: Path,
    output_dir: Path,
    model_name: str = "balanced_sampling_v3",
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Train CNN using balanced class sampling."""
def train_sampled_2d_cnn(
'''


def plot_training_history(
    history: keras.callbacks.History,
    output_path: Path,
    model_name: str,
) -> None:
    """Save accuracy and validation-loss curves from a training run."""

    metrics = history.history
    epochs = range(1, len(metrics["accuracy"]) + 1)

    figure, accuracy_axis = plt.subplots(figsize=(10, 6))
    loss_axis = accuracy_axis.twinx()

    accuracy_axis.plot(
        epochs,
        metrics["accuracy"],
        label="accuracy",
        color="tab:blue",
        linewidth=2,
    )
    accuracy_axis.plot(
        epochs,
        metrics["val_accuracy"],
        label="val_accuracy",
        color="tab:orange",
        linewidth=2,
    )
    loss_axis.plot(
        epochs,
        metrics["val_loss"],
        label="val_loss",
        color="tab:green",
        linewidth=2,
    )

    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    loss_axis.set_ylabel("Validation loss")
    accuracy_axis.set_title(f"Training history: {model_name}")
    accuracy_axis.grid(True, alpha=0.25)

    lines = accuracy_axis.lines + loss_axis.lines
    accuracy_axis.legend(lines, [line.get_label() for line in lines])

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train_sampled_2d_cnn(
    metadata_path: Path,
    output_dir: Path,
    model_name: str,
    sampling_weights: tuple[float, float],
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    use_augmentation: bool = False,
) -> keras.Model:
    print("\n" + "=" * 60)
    print("COUGHVID V3 BALANCED SAMPLING CNN")
    print("=" * 60)

    print(f"Experiment:       {model_name}")
    print(f"Batch size:       {batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"Learning rate:    {learning_rate}")

    # -----------------------------------------------------
    # Balanced TRAINING dataset
    # -----------------------------------------------------
    train_dataset = create_balanced_tf_dataset(
    metadata_path=metadata_path,
    batch_size=batch_size,
    seed=42,
    sampling_weights=sampling_weights,
    use_trim=use_trim,
    use_active_extraction=use_active_extraction,
    use_segmentation=use_segmentation,
    use_augmentation=use_augmentation,
    )

    # -----------------------------------------------------
    # ORIGINAL validation distribution.
    #
    # Do NOT balance validation.
    # -----------------------------------------------------

    validation_dataset = create_tf_dataset(
        metadata_path=metadata_path,
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        use_trim=use_trim,
        use_active_extraction=use_active_extraction,
        use_segmentation=use_segmentation,
        use_augmentation=False,
    )

    model = build_baseline_2d_cnn(
        input_shape=get_model_input_shape(use_segmentation)
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            ),
        ],
    )

    model_dir = (
        output_dir / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / f"{model_name}.keras"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Balanced dataset repeats forever.
    # Therefore Keras needs steps_per_epoch.
    #
    # We keep approximately the same number of training
    # samples per epoch as V1/V2:
    #
    # 9474 / 32 ≈ 297 batches.
    # -----------------------------------------------------

    steps_per_epoch = 297

    print(
        f"Steps per epoch:  "
        f"{steps_per_epoch}"
    )

    print("\n=== Training ===")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping],
    )

    plot_path = model_dir / f"{model_name}_training.png"
    plot_training_history(
        history=history,
        output_path=plot_path,
        model_name=model_name,
    )

    print(f"Training plot saved to: {plot_path}")

    model.save(
        model_path
    )

    print(
        f"\nModel saved to: "
        f"{model_path}"
    )

    return model