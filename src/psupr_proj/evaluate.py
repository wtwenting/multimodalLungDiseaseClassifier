"""Evaluation utilities for COUGHVID classification models."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

from .model_data import (
    CLASS_NAMES,
    create_tf_dataset,
)


def evaluate_baseline_2d_cnn(
    metadata_path: Path,
    model_path: Path,
    batch_size: int = 32,
) -> None:
    """Evaluate the baseline CNN on the validation dataset."""

    print("\n" + "=" * 60)
    print("COUGHVID BASELINE 2D CNN EVALUATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load saved model
    # -----------------------------------------------------

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print(f"Model: {model_path}")

    model = keras.models.load_model(
        model_path
    )

    # -----------------------------------------------------
    # Validation dataset
    #
    # IMPORTANT:
    # Do not shuffle during evaluation.
    # -----------------------------------------------------

    validation_dataset = create_tf_dataset(
        metadata_path=metadata_path,
        split="validation",
        batch_size=batch_size,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Collect true labels and predictions
    # -----------------------------------------------------

    y_true = []
    y_pred = []

    print("\nGenerating predictions...")

    for x_batch, y_batch in validation_dataset:

        probabilities = model.predict(
            x_batch,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        y_true.extend(
            y_batch.numpy()
        )

        y_pred.extend(
            predictions
        )

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    # -----------------------------------------------------
    # Overall accuracy
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print("\n=== Overall Accuracy ===")

    print(
        f"Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    # -----------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------

    print("\n=== Prediction Distribution ===")

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(y_pred == index)
        )

        percentage = (
            count / len(y_pred) * 100
        )

        print(
            f"{class_name:15} "
            f"{count:6,} "
            f"({percentage:6.2f}%)"
        )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
    )

    print("\n=== Confusion Matrix ===")

    print(
        "\nRows    = Actual class"
        "\nColumns = Predicted class\n"
    )

    print(
        f"{'':15}"
        f"{'healthy':>12}"
        f"{'symptomatic':>15}"
        f"{'COVID-19':>12}"
    )

    for class_name, row in zip(
        CLASS_NAMES,
        matrix,
    ):

        print(
            f"{class_name:15}"
            f"{row[0]:12,}"
            f"{row[1]:15,}"
            f"{row[2]:12,}"
        )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\n=== Classification Report ===")

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report)

    # -----------------------------------------------------
    # Macro F1
    # -----------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )