"""Evaluation utilities for COUGHVID classification models."""

import json
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
    use_trim: bool = False,
    use_active_extraction: bool = False,
    use_segmentation: bool = False,
    model_type: str = "model1",
    split: str = "test",
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
        split=split,
        batch_size=batch_size,
        shuffle=False,
        use_trim=use_trim,
        use_active_extraction=use_active_extraction,
        use_segmentation=use_segmentation,
        representation={"model1": "logmel", "model2": "waveform", "model3": "mfcc"}[model_type],
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
        + "".join(
            f"{class_name:>15}"
            for class_name in CLASS_NAMES
        )
    )

    for class_name, row in zip(
        CLASS_NAMES,
        matrix,
    ):

        print(
            f"{class_name:15}"
            + "".join(
                f"{value:15,}"
                for value in row
            )
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

    report_data = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
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

    evaluation_path = model_path.with_suffix(".evaluation.json")
    training_metadata_path = model_path.with_suffix(".training.json")
    evaluation = {
        "model": str(model_path),
        "metadata": str(metadata_path),
        "split": split,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "best_epoch": None,
        "classes": CLASS_NAMES,
        "prediction_distribution": {
            class_name: int(np.sum(y_pred == index))
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_data,
    }

    if training_metadata_path.exists():
        training_metadata = json.loads(
            training_metadata_path.read_text(encoding="utf-8")
        )
        evaluation["best_epoch"] = training_metadata.get("best_epoch")

    evaluation_path.write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Evaluation report saved to: {evaluation_path}")

    text_path = model_path.with_suffix(".evaluation.txt")
    lines = [
        f"Model: {model_path}",
        f"Split: {split}",
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"Macro F1: {macro_f1:.4f}",
        "",
        "Rows = Actual class; Columns = Predicted class",
        "",
        f"{'':15}" + "".join(f"{name:>15}" for name in CLASS_NAMES),
    ]
    for class_name, row in zip(CLASS_NAMES, matrix):
        lines.append(f"{class_name:15}" + "".join(f"{int(value):15,d}" for value in row))
    lines.extend(["", "=== Classification Report ===", report])
    text_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Human-readable report saved to: {text_path}")