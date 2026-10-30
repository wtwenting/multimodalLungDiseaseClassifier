"""Train a nonlinear classifier on frozen 512-D HeAR embeddings.

Development workflow:
  TRAIN      -> fit scaler and neural-network weights
  VALIDATION -> early stopping and development metrics
  TEST       -> NOT evaluated unless --evaluate-test is explicitly supplied

HeAR remains frozen; this script trains only the downstream classifier.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def load_split(path: Path):
    d = np.load(path, allow_pickle=False)
    X = d["embeddings"].astype(np.float32)
    labels = d["label"].astype(str)
    uuids = d["uuid"].astype(str)
    y = (labels == "COVID-19").astype(np.float32)

    if X.ndim != 2 or X.shape[1] != 512:
        raise ValueError(f"{path}: expected (N, 512), got {X.shape}")
    if len(X) != len(y) or len(y) != len(uuids):
        raise ValueError(f"{path}: inconsistent row counts")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: embeddings contain NaN/Inf")
    return X, y, labels, uuids


def balanced_class_weights(y):
    n = len(y)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    return {0: n / (2.0 * n0), 1: n / (2.0 * n1)}


def build_model(input_dim: int = 512):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    inputs = keras.Input(shape=(input_dim,), name="hear_embedding")

    x = layers.Dense(
        128,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_128",
    )(inputs)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Activation("relu", name="relu_128")(x)
    x = layers.Dropout(0.40, name="dropout_128")(x)

    x = layers.Dense(
        32,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_32",
    )(x)
    x = layers.BatchNormalization(name="bn_32")(x)
    x = layers.Activation("relu", name="relu_32")(x)
    x = layers.Dropout(0.30, name="dropout_32")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="covid_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="hear_dense_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def report(name, model, X, y, threshold=0.5):
    prob = model.predict(X, verbose=0).reshape(-1)
    pred = (prob >= threshold).astype(np.int64)
    yi = y.astype(np.int64)

    acc = accuracy_score(yi, pred)
    macro_f1 = f1_score(yi, pred, average="macro")
    auc = roc_auc_score(yi, prob)
    p, r, f1, support = precision_recall_fscore_support(
        yi, pred, labels=[0, 1], zero_division=0
    )

    print(f"\n{'=' * 68}")
    print(f"{name} @ threshold={threshold:.2f}")
    print("=" * 68)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")
    print(f"COVID precision: {p[1]:.4f}")
    print(f"COVID recall:    {r[1]:.4f}")
    print(f"COVID F1:        {f1[1]:.4f}")
    print("\nConfusion matrix [healthy, COVID-19]:")
    print(confusion_matrix(yi, pred, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            yi,
            pred,
            labels=[0, 1],
            target_names=["healthy", "COVID-19"],
            digits=4,
            zero_division=0,
        )
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "roc_auc": float(auc),
        "covid_precision": float(p[1]),
        "covid_recall": float(r[1]),
        "covid_f1": float(f1[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/hear_embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hear_dense"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicitly evaluate the held-out test set. Do this only after development is finished.",
    )
    args = parser.parse_args()

    set_seed()

    X_train, y_train, _, _ = load_split(args.embedding_dir / "hear_train.npz")
    X_val, y_val, _, _ = load_split(args.embedding_dir / "hear_validation.npz")

    print("HeAR DENSE CLASSIFIER — DEVELOPMENT")
    print("=" * 68)
    print(f"Train:      {X_train.shape} | COVID={int(y_train.sum())} | healthy={int((y_train == 0).sum())}")
    print(f"Validation: {X_val.shape} | COVID={int(y_val.sum())} | healthy={int((y_val == 0).sum())}")
    print("Test:       intentionally untouched during development")

    # Fit normalization on TRAIN only.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    class_weight = balanced_class_weights(y_train)
    print(f"\nClass weights: healthy={class_weight[0]:.4f}, COVID-19={class_weight[1]:.4f}")

    model = build_model(512)
    model.summary()

    from tensorflow import keras

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_model.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    # Ensure evaluation uses the best saved validation-AUC checkpoint.
    model = keras.models.load_model(best_model_path)

    train_metrics = report("TRAIN", model, X_train_s, y_train)
    val_metrics = report("VALIDATION", model, X_val_s, y_val)

    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    with open(args.output_dir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    summary = {"train": train_metrics, "validation": val_metrics}

    if args.evaluate_test:
        print("\nWARNING: evaluating the final held-out test split.")
        X_test, y_test, _, _ = load_split(args.embedding_dir / "hear_test.npz")
        X_test_s = scaler.transform(X_test).astype(np.float32)
        summary["test"] = report("TEST — FINAL HELD-OUT", model, X_test_s, y_test)

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"  {best_model_path}")
    print(f"  {args.output_dir / 'scaler.joblib'}")
    print(f"  {args.output_dir / 'history.json'}")
    print(f"  {args.output_dir / 'metrics.json'}")

    print("\nDEVELOPMENT SUMMARY")
    print("split,accuracy,macro_f1,roc_auc,covid_precision,covid_recall,covid_f1")
    for split_name in ["train", "validation", "test"]:
        if split_name not in summary:
            continue
        m = summary[split_name]
        print(
            f"{split_name},{m['accuracy']:.4f},{m['macro_f1']:.4f},{m['roc_auc']:.4f},"
            f"{m['covid_precision']:.4f},{m['covid_recall']:.4f},{m['covid_f1']:.4f}"
        )

    if not args.evaluate_test:
        print("\nTest set was NOT evaluated. Keep it untouched until the Dense model design is finalized.")


if __name__ == "__main__":
    main()
