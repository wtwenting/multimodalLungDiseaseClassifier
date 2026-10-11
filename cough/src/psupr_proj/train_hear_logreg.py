"""Train a balanced logistic-regression classifier on frozen HeAR embeddings.

Phase-1 downstream baseline:
    HeAR 512-D embedding -> StandardScaler -> LogisticRegression

The classifier is fit on TRAIN only. Validation and test are reported separately.
No HeAR weights are changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_split(path: Path):
    d = np.load(path, allow_pickle=False)
    X = d["embeddings"].astype(np.float32)
    y_text = d["label"].astype(str)
    uuid = d["uuid"].astype(str)

    # Positive class = COVID-19
    y = (y_text == "COVID-19").astype(np.int64)

    if X.ndim != 2 or X.shape[1] != 512:
        raise ValueError(f"{path}: expected embeddings shape (N, 512), got {X.shape}")
    if len(X) != len(y) or len(y) != len(uuid):
        raise ValueError(f"{path}: inconsistent row counts")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: embeddings contain NaN or Inf")

    return X, y, y_text, uuid


def report(name: str, model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, pred, labels=[0, 1])
    macro_f1 = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, prob)

    p, r, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )

    print(f"\n{'=' * 68}")
    print(name)
    print("=" * 68)
    print(f"Accuracy:       {acc:.4f}")
    print(f"Macro-F1:       {macro_f1:.4f}")
    print(f"ROC-AUC:        {auc:.4f}")
    print(f"COVID precision:{p[1]:.4f}")
    print(f"COVID recall:   {r[1]:.4f}")
    print(f"COVID F1:       {f1[1]:.4f}")
    print(f"COVID support:  {support[1]}")
    print("\nConfusion matrix [healthy, COVID-19]:")
    print(cm)
    print("\nClassification report:")
    print(
        classification_report(
            y,
            pred,
            labels=[0, 1],
            target_names=["healthy", "COVID-19"],
            digits=4,
            zero_division=0,
        )
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "covid_precision": p[1],
        "covid_recall": r[1],
        "covid_f1": f1[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/hear_embeddings"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hear_logreg.joblib"),
        help="Where to save the fitted scaler + classifier.",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    args = parser.parse_args()

    train_path = args.embedding_dir / "hear_train.npz"
    val_path = args.embedding_dir / "hear_validation.npz"
    test_path = args.embedding_dir / "hear_test.npz"

    X_train, y_train, _, _ = load_split(train_path)
    X_val, y_val, _, _ = load_split(val_path)
    X_test, y_test, _, _ = load_split(test_path)

    print("HeAR LOGISTIC-REGRESSION BASELINE")
    print("=" * 68)
    print(f"Train:      {X_train.shape} | COVID={y_train.sum()} | healthy={(y_train == 0).sum()}")
    print(f"Validation: {X_val.shape} | COVID={y_val.sum()} | healthy={(y_val == 0).sum()}")
    print(f"Test:       {X_test.shape} | COVID={y_test.sum()} | healthy={(y_test == 0).sum()}")
    print("\nClassifier:")
    print("  StandardScaler")
    print("  LogisticRegression(class_weight='balanced')")
    print("  Positive class = COVID-19")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    # Fit TRAIN only.
    model.fit(X_train, y_train)

    train_metrics = report("TRAIN", model, X_train, y_train)
    val_metrics = report("VALIDATION", model, X_val, y_val)
    test_metrics = report("TEST (final held-out split)", model, X_test, y_test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    print(f"\nSaved fitted model: {args.output}")
    print("\nComparison priorities:")
    print("  1. Macro-F1")
    print("  2. COVID-19 F1")
    print("  3. COVID-19 recall")
    print("  4. Confusion matrix")
    print("  5. Accuracy (secondary because the classes are imbalanced)")

    # Compact final summary for easy copy/paste.
    print("\nSUMMARY")
    print("split,accuracy,macro_f1,roc_auc,covid_precision,covid_recall,covid_f1")
    for name, m in [
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics),
    ]:
        print(
            f"{name},{m['accuracy']:.4f},{m['macro_f1']:.4f},{m['roc_auc']:.4f},"
            f"{m['covid_precision']:.4f},{m['covid_recall']:.4f},{m['covid_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
