#!/usr/bin/env python3
"""Train/evaluate logistic regression from finalized Model C 128-D features.
Read-only with respect to source cleaning outputs and original X-rays.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, precision_score, roc_auc_score,
                             confusion_matrix)
import joblib


def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "covid_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "noncovid_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "covid_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/xray_modelc_cleaned_v2")
    ap.add_argument("--output", default="output/xray_modelc_cleaned_baseline")
    ap.add_argument("--c", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.425)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = Path(args.input)
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)

    # Fail closed on unfinalized or incomplete cleaning.
    final_summary = json.loads((root / "final_summary.json").read_text())
    if final_summary.get("review_pending", -1) != 0:
        raise RuntimeError("Cleaning has pending reviews. Finalize first.")
    clean = pd.read_csv(root / "clean_manifest.csv")
    idx = pd.read_csv(root / "modelC_features_index.csv")
    features = np.load(root / "modelC_features.npy", allow_pickle=False)
    needed = ["image_id", "key", "patient_id", "label", "split", "png_path"]
    if any(c not in clean or c not in idx for c in needed):
        raise RuntimeError("Missing required columns in feature index or clean manifest")
    if len(features) != len(idx) or features.shape != (len(idx), 128):
        raise RuntimeError("Feature shape/index mismatch")
    if idx["image_id"].duplicated().any() or clean["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image IDs")
    # Feature order is critical: do not silently reorder.
    if not idx[needed].reset_index(drop=True).equals(clean[needed].reset_index(drop=True)):
        raise RuntimeError("Feature index differs from clean manifest or row order")
    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite features")
    if set(idx["split"]) - {"train", "val"}:
        raise RuntimeError("Unexpected split: test data must not be loaded")
    if set(idx["label"]) != {0, 1}:
        raise RuntimeError("Expected both binary labels 0/1")
    if idx.groupby("key")["split"].nunique().max() != 1:
        raise RuntimeError("Patient leakage across splits")
    if idx.groupby("patient_id")["split"].nunique().max() != 1:
        raise RuntimeError("Raw patient ID leakage across splits")
    if idx.groupby("patient_id")["label"].nunique().max() != 1:
        raise RuntimeError("Conflicting patient labels")
    train_mask = idx["split"].eq("train").to_numpy()
    val_mask = idx["split"].eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("Missing train/validation data")
    counts = idx.loc[train_mask].groupby("key")["image_id"].transform("size").to_numpy()
    weights = 1.0 / counts
    print(f"VALIDATED: {len(idx)} images, {idx.key.nunique()} patients; "
          f"train={int(train_mask.sum())}, val={int(val_mask.sum())}", flush=True)
    if args.smoke:
        print("SMOKE PASS: features and patient isolation verified; no model fitting", flush=True)
        return
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c, max_iter=2000, random_state=42)
    )
    model.fit(features[train_mask], idx.loc[train_mask, "label"].to_numpy(),
              logisticregression__sample_weight=weights)
    val = idx.loc[val_mask, needed].copy().reset_index(drop=True)
    val["probability"] = model.predict_proba(features[val_mask])[:, 1]
    # Same aggregation strategy as earlier benchmark: mean probability per patient.
    patients = val.groupby("key", sort=True).agg(
        label=("label", "first"),
        probability=("probability", "mean"),
        image_count=("image_id", "size")
    ).reset_index()
    y = patients["label"].to_numpy()
    prob = patients["probability"].to_numpy()
    fixed = metrics(y, prob, args.threshold)
    best = None
    for t in np.arange(0.2, 0.8001, 0.025):
        m = metrics(y, prob, t)
        rank = (m["macro_f1"], m["balanced_accuracy"], -abs(t - 0.5))
        if best is None or rank > best[0]:
            best = (rank, m)
    tuned = best[1]
    val.to_csv(dest / "validation_image_predictions.csv", index=False)
    patients.to_csv(dest / "validation_patient_predictions.csv", index=False)
    joblib.dump(model, dest / "cleaned_modelc_logistic.joblib")
    result = {
        "encoder": "Model C ResNet50, frozen features from finalized cleaning",
        "classifier": "StandardScaler + LogisticRegression",
        "C": args.c, "patient_aggregation": "mean image probability",
        "train_images": int(train_mask.sum()),
        "val_images": int(val_mask.sum()),
        "train_patients": int(idx.loc[train_mask, "key"].nunique()),
        "val_patients": int(len(patients)),
        "fixed_validation": fixed,
        "validation_tuned_exploratory": tuned,
        "test": "NOT_LOADED_OR_EVALUATED",
        "note": "Previous uncleaned results are not directly comparable if validation cohort changed."
    }
    (dest / "experiment_summary.json").write_text(json.dumps(result, indent=2))
    print("FIXED RESULT", json.dumps(fixed), flush=True)
    print("TUNED RESULT (exploratory)", json.dumps(tuned), flush=True)
    print("DONE", dest.resolve(), flush=True)


if __name__ == "__main__":
    main()
