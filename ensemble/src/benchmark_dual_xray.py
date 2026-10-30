#!/usr/bin/env python3
"""Compare Model B, Model C, and concatenated 256D features on identical cleaned patients.
Reads existing cached feature vectors; never decodes test images or modifies earlier results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def score(y, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "covid_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "noncovid_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "covid_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", type=Path, default=Path("output/xray_modelc_cleaned_v2"))
    p.add_argument("--comparison", type=Path, default=Path("output/xray_modelb_vs_modelc_cleaned"))
    p.add_argument("--output", type=Path, default=Path("output/xray_dual_encoder_256d"))
    p.add_argument("--c", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.425)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    clean = pd.read_csv(args.clean / "clean_manifest.csv")
    idx = pd.read_csv(args.clean / "modelC_features_index.csv")
    required = ["image_id", "key", "patient_id", "label", "split", "png_path"]
    if any(k not in clean.columns or k not in idx.columns for k in required):
        raise RuntimeError("Missing required columns in clean manifest or Model C feature index")
    if not clean[required].reset_index(drop=True).equals(idx[required].reset_index(drop=True)):
        raise RuntimeError("Model C feature index order does not match cleaned manifest")
    if clean.image_id.duplicated().any():
        raise RuntimeError("Duplicate image_id in clean manifest")
    if set(clean.split.unique()) != {"train", "val"}:
        raise RuntimeError("Unexpected split. TEST must not be included")
    if clean.groupby("patient_id").split.nunique().max() != 1:
        raise RuntimeError("Patient leakage across splits")
    if clean.groupby("patient_id").label.nunique().max() != 1:
        raise RuntimeError("Patient label conflict")
    summary = json.loads((args.clean / "final_summary.json").read_text())
    if summary.get("review_pending", -1) != 0:
        raise RuntimeError("Cleaning review not finalized")

    xc = np.load(args.clean / "modelC_features.npy", allow_pickle=False)
    if xc.shape != (len(clean), 128) or not np.isfinite(xc).all():
        raise RuntimeError("Model C features invalid")

    comparison_summary = json.loads((args.comparison / "experiment_summary.json").read_text())
    if comparison_summary.get("cohort_images") != len(clean) or comparison_summary.get("cohort_patients") != clean.key.nunique():
        raise RuntimeError("Previous comparison cohort size differs")
    tag = comparison_summary["model_b_cache_tag"]
    cache_dir = args.comparison / "features" / tag
    if not cache_dir.is_dir():
        raise RuntimeError(f"Model B feature cache missing: {cache_dir}. Run compare_encoders.py first.")
    xb = []
    for image_id in clean.image_id.astype(str):
        f = cache_dir / (image_id + ".npy")
        if not f.is_file():
            raise RuntimeError(f"Missing Model B feature: {f}. Run compare_encoders.py first.")
        vec = np.load(f, allow_pickle=False)
        if vec.shape != (128,) or not np.isfinite(vec).all():
            raise RuntimeError(f"Bad Model B feature: {f}")
        xb.append(vec)
    xb = np.asarray(xb, dtype=np.float32)
    xc = np.asarray(xc, dtype=np.float32)
    xbc = np.concatenate([xb, xc], axis=1)

    train = clean.split.eq("train").to_numpy()
    val = clean.split.eq("val").to_numpy()
    train_counts = clean.loc[train].groupby("key")["image_id"].transform("size").to_numpy()
    weights = 1.0 / train_counts
    weights *= len(weights) / weights.sum()
    print("COHORT", {"train_images": int(train.sum()), "val_images": int(val.sum()),
                     "train_patients": int(clean.loc[train, 'key'].nunique()),
                     "val_patients": int(clean.loc[val, 'key'].nunique()),
                     "model_b_dim": xb.shape[1], "model_c_dim": xc.shape[1],
                     "combined_dim": xbc.shape[1], "test": "NEVER LOADED"}, flush=True)
    if args.smoke:
        print("SMOKE PASS: aligned feature vectors and split integrity verified; no training", flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, X in (("model_b_128d", xb), ("model_c_128d", xc), ("model_bc_256d", xbc)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.c, max_iter=2000, random_state=42))
        clf.fit(X[train], clean.loc[train, "label"].to_numpy(),
                logisticregression__sample_weight=weights)
        image_preds = clean.loc[val, ["key", "label", "image_id"]].copy()
        image_preds["probability"] = clf.predict_proba(X[val])[:, 1]
        patient_preds = image_preds.groupby("key", sort=True).agg(
            label=("label", "first"), probability=("probability", "mean"),
            images=("image_id", "size")
        ).reset_index()
        y = patient_preds.label.to_numpy()
        prob = patient_preds.probability.to_numpy()
        fixed = score(y, prob, args.threshold)
        tuned = max((score(y, prob, t) for t in np.arange(0.2, 0.80001, 0.025)),
                    key=lambda m: (m["macro_f1"], m["balanced_accuracy"], -abs(m["threshold"] - .5)))
        image_preds.to_csv(args.output / f"{name}_image_predictions.csv", index=False)
        patient_preds.to_csv(args.output / f"{name}_patient_predictions.csv", index=False)
        joblib.dump(clf, args.output / f"{name}_logistic.joblib")
        print(name, "FIXED", json.dumps(fixed), flush=True)
        print(name, "TUNED exploratory", json.dumps(tuned), flush=True)
        for mode, values in (("fixed", fixed), ("validation_tuned_exploratory", tuned)):
            reports.append({"model": name, "mode": mode, "features": X.shape[1],
                            "val_patients": len(patient_preds),
                            **{k: v for k, v in values.items() if k != "confusion_matrix"},
                            "confusion_matrix": json.dumps(values["confusion_matrix"])})
    pd.DataFrame(reports).to_csv(args.output / "validation_comparison.csv", index=False)
    report = {
        "train_images": int(train.sum()), "val_images": int(val.sum()),
        "train_patients": int(clean.loc[train, "key"].nunique()),
        "val_patients": int(clean.loc[val, "key"].nunique()),
        "features": {"B": 128, "C": 128, "B_plus_C": 256},
        "model_b_cache_tag": tag,
        "model_b_checkpoint_sha256": comparison_summary.get("model_b_sha256"),
        "model_c_features_file": str(args.clean / "modelC_features.npy"),
        "classifier": {"type": "StandardScaler + LogisticRegression", "C": args.c,
                       "patient_balanced_training": True, "aggregation": "mean probability"},
        "comparison": reports,
        "test": "NOT_LOADED_OR_EVALUATED",
        "caveats": ["Validation threshold tuning is exploratory, not test performance.",
                    "B and C use encoder-specific preprocessing and the same cleaned cohort.",
                    "B feature cache must originate from the prior cleaned comparison."]
    }
    (args.output / "experiment_summary.json").write_text(json.dumps(report, indent=2))
    print("DONE", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
