#!/usr/bin/env python3
"""Patient-level mean 256D B+C features -> Logistic Regression and MLP.
Reads cached frozen-encoder features. Does not read original or test X-rays.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, precision_score, roc_auc_score, confusion_matrix)
from joblib import dump


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return dict(threshold=round(float(threshold), 6),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                covid_recall=float(recall_score(y, pred, pos_label=1, zero_division=0)),
                noncovid_recall=float(recall_score(y, pred, pos_label=0, zero_division=0)),
                covid_precision=float(precision_score(y, pred, pos_label=1, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def tuned_metrics(y, p):
    return max((metrics(y, p, t) for t in np.arange(.2, .80001, .025)),
               key=lambda m: (m["macro_f1"], m["balanced_accuracy"],
                              -abs(m["threshold"] - .5)))


def load_patient_features(clean_dir, comparison_dir):
    clean = pd.read_csv(clean_dir / "clean_manifest.csv")
    idx = pd.read_csv(clean_dir / "modelC_features_index.csv")
    cols = ["image_id", "key", "patient_id", "label", "split", "png_path"]
    if any(c not in clean.columns or c not in idx.columns for c in cols):
        raise ValueError("Missing required manifest/index columns")
    if not clean[cols].reset_index(drop=True).equals(idx[cols].reset_index(drop=True)):
        raise ValueError("Model C feature index is not aligned with clean manifest")
    if clean.image_id.duplicated().any() or set(clean.split) != {"train", "val"}:
        raise ValueError("Duplicate images or unexpected split (test forbidden)")
    if clean.groupby("patient_id").split.nunique().max() != 1:
        raise ValueError("Patient overlap across train/val")
    if clean.groupby("patient_id").label.nunique().max() != 1:
        raise ValueError("Patient label conflict")
    if clean.groupby("key").label.nunique().max() != 1:
        raise ValueError("Key label conflict")
    final = json.loads((clean_dir / "final_summary.json").read_text())
    if final.get("review_pending") != 0:
        raise ValueError("Manual cleaning review not finalized")
    comp = json.loads((comparison_dir / "experiment_summary.json").read_text())
    if comp.get("cohort_images") != len(clean) or comp.get("cohort_patients") != clean.key.nunique():
        raise ValueError("Model B cache cohort differs from cleaned manifest")
    cache = comparison_dir / "features" / comp["model_b_cache_tag"]
    xc = np.load(clean_dir / "modelC_features.npy", allow_pickle=False)
    xb = np.stack([np.load(cache / (str(i) + ".npy"), allow_pickle=False)
                   for i in clean.image_id.astype(str)])
    if xb.shape != (len(clean), 128) or xc.shape != (len(clean), 128):
        raise ValueError("Feature shape mismatch")
    x = np.concatenate((xb, xc), axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    # Group by patient, preserving original train/val assignments.
    f = pd.DataFrame(x, columns=[f"f_{i:03d}" for i in range(256)])
    f[["key", "patient_id", "label", "split"]] = clean[
        ["key", "patient_id", "label", "split"]].reset_index(drop=True)
    g = f.groupby("key", sort=True)
    patient = g[[f"f_{i:03d}" for i in range(256)]].mean()
    info = g.agg(patient_id=("patient_id", "first"), label=("label", "first"),
                 split=("split", "first"), image_count=("label", "size"))
    info = info.join(patient).reset_index()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, default=Path("output/xray_modelc_cleaned_v2"))
    ap.add_argument("--comparison", type=Path, default=Path("output/xray_modelb_vs_modelc_cleaned"))
    ap.add_argument("--output", type=Path, default=Path("output/xray_patient_feature_aggregation"))
    ap.add_argument("--threshold", type=float, default=.425)
    ap.add_argument("--c", type=float, default=.01)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    pat = load_patient_features(a.clean, a.comparison)
    tr = pat.split.eq("train").to_numpy()
    va = pat.split.eq("val").to_numpy()
    feature_cols = [f"f_{i:03d}" for i in range(256)]
    xtr = pat.loc[tr, feature_cols].to_numpy(dtype=np.float32)
    xval = pat.loc[va, feature_cols].to_numpy(dtype=np.float32)
    ytr = pat.loc[tr, "label"].to_numpy(dtype=int)
    yval = pat.loc[va, "label"].to_numpy(dtype=int)
    if len(set(ytr)) != 2 or len(set(yval)) != 2:
        raise ValueError("Both classes required in train and val")
    print("COHORT", json.dumps(dict(train_patients=len(ytr), val_patients=len(yval),
        train_negative=int((ytr==0).sum()), train_positive=int((ytr==1).sum()),
        features=256, test="NEVER LOADED")), flush=True)
    if a.smoke:
        print("SMOKE PASS: aligned patient-aggregated features; no training", flush=True)
        return
    a.output.mkdir(parents=True, exist_ok=True)
    pat[["key", "patient_id", "label", "split", "image_count"]].to_csv(
        a.output / "patient_aggregation_index.csv", index=False)
    np.save(a.output / "patient_features_256d.npy",
            pat[feature_cols].to_numpy(dtype=np.float32))
    results = {}

    lr = make_pipeline(StandardScaler(), LogisticRegression(
        C=a.c, max_iter=2000, random_state=a.seed))
    lr.fit(xtr, ytr)
    p_lr = lr.predict_proba(xval)[:, 1]
    dump(lr, a.output / "patient_logistic.joblib")
    results["patient_logistic"] = {
        "fixed": metrics(yval, p_lr, a.threshold),
        "tuned_exploratory": tuned_metrics(yval, p_lr)}
    print("PATIENT LOGISTIC FIXED", json.dumps(results["patient_logistic"]["fixed"]), flush=True)
    print("PATIENT LOGISTIC TUNED exploratory",
          json.dumps(results["patient_logistic"]["tuned_exploratory"]), flush=True)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr).astype(np.float32)
    xval_s = scaler.transform(xval).astype(np.float32)
    reg = tf.keras.regularizers.l2(1e-4)
    mlp = tf.keras.Sequential([
        tf.keras.Input(shape=(256,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="binary_crossentropy")
    best = -1
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, a.epochs + 1):
        h = mlp.fit(xtr_s, ytr, epochs=1, batch_size=a.batch_size,
                    verbose=0, shuffle=True)
        prob = mlp.predict(xval_s, verbose=0).ravel()
        m = metrics(yval, prob, a.threshold)
        history.append(dict(epoch=epoch, train_loss=float(h.history["loss"][0]),
                            val_macro_f1=m["macro_f1"], val_auc=m["roc_auc"]))
        print(f"epoch={epoch:03d} loss={history[-1]['train_loss']:.4f} "
              f"val_macro_f1={m['macro_f1']:.4f} val_auc={m['roc_auc']:.4f}", flush=True)
        if m["macro_f1"] > best + 1e-8:
            best = m["macro_f1"]
            best_epoch = epoch
            bad = 0
            mlp.save(a.output / "best_patient_mlp.keras")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"EARLY STOP: no macro F1 improvement for {a.patience} epochs", flush=True)
                break
    mlp = tf.keras.models.load_model(a.output / "best_patient_mlp.keras")
    p_mlp = mlp.predict(xval_s, verbose=0).ravel()
    dump(scaler, a.output / "patient_mlp_scaler.joblib")
    pd.DataFrame(history).to_csv(a.output / "training_history.csv", index=False)
    results["patient_mlp"] = {
        "best_epoch": best_epoch,
        "fixed": metrics(yval, p_mlp, a.threshold),
        "tuned_exploratory": tuned_metrics(yval, p_mlp)}
    print("PATIENT MLP FIXED", json.dumps(results["patient_mlp"]["fixed"]), flush=True)
    print("PATIENT MLP TUNED exploratory",
          json.dumps(results["patient_mlp"]["tuned_exploratory"]), flush=True)

    pred = pat.loc[va, ["key", "patient_id", "label", "image_count"]].copy()
    pred["logistic_probability"] = p_lr
    pred["mlp_probability"] = p_mlp
    pred.to_csv(a.output / "validation_patient_predictions.csv", index=False)
    rows = []
    for name, result in results.items():
        for mode in ("fixed", "tuned_exploratory"):
            m = result[mode]
            rows.append(dict(model=name, mode=mode,
                             **{k:v for k,v in m.items() if k!="confusion_matrix"},
                             confusion_matrix=json.dumps(m["confusion_matrix"])))
    pd.DataFrame(rows).to_csv(a.output / "validation_comparison.csv", index=False)
    summary = dict(cohort=dict(train_patients=len(ytr), val_patients=len(yval),
                   train_images=int(pat.loc[tr, "image_count"].sum()),
                   val_images=int(pat.loc[va, "image_count"].sum())),
                   aggregation="Mean B+C 256D image features per patient before training",
                   models=results, test="NOT LOADED OR EVALUATED",
                   caveat="Validation reused for early stopping and threshold exploration.")
    (a.output / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", a.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
