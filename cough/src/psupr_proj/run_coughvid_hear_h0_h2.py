#!/usr/bin/env python3
"""Reproducible COUGHVID + frozen HeAR H0/H1/H2 experiments.

Research question
-----------------
Does COVID-only waveform augmentation add predictive information beyond merely
repeating minority-class HeAR embeddings at the same 70:30 training ratio?

H0: original HeAR embeddings, class-weighted logistic regression / MLP.
H1: original HeAR embeddings, COVID oversampled to 30% of training examples.
H2: original HeAR embeddings plus independently seeded *augmented* COVID audio
    embedded by the same frozen HeAR model; COVID reaches 30% of presentations.

The script NEVER reads test embeddings and NEVER augments validation recordings.
Feature selection (512/256/128/64/32) uses ANOVA fitted to ORIGINAL training
recordings only; the fitted scaler and selector are reused for all strategies.

Outputs are written incrementally and can be resumed. The source embeddings
and source recordings are read-only. Augmented WAVs, provenance CSV, augmented
NPZ, model pipelines, and validation reports are retained for reproducibility.

Run from repository root:
    python src/psupr_proj/run_coughvid_hear_h0_h2.py --stages h0 h1 h2
    python src/psupr_proj/run_coughvid_hear_h0_h2.py --stages h0 h1

Requires: numpy, scipy, scikit-learn, librosa, soundfile, joblib.
Only H2 embedding extraction requires HeAR dependencies and model access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LABELS = ("healthy", "COVID-19")
FEATURE_COUNTS = (512, 256, 128, 64, 32)
SAMPLE_RATE = 16000
CLIP_SAMPLES = 32000
FEATURE_DIM = 512
SEED = 42


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data):
    """Atomically replace JSON so a stopped run leaves the previous valid file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(uuid: str, augmentation_index: int, base_seed: int):
    """Stable per-recording seed independent of Python hash randomization."""
    payload = f"{base_seed}|{uuid}|{augmentation_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_embeddings(path: Path, expected_split: str):
    """Read only allowed binary labels; reject malformed and duplicate IDs."""
    with np.load(path, allow_pickle=False) as z:
        for key in ("embeddings", "uuid", "label"):
            if key not in z:
                raise ValueError(f"{path} missing NPZ key {key}")
        x = z["embeddings"].astype(np.float32)
        ids = z["uuid"].astype(str)
        labels = z["label"].astype(str)
        if "split" in z and not np.all(z["split"].astype(str) == expected_split):
            raise ValueError(f"Unexpected split inside {path}")
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or len(x) != len(ids) or len(x) != len(labels):
        raise ValueError(f"Invalid embedding array shape in {path}: {x.shape}")
    mask = np.isin(labels, LABELS)
    x, ids, labels = x[mask], ids[mask], labels[mask]
    if not np.isfinite(x).all() or len(ids) != len(set(ids)):
        raise ValueError(f"Nonfinite embeddings or repeated UUIDs in {path}")
    if set(labels) != set(LABELS):
        raise ValueError(f"Expected both classes in {path}")
    return x, ids, (labels == "COVID-19").astype(np.int64)


def metadata_train_paths(path: Path):
    """Match original COVID UUIDs to original WAVs; never use non-train rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"uuid", "status", "split", "audio_path"}.issubset(reader.fieldnames or []):
            raise ValueError("Metadata requires uuid,status,split,audio_path")
        mapping = {}
        for row in reader:
            if row["split"].strip() != "train" or row["status"].strip() != "COVID-19":
                continue
            uid = row["uuid"].strip()
            if uid in mapping:
                raise ValueError(f"Duplicate COVID training UUID: {uid}")
            mapping[uid] = Path(row["audio_path"]).expanduser()
    return mapping


def target_covid_count(healthy_count: int, covid_fraction: float):
    if not 0 < covid_fraction < 0.5:
        raise ValueError("--covid-fraction must be between 0 and 0.5")
    return int(round(healthy_count * covid_fraction / (1 - covid_fraction)))


def deterministic_oversampling(y, target, seed):
    """Retain all original rows and sample additional COVID indices with replacement."""
    positives = np.flatnonzero(y == 1)
    if not len(positives) or target < len(positives):
        raise ValueError("Invalid minority oversampling target")
    rng = np.random.default_rng(seed)
    extra = rng.choice(positives, size=target - len(positives), replace=True)
    return np.concatenate([np.arange(len(y)), extra])


def augment_waveform(wave, rng):
    """Gain + low Gaussian noise + noncircular time shift; return audit values."""
    gain = float(rng.uniform(0.9, 1.1))
    x = np.asarray(wave, dtype=np.float32) * gain
    noise_std = max(float(np.max(np.abs(x))) * 0.005, 1e-5)
    x = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)
    max_shift = int(round(len(x) * 0.10))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    shifted = np.zeros_like(x)
    if shift > 0:
        shifted[shift:] = x[:-shift]
    elif shift < 0:
        shifted[:shift] = x[-shift:]
    else:
        shifted[:] = x
    shifted = np.clip(shifted, -1.0, 1.0).astype(np.float32)
    return shifted, {"gain": gain, "noise_std": noise_std, "shift_samples": shift}


def make_clips(wave):
    """Use the ORIGINAL HeAR extraction policy: nonoverlapping 2s, pad final."""
    count = max(1, math.ceil(len(wave) / CLIP_SAMPLES))
    return np.pad(wave, (0, count * CLIP_SAMPLES - len(wave))).reshape(count, CLIP_SAMPLES)


def load_hear():
    """Use the same loading API as the project's original hear_features.py."""
    try:
        from huggingface_hub import from_pretrained_keras
    except ImportError as exc:
        raise RuntimeError("Install compatible huggingface_hub<1.0 and tf_keras for H2") from exc
    model = from_pretrained_keras("google/hear")
    return model.signatures["serving_default"]


def embed_hear(signature, wave):
    clips = make_clips(wave).astype(np.float32)
    # This signature matches the existing original HeAR extraction module.
    result = signature(x=clips)["output_0"].numpy()
    if result.shape != (len(clips), FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError(f"Invalid HeAR output shape {result.shape}")
    return result.mean(axis=0).astype(np.float32)


def create_augmented_embeddings(args, train_ids, y_train, extra_indices, output_dir):
    """Cache exact WAVs and per-recording embeddings with provenance.

    A run with an incomplete augmentation cache is rebuilt safely. Each WAV
    has a deterministic seed and is checked by SHA256 in the manifest. Only
    *training* COVID UUIDs may be augmented.
    """
    import librosa
    import soundfile as sf

    audio_dir = output_dir / "augmented_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    embeddings_file = output_dir / "covid_augmented_train.npz"
    manifest_file = output_dir / "augmentation_manifest.csv"
    config_file = output_dir / "augmentation_config.json"
    metadata = metadata_train_paths(args.metadata)
    planned = []
    # Repeated presentations of a UUID receive distinct, reproducible seeds.
    counts = Counter()
    for idx in extra_indices:
        idx = int(idx)
        if y_train[idx] != 1:
            raise ValueError("Attempt to augment a non-COVID training recording")
        uid = str(train_ids[idx])
        counts[uid] += 1
        k = counts[uid]
        if uid not in metadata:
            raise ValueError(f"COVID UUID absent from training metadata: {uid}")
        planned.append((uid, k, stable_seed(uid, k, args.seed), metadata[uid]))
    config = {
        "experiment": "H2", "source_split": "train", "seed": args.seed,
        "covid_fraction": args.covid_fraction, "augmentation": {
            "gain_range": [0.9, 1.1], "noise_ratio_to_peak": 0.005,
            "shift_max_fraction": 0.10, "clip_to_minus_one_plus_one": True,
        }, "hear": {"repo": "google/hear", "embedding_dim": 512,
                    "sample_rate": SAMPLE_RATE, "clip_samples": CLIP_SAMPLES,
                    "window_aggregation": "mean"},
        "source_train_npz_sha256": sha256(args.train_npz),
        "source_metadata_sha256": sha256(args.metadata),
        "planned_augmented_samples": len(planned),
    }
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError("H2 cache configuration changed; choose a new output directory")
    else:
        write_json(config_file, config)

    # If complete cache exists, verify its source IDs, seeds and audio hashes.
    if embeddings_file.exists() and manifest_file.exists():
        with manifest_file.open(newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        with np.load(embeddings_file, allow_pickle=False) as z:
            x = z["embeddings"].astype(np.float32)
            source = z["source_uuid"].astype(str)
            seeds = z["seed"].astype(np.uint64)
        valid = (len(manifest) == len(planned) == len(x)
                 and x.shape == (len(planned), FEATURE_DIM) and np.isfinite(x).all())
        if valid:
            for i, (uid, k, seed, _) in enumerate(planned):
                m = manifest[i]
                p = output_dir / m["augmented_audio"]
                if (source[i] != uid or int(seeds[i]) != seed or
                    m["source_uuid"] != uid or int(m["seed"]) != seed or
                    int(m["augmentation_index"]) != k or
                    not p.is_file() or sha256(p) != m["audio_sha256"]):
                    valid = False
                    break
        if valid:
            print(f"[H2] Verified and reused {len(x)} cached embeddings", flush=True)
            return x
        raise ValueError("H2 cache is incomplete/inconsistent. Inspect it; use a new output folder")

    signature = load_hear()
    extracted = []
    manifest = []
    # Per-WAV and per-embedding checkpoints enable safe recovery after interruption.
    checkpoint_dir = output_dir / "embedding_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for i, (uid, k, seed, src) in enumerate(planned):
        audio_name = f"aug_{i:05d}.wav"
        wav_path = audio_dir / audio_name
        checkpoint = checkpoint_dir / f"aug_{i:05d}.npy"
        if not src.is_file():
            raise FileNotFoundError(f"Missing source audio {src}")
        # Deterministic regeneration, including audit parameters.
        wave, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError(f"Invalid source audio {src}")
        transformed, audit = augment_waveform(wave, np.random.default_rng(seed))
        if wav_path.exists():
            saved, sr = sf.read(wav_path, dtype="float32")
            if sr != SAMPLE_RATE or saved.shape != transformed.shape or not np.allclose(saved, transformed, atol=4e-5):
                raise ValueError(f"Cached WAV does not match deterministic regeneration: {wav_path}")
        else:
            # FLOAT subtype preserves waveform fidelity (not 16-bit quantization).
            sf.write(wav_path, transformed, SAMPLE_RATE, subtype="FLOAT")
        if checkpoint.exists():
            embedding = np.load(checkpoint, allow_pickle=False)
            if embedding.shape != (FEATURE_DIM,) or not np.isfinite(embedding).all():
                raise ValueError(f"Invalid embedding checkpoint {checkpoint}")
        else:
            # Embed the actual saved WAV, ensuring provenance matches extracted features.
            saved, _ = sf.read(wav_path, dtype="float32")
            embedding = embed_hear(signature, saved)
            np.save(checkpoint, embedding)
        extracted.append(embedding)
        manifest.append({
            "source_uuid": uid, "source_audio": str(src), "source_audio_sha256": sha256(src),
            "augmentation_index": k, "seed": seed,
            "augmented_audio": str(wav_path.relative_to(output_dir)),
            "audio_sha256": sha256(wav_path), "gain": audit["gain"],
            "noise_std": audit["noise_std"], "shift_samples": audit["shift_samples"],
            "embedding_checkpoint": str(checkpoint.relative_to(output_dir)),
        })
        if (i + 1) % 50 == 0 or i + 1 == len(planned):
            print(f"[H2] Embedded {i + 1}/{len(planned)} augmented recordings", flush=True)
    x = np.stack(extracted).astype(np.float32)
    np.savez_compressed(embeddings_file, embeddings=x,
                        source_uuid=np.array([r["source_uuid"] for r in manifest]),
                        seed=np.array([int(r["seed"]) for r in manifest], dtype=np.uint64),
                        label=np.array(["COVID-19"] * len(manifest)))
    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return x


def evaluate(y, scores, threshold):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score
    pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "covid_precision": float(precision_score(y, pred, zero_division=0)),
        "covid_recall": float(recall_score(y, pred, zero_division=0)),
        "covid_f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
    }


def tune_threshold(y, scores):
    """Threshold selection is a *validation-only* model selection procedure."""
    candidates = [evaluate(y, scores, float(t)) for t in np.round(np.arange(0.10, 0.901, 0.05), 2)]
    best = max(candidates, key=lambda row: (row["macro_f1"], row["covid_recall"], row["specificity"]))
    return {"selected": best, "grid": candidates}


def fit_transformers(x, y):
    """Fit scaling and feature ranking ONCE on unaugmented train only."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import f_classif
    scaler = StandardScaler().fit(x)
    normalized = scaler.transform(x)
    f_scores, _ = f_classif(normalized, y)
    f_scores = np.nan_to_num(f_scores, nan=-np.inf, posinf=np.inf)
    rank = np.argsort(-f_scores, kind="stable")
    return scaler, rank, f_scores


def train_classifier(name, k, strategy, x_train, y_train, x_val, y_val, scaler, rank, args, out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    import joblib
    selected = rank[:k].astype(int)
    train = scaler.transform(x_train)[:, selected]
    validation = scaler.transform(x_val)[:, selected]
    if name == "logreg":
        clf = LogisticRegression(max_iter=1500, class_weight="balanced" if strategy == "h0" else None,
                                 random_state=args.seed)
    else:
        # Avoid MLPClassifier's random non-stratified internal early-stop split:
        # use fixed max iterations and external validation for comparison.
        clf = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=0.01,
                            max_iter=150, early_stopping=False, random_state=args.seed)
        # scikit-learn MLPClassifier has no class_weight: H0 applies sample_weight
        # when supported by the installed scikit-learn version.
    fit_args = {}
    if strategy == "h0" and name == "mlp":
        counts = np.bincount(y_train, minlength=2)
        sample_weight = len(y_train) / (2.0 * counts[y_train])
        import inspect
        if "sample_weight" not in inspect.signature(clf.fit).parameters:
            raise RuntimeError("H0 MLP class weighting needs scikit-learn with MLPClassifier.fit(sample_weight). Upgrade scikit-learn.")
        fit_args["sample_weight"] = sample_weight
    start = time.perf_counter()
    clf.fit(train, y_train, **fit_args)
    seconds = time.perf_counter() - start
    scores = clf.predict_proba(validation)[:, 1]
    result = {
        "strategy": strategy, "classifier": name, "feature_count": int(k),
        "selected_feature_indices_zero_based": selected.tolist(),
        "train_presentations": int(len(y_train)), "train_healthy": int((y_train == 0).sum()),
        "train_covid": int((y_train == 1).sum()), "validation_count": int(len(y_val)),
        "fit_seconds": float(seconds), "validation_at_0_5": evaluate(y_val, scores, 0.5),
        "threshold_tuning": tune_threshold(y_val, scores),
    }
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_indices": selected, "classifier": clf}, out / "pipeline.joblib")
    write_json(out / "results.json", result)
    return result


def run(args):
    import joblib
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_train, train_ids, y_train = read_embeddings(args.train_npz, "train")
    x_val, val_ids, y_val = read_embeddings(args.validation_npz, "validation")
    if set(train_ids).intersection(val_ids):
        raise ValueError("Train/validation UUID overlap")
    original_counts = {"healthy": int((y_train == 0).sum()), "COVID-19": int((y_train == 1).sum())}
    target = target_covid_count(original_counts["healthy"], args.covid_fraction)
    selected = deterministic_oversampling(y_train, target, args.seed)
    extras = selected[len(y_train):]
    scaler, rank, f_scores = fit_transformers(x_train, y_train)
    feature_dir = args.output_dir / "feature_selection"
    feature_dir.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "ranking": rank}, feature_dir / "train_only_preprocessing.joblib")
    write_json(feature_dir / "feature_ranking.json", {
        "method": "ANOVA F score", "fitted_on": "original_binary_train_only",
        "ranking_zero_based": rank.tolist(),
        "top_20_f_scores": [float(f_scores[i]) for i in rank[:20]],
        "feature_counts": list(FEATURE_COUNTS),
    })
    summary_file = args.output_dir / "summary.json"
    signature = {
        "train_npz_sha256": sha256(args.train_npz),
        "validation_npz_sha256": sha256(args.validation_npz),
        "metadata_sha256": sha256(args.metadata) if args.metadata.exists() else None,
        "seed": args.seed, "covid_fraction": args.covid_fraction,
        "feature_counts": list(FEATURE_COUNTS), "classifiers": ["logreg", "mlp"],
    }
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        if summary.get("signature") != signature:
            raise ValueError("Run configuration/input changed; choose a new output directory")
    else:
        summary = {"created_utc": timestamp(), "signature": signature,
                   "original_train_counts": original_counts,
                   "validation_counts": {"healthy": int((y_val == 0).sum()), "COVID-19": int((y_val == 1).sum())},
                   "experiments": {}, "test_set_used": False}
        write_json(summary_file, summary)
    for strategy in args.stages:
        print(f"\n=== {strategy.upper()} ===", flush=True)
        if strategy == "h0":
            xx, yy = x_train, y_train
        elif strategy == "h1":
            xx, yy = x_train[selected], y_train[selected]
        else:
            h2_dir = args.output_dir / "H2_augmentation"
            aug_x = create_augmented_embeddings(args, train_ids, y_train, extras, h2_dir)
            xx = np.concatenate([x_train, aug_x], axis=0)
            yy = np.concatenate([y_train, np.ones(len(aug_x), dtype=np.int64)])
        for k in FEATURE_COUNTS:
            for classifier in ("logreg", "mlp"):
                key = f"{strategy}_{classifier}_{k}"
                path = args.output_dir / strategy.upper() / f"{classifier}_{k}"
                if (path / "results.json").exists() and (path / "pipeline.joblib").exists():
                    result = json.loads((path / "results.json").read_text())
                    print(f"[SKIP] {key}", flush=True)
                else:
                    try:
                        result = train_classifier(classifier, k, strategy, xx, yy, x_val, y_val,
                                                  scaler, rank, args, path)
                        print(f"[OK] {key}: val macro F1={result['threshold_tuning']['selected']['macro_f1']:.4f}", flush=True)
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                        print(f"[FAILED] {key}: {exc}", flush=True)
                summary["experiments"][key] = result
                summary["updated_utc"] = timestamp()
                write_json(summary_file, summary)
    # Compact CSV for report drafting, with one row per completed configuration.
    rows = []
    for key, value in summary["experiments"].items():
        if "threshold_tuning" not in value:
            continue
        m = value["threshold_tuning"]["selected"]
        rows.append({"experiment": key, "strategy": value["strategy"],
                     "classifier": value["classifier"], "features": value["feature_count"],
                     "validation_macro_f1": m["macro_f1"], "covid_recall": m["covid_recall"],
                     "covid_precision": m["covid_precision"], "specificity": m["specificity"],
                     "threshold": m["threshold"], "validation_accuracy": m["accuracy"]})
    if rows:
        csv_path = args.output_dir / "experiment_comparison.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved: {summary_file}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-npz", type=Path, default=Path("data/coughvid_hear_embeddings/hear_train.npz"))
    parser.add_argument("--validation-npz", type=Path, default=Path("data/coughvid_hear_embeddings/hear_validation.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/coughvid_20211012_processed/metadata_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/coughvid_hear_experiments"))
    parser.add_argument("--stages", nargs="+", choices=("h0", "h1", "h2"), default=["h0", "h1", "h2"])
    parser.add_argument("--covid-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
