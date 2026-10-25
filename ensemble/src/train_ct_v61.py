
"""
BIMCV V6.1 - Standalone adaptive CT encoder training.

Inputs:
  ct_v5_corrected_preprocessed/patient_manifest.csv
  ct_v5_corrected_preprocessed/tensors/
  ct_training_v5_pilot_output/patient_splits.json

Outputs:
  ct_training_v61_output/

No dependency on previous training scripts.
No test-set evaluation during model selection.
"""

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models


# ============================================================
# 1. Paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ct_v5_corrected_preprocessed"
MANIFEST = DATA / "patient_manifest.csv"
SPLITS = ROOT / "ct_training_v5_pilot_output" / "patient_splits.json"

OUT = ROOT / "ct_training_v61_output"
CKPT = OUT / "checkpoints"
PRED = OUT / "predictions"

for folder in (OUT, CKPT, PRED):
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_DISK_GB = 8
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUT / "progress.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.info


# ============================================================
# 2. Utilities and recovery
# ============================================================

def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_pt(path, data):
    path = Path(path)

    if shutil.disk_usage(path.parent).free < MIN_DISK_GB * 1024**3:
        raise RuntimeError("DISK_LOW: insufficient checkpoint space")

    tmp = path.with_suffix(".tmp")
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else []
    }


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())

    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def config_id(cfg):
    text = json.dumps(cfg, sort_keys=True)
    return "v61_" + hashlib.sha256(text.encode()).hexdigest()[:12]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ============================================================
# 3. V5 dataset and split validation
# ============================================================

def load_records():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {"patient", "label", "status", "tensor_path"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("Invalid V5 manifest schema")

    records = {}

    for row in rows:
        if row["status"].upper() != "PASS":
            continue

        patient = row["patient"].strip().lower()
        label = row["label"].strip().lower()

        if label not in ("positive", "negative"):
            raise RuntimeError(f"Invalid label: {patient}")

        path = Path(row["tensor_path"])

        if not path.is_absolute():
            candidates = [ROOT / path, DATA / path]
            path = next((p for p in candidates if p.exists()), candidates[0])

        if not path.exists():
            raise FileNotFoundError(f"{patient}: {path}")

        if patient in records:
            raise RuntimeError(f"Duplicate patient: {patient}")

        records[patient] = {
            "path": str(path),
            "label": int(label == "positive")
        }

    if len(records) != 958:
        raise RuntimeError(
            f"Expected 958 V5 PASS patients; found {len(records)}"
        )

    return records


def load_splits(records):
    if not SPLITS.exists():
        raise FileNotFoundError(SPLITS)

    raw = read_json(SPLITS)
    groups = {}

    for key in ("train", "val", "test"):
        if key not in raw:
            raise RuntimeError(f"Missing split: {key}")

        groups[key] = [str(p).lower() for p in raw[key]]

        if len(groups[key]) != len(set(groups[key])):
            raise RuntimeError(f"Duplicate IDs in {key}")

    sets = [set(groups[k]) for k in ("train", "val", "test")]

    for i in range(3):
        for j in range(i + 1, 3):
            if sets[i] & sets[j]:
                raise RuntimeError("Patient leakage between splits")

    if set.union(*sets) != set(records):
        raise RuntimeError("Frozen splits differ from V5 manifest")

    for key, ids in groups.items():
        counts = [
            sum(records[p]["label"] == k for p in ids)
            for k in (0, 1)
        ]
        log("%s: %d patients, negative=%d positive=%d",
            key, len(ids), *counts)

    return groups


def read_tensor(path):
    path = Path(path)

    if path.suffix == ".npz":
        with np.load(path) as f:
            key = "tensor" if "tensor" in f.files else f.files[0]
            arr = f[key].astype(np.float32)

    elif path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)

    elif path.suffix in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            obj = obj.get("tensor", obj.get("volume"))
        if not isinstance(obj, torch.Tensor):
            raise RuntimeError(f"Unsupported tensor: {path}")
        arr = obj.numpy().astype(np.float32)

    else:
        raise RuntimeError(f"Unsupported format: {path}")

    arr = np.squeeze(arr)

    if arr.shape != (64, 192, 192):
        raise RuntimeError(f"Invalid tensor shape: {path}: {arr.shape}")

    if not np.isfinite(arr).all():
        raise RuntimeError(f"Nonfinite tensor: {path}")

    if arr.min() < -0.01 or arr.max() > 1.01:
        raise RuntimeError(f"Invalid normalization: {path}")

    return torch.from_numpy(np.ascontiguousarray(arr))


class CTDataset(Dataset):
    def __init__(self, records, patients, views, augment=False):
        self.records = records
        self.patients = patients
        self.views = views
        self.augment = augment

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient = self.patients[index]
        item = self.records[patient]

        x = read_tensor(item["path"])

        if self.augment:
            if random.random() < 0.5:
                x = x.flip(-1)

            if random.random() < 0.35:
                scale = random.uniform(0.95, 1.05)
                shift = random.uniform(-0.025, 0.025)
                x = (x * scale + shift).clamp(0, 1)

        centers = torch.linspace(3, 60, self.views).round().long()

        # Three neighboring axial slices per view.
        x = torch.stack([x[c-1:c+2] for c in centers])

        return x, item["label"], patient


def make_loaders(records, splits, cfg):
    train = CTDataset(
        records, splits["train"], cfg["views"], augment=True
    )
    val = CTDataset(
        records, splits["val"], cfg["views"], augment=False
    )

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=cfg["batch"],
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )

    return loader(train, True), loader(val, False)


# ============================================================
# 4. Independent CNN architectures
# ============================================================

class CTEncoder(nn.Module):
    def __init__(self, architecture, pooling, dropout):
        super().__init__()
        self.pooling = pooling

        if architecture == "densenet121":
            backbone = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT
            )
            dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()

        elif architecture == "convnext_tiny":
            backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT
            )
            dim = backbone.classifier[2].in_features
            backbone.classifier[2] = nn.Identity()

        else:
            raise ValueError(architecture)

        self.backbone = backbone
        self.feature_dim = dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )

    def forward(self, x):
        batch, views, channels, height, width = x.shape

        features = self.backbone(
            x.reshape(batch * views, channels, height, width)
        )
        features = features.reshape(batch, views, -1)

        if self.pooling == "mean":
            pooled = features.mean(dim=1)

        elif self.pooling == "topk":
            scores = self.head(features).softmax(-1)[..., 1]
            indices = scores.topk(
                k=min(3, views), dim=1
            ).indices

            selected = features.gather(
                1,
                indices[..., None].expand(
                    -1, -1, features.shape[-1]
                )
            )
            pooled = selected.mean(dim=1)

        else:
            raise ValueError(self.pooling)

        return self.head(pooled)


def freeze_backbone(model, freeze):
    for p in model.backbone.parameters():
        p.requires_grad_(not freeze)


# ============================================================
# 5. Evaluation
# ============================================================

def evaluate(model, loader):
    model.eval()
    labels, probabilities, patients = [], [], []

    with torch.inference_mode():
        for x, y, ids in loader:
            x = x.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda"
            ):
                logits = model(x)

            probs = logits.float().softmax(-1)[:, 1].cpu()

            labels.extend(y.tolist())
            probabilities.extend(probs.tolist())
            patients.extend(ids)

    y = np.asarray(labels)
    p = np.asarray(probabilities)
    predicted = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()

    result = {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

    predictions = [
        {"patient": pid, "label": int(label), "probability": float(prob)}
        for pid, label, prob in zip(patients, labels, probabilities)
    ]

    return result, predictions


# ============================================================
# 6. Experiment training and recovery
# ============================================================

def train_trial(cfg, records, splits, deadline, board):
    name = cfg["id"]
    best_path = CKPT / f"{name}_best.pt"
    latest_path = CKPT / f"{name}_latest.pt"

    if board.get(name, {}).get("status") == "COMPLETE":
        log("SKIP completed %s", name)
        return

    seed_all(cfg["seed"])

    model = CTEncoder(
        cfg["arch"], cfg["pooling"], cfg["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {
            "params": model.backbone.parameters(),
            "lr": cfg["backbone_lr"]
        },
        {
            "params": model.head.parameters(),
            "lr": cfg["head_lr"]
        }
    ], weight_decay=cfg["wd"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=DEVICE.type == "cuda"
    )

    train_loader, val_loader = make_loaders(records, splits, cfg)

    start_epoch = 0
    best_auc = -1.0
    best_epoch = 0
    stale = 0

    if latest_path.exists():
        state = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )

        if state["config"] != cfg:
            raise RuntimeError("Checkpoint configuration mismatch")

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])

        start_epoch = state["epoch"]
        best_auc = state["best_auc"]
        best_epoch = state["best_epoch"]
        stale = state["stale"]

        if "rng" in state:
            set_rng(state["rng"])

        log("RESUME %s epoch=%d", name, start_epoch)

    status = "PAUSED"

    for epoch in range(start_epoch, cfg["epochs"]):
        if time.monotonic() >= deadline:
            break

        frozen = epoch < cfg["freeze_epochs"]
        freeze_backbone(model, frozen)

        model.train()
        if frozen:
            model.backbone.eval()

        losses = []

        for x, y, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda"
            ):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float(),
                    y,
                    label_smoothing=cfg["smoothing"]
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))

        result, predictions = evaluate(model, val_loader)
        auc = result["auc"]

        if not frozen:
            scheduler.step(auc)

        if auc > best_auc + 0.0001:
            best_auc = auc
            best_epoch = epoch + 1
            stale = 0

            save_pt(best_path, {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": best_epoch,
                "validation": result,
                "feature_dim": model.feature_dim,
                "preprocessing": "V5 corrected",
                "split_file": str(SPLITS)
            })

            save_json(
                PRED / f"{name}_validation.json",
                predictions
            )
        else:
            stale += 1

        save_pt(latest_path, {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": get_rng(),
            "config": cfg,
            "epoch": epoch + 1,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "stale": stale
        })

        history = {
            "id": name,
            "epoch": epoch + 1,
            "frozen": frozen,
            "train_loss": float(np.mean(losses)),
            "validation": result,
            "best_auc": best_auc,
            "learning_rates": [
                group["lr"] for group in optimizer.param_groups
            ],
            "config": cfg
        }

        with (OUT / "epoch_history.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(history) + "\n")
            f.flush()

        log(
            "%s epoch=%d/%d frozen=%s loss=%.4f "
            "auc=%.4f acc=%.4f f1=%.4f best=%.4f",
            name, epoch + 1, cfg["epochs"], frozen,
            history["train_loss"], auc,
            result["accuracy"], result["f1"], best_auc
        )

        if stale >= cfg["patience"]:
            status = "COMPLETE"
            log("EARLY STOP %s", name)
            break
    else:
        status = "COMPLETE"

    best_metrics = {}
    if best_path.exists():
        state = torch.load(
            best_path, map_location="cpu", weights_only=False
        )
        best_metrics = state["validation"]

    board[name] = {
        "status": status,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "validation": best_metrics,
        "config": cfg,
        "checkpoint": str(best_path)
    }

    save_json(OUT / "leaderboard.json", board)

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 7. Adaptive experiment search
# ============================================================

def make_config(
    arch, pooling="mean", lr=3e-5,
    wd=1e-4, dropout=0.2, views=8,
    batch=2, freeze=3, smoothing=0.05, seed=42
):
    cfg = {
        "arch": arch,
        "pooling": pooling,
        "backbone_lr": lr,
        "head_lr": 1e-4,
        "wd": wd,
        "dropout": dropout,
        "views": views,
        "batch": batch,
        "freeze_epochs": freeze,
        "smoothing": smoothing,
        "seed": seed,
        "epochs": 25,
        "patience": 6
    }

    cfg["id"] = config_id(cfg)
    return cfg


def screening():
    configs = []

    # Controlled 8-experiment screening.
    for arch in ("densenet121", "convnext_tiny"):
        for freeze in (0, 3):
            for views in (8, 12):
                configs.append(make_config(
                    arch=arch,
                    pooling="mean",
                    lr=3e-5,
                    views=views,
                    freeze=freeze
                ))

    return configs


def adaptive_config(board, rng):
    completed = [
        entry for entry in board.values()
        if entry.get("status") == "COMPLETE"
        and entry.get("best_auc", -1) > 0
    ]

    completed.sort(
        key=lambda x: x["best_auc"],
        reverse=True
    )

    best_arch = (
        completed[0]["config"]["arch"]
        if completed else "densenet121"
    )

    other_arch = (
        "convnext_tiny"
        if best_arch == "densenet121"
        else "densenet121"
    )

    arch = (
        best_arch if rng.random() < 0.75
        else other_arch
    )

    return make_config(
        arch=arch,
        pooling=rng.choice(["mean", "mean", "topk"]),
        lr=rng.choice([1e-5, 2e-5, 3e-5, 5e-5]),
        wd=rng.choice([1e-5, 1e-4, 5e-4]),
        dropout=rng.choice([0.2, 0.3, 0.4]),
        views=rng.choice([8, 12]),
        batch=rng.choice([2, 4]),
        freeze=rng.choice([0, 2, 3, 5]),
        smoothing=rng.choice([0.0, 0.05, 0.1]),
        seed=rng.choice([42, 123, 777])
    )


# ============================================================
# 8. Main execution
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=5.5)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU unavailable")

    seed_all(SEED)

    log("V6.1 START device=%s", DEVICE)

    records = load_records()
    splits = load_splits(records)

    # Check actual V5 tensor and both CNN forward/backward passes.
    for arch in ("densenet121", "convnext_tiny"):
        cfg = make_config(arch)
        loader, _ = make_loaders(records, splits, cfg)
        x, y, _ = next(iter(loader))

        model = CTEncoder(
            cfg["arch"], cfg["pooling"], cfg["dropout"]
        ).to(DEVICE)

        with torch.autocast("cuda"):
            logits = model(x.to(DEVICE))
            loss = F.cross_entropy(
                logits.float(), y.to(DEVICE)
            )

        loss.backward()

        log(
            "PREFLIGHT PASS %s shape=%s loss=%.4f",
            arch, tuple(x.shape), float(loss.detach().cpu())
        )

        del model, loader, x, y, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

    if args.preflight:
        log("PREFLIGHT COMPLETE - NO TRAINING")
        return

    start = time.monotonic()
    deadline = start + args.hours * 3600

    board_path = OUT / "leaderboard.json"
    board = read_json(board_path) if board_path.exists() else {}

    queue = screening()
    rng = random.Random(61061)

    failures = 0
    completed_this_run = 0

    while time.monotonic() < deadline:
        if queue:
            cfg = queue.pop(0)
        else:
            cfg = None

            for _ in range(200):
                candidate = adaptive_config(board, rng)
                if (
                    candidate["id"] not in board
                    or board[candidate["id"]]["status"] == "PAUSED"
                ):
                    cfg = candidate
                    break

            if cfg is None:
                log("No additional configurations available")
                break

        if board.get(cfg["id"], {}).get("status") == "COMPLETE":
            continue

        log("START %s config=%s", cfg["id"], cfg)

        try:
            train_trial(cfg, records, splits, deadline, board)
            failures = 0
            completed_this_run += 1

        except Exception as exc:
            failures += 1
            log(
                "FAILED %s\n%s",
                cfg["id"], traceback.format_exc()
            )

            board[cfg["id"]] = {
                "status": "FAILED",
                "config": cfg,
                "error": str(exc)
            }
            save_json(board_path, board)

            if (
                failures >= 3
                or "DISK_LOW" in str(exc)
                or "out of memory" in str(exc).lower()
            ):
                log("STOP: resource failure or repeated errors")
                break

    log(
        "V6.1 FINISHED elapsed=%.2f hours experiments=%d",
        (time.monotonic() - start) / 3600,
        completed_this_run
    )
    log("HELD-OUT TEST SET NOT EVALUATED")


if __name__ == "__main__":
    main()
