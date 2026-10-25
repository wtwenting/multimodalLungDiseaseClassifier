#!/usr/bin/env python3
"""CT-only DenseNet121 baseline using V3 manifest. Never loads test CTs.

Known visually suspicious selected scans are excluded from TRAIN by default;
validation suspects are not silently removed. Override via a reviewed CSV.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

KNOWN_RISK = {'S03048', 'S03074', 'S04583', 'S05295', 'S03834', 'S03259'}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reviewed-overrides', type=Path, help='CSV with patient_id,ct_path; manually approved replacements, training only')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--block-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--check-only', action='store_true')
    return p.parse_args()


def read_rows(args):
    with args.manifest.open(newline='', encoding='utf-8-sig') as f:
        raw = list(csv.DictReader(f))
    overrides = {}
    if args.reviewed_overrides:
        with args.reviewed_overrides.open(newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                pid = r['patient_id'].strip()
                if pid in overrides:
                    raise ValueError(f'Duplicate override: {pid}')
                overrides[pid] = Path(r['ct_path'])
    groups = {'train': [], 'val': []}
    seen = set()
    excluded = []
    splits = {'train': set(), 'val': set(), 'test': set(), 'not_in_xray_split': set()}
    for r in raw:
        pid = r['patient_id'].strip()
        split = r['split'].strip().lower()
        if pid in seen:
            raise ValueError(f'Duplicate patient: {pid}')
        seen.add(pid)
        if split not in splits:
            raise ValueError(f'Unknown split {split!r} for {pid}')
        splits[split].add(pid)
        if split not in groups:
            continue  # Never load test or unassigned volumes.
        status = r['selection_status'].strip()
        if status == 'NO_CANDIDATE_CT':
            excluded.append({'patient_id': pid, 'reason': 'NO_CANDIDATE_CT'})
            continue
        if status not in ('ADEQUATE_COVERAGE', 'LOW_COVERAGE_REVIEW'):
            raise ValueError(f'Unexpected selection status {status}: {pid}')
        if r['geometry_flag'].strip() != 'HEADER_PASS':
            raise ValueError(f'Non-passing geometry for selected CT: {pid}')
        label = {'covidNegative': 0, 'covidPositive': 1}.get(r['class_name'].strip())
        if label is None:
            raise ValueError(f'Unknown label for {pid}')
        path = Path(r['ct_path'])
        if pid in overrides:
            if split != 'train':
                raise ValueError('Overrides only supported for train patients')
            path = overrides[pid]
        elif split == 'train' and pid in KNOWN_RISK:
            excluded.append({'patient_id': pid, 'reason': 'KNOWN_VISUAL_RISK_UNRESOLVED'})
            continue
        if not path.is_file():
            raise FileNotFoundError(f'{pid}: {path}')
        groups[split].append({'patient_id': pid, 'path': path, 'label': label})
    if set(overrides) - splits['train']:
        raise ValueError('Override contains patient not in training split')
    if not groups['train'] or not groups['val']:
        raise ValueError('Empty train/validation cohort')
    if set(x['patient_id'] for x in groups['train']) & set(x['patient_id'] for x in groups['val']):
        raise ValueError('Patient leakage')
    return groups, excluded, {k: len(v) for k, v in splits.items()}


def dataset_for(records, augment=False):
    import nibabel as nib
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Dataset

    class CTDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, i):
            row = records[i]
            nii = nib.as_closest_canonical(nib.load(str(row['path'])))
            if len(nii.shape) != 3 or min(nii.shape) < 32:
                raise ValueError(f'Invalid CT shape: {row["path"]}')
            arr = np.asarray(nii.dataobj, dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError(f'Nonfinite intensity: {row["path"]}')
            arr = np.transpose(arr, (2, 1, 0))
            lo, hi = np.percentile(arr, [1, 99])
            if hi <= lo:
                raise ValueError(f'Constant/invalid intensity: {row["path"]}')
            arr = np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)
            x = torch.from_numpy(np.ascontiguousarray(arr))[None, None]
            x = F.interpolate(x, size=(64, 128, 128), mode='trilinear', align_corners=False)[0]
            if augment and random.random() < 0.5:
                x = torch.flip(x, dims=[3])
            return x, torch.tensor(row['label'], dtype=torch.long)

    return CTDataset()


def main():
    a = args_parser()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from monai.networks.nets import DenseNet121
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; use --device cpu')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cohorts, excluded, split_counts = read_rows(a)
    report = {
        'source_manifest': str(a.manifest), 'split_patient_counts': split_counts,
        'train_count': len(cohorts['train']), 'val_count': len(cohorts['val']),
        'train_positive': sum(r['label'] for r in cohorts['train']),
        'val_positive': sum(r['label'] for r in cohorts['val']),
        'excluded_from_training': excluded,
        'note': 'Historical preprocessing baseline; test not evaluated. Unreviewed risk patients omitted from training only.'
    }
    print(json.dumps(report, indent=2), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'cohort_report.json').write_text(json.dumps(report, indent=2))
    if a.check_only:
        return
    if not a.checkpoint.is_file():
        raise FileNotFoundError(a.checkpoint)
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.features.denseblock4.requires_grad_(True)
    model.class_layers.requires_grad_(True)
    device = torch.device(a.device)
    model.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.features.denseblock4.parameters(), 'lr': a.block_lr},
        {'params': model.class_layers.parameters(), 'lr': a.head_lr},
    ])
    criterion = nn.CrossEntropyLoss()
    loaders = {split: DataLoader(dataset_for(rows, augment=(split == 'train')),
                                 batch_size=a.batch_size, shuffle=(split == 'train'),
                                 num_workers=a.workers)
               for split, rows in cohorts.items()}

    def set_train_mode():
        model.train()
        # Keep all frozen batchnorm layers from updating their running statistics.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    if a.smoke:
        set_train_mode()
        x, y = next(iter(loaders['train']))
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        print('SMOKE PASS:', tuple(x.shape), 'loss=', loss.detach().item(), flush=True)
        return

    history = []
    best, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, a.epochs + 1):
        start = time.monotonic()
        set_train_mode()
        total_loss, total_n = 0., 0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_n += len(y)
        model.eval()
        labels, probabilities = [], []
        with torch.inference_mode():
            for x, y in loaders['val']:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
                probabilities.extend(p.cpu().tolist())
                labels.extend(y.tolist())
        pred = [int(p >= .5) for p in probabilities]
        metrics = {
            'accuracy': float(accuracy_score(labels, pred)),
            'macro_f1': float(f1_score(labels, pred, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
            'covid_recall': float(recall_score(labels, pred, pos_label=1, zero_division=0)),
            'noncovid_recall': float(recall_score(labels, pred, pos_label=0, zero_division=0)),
            'covid_precision': float(precision_score(labels, pred, pos_label=1, zero_division=0)),
            'roc_auc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(labels, pred, labels=[0, 1]).ravel().tolist(),
        }
        record = {'epoch': epoch, 'train_loss': total_loss / total_n,
                  'validation': metrics, 'seconds': round(time.monotonic() - start, 1)}
        history.append(record)
        print(f'Epoch {epoch}: loss={record["train_loss"]:.4f} val_f1={metrics["macro_f1"]:.4f} val_auc={metrics["roc_auc"]}', flush=True)
        (a.output / 'history.json').write_text(json.dumps(history, indent=2))
        if metrics['macro_f1'] > best + .001:
            best, best_epoch, stale = metrics['macro_f1'], epoch, 0
            torch.save(model.state_dict(), a.output / 'ct_densenet121_best.pt')
        else:
            stale += 1
        if stale >= a.patience:
            break
    (a.output / 'summary.json').write_text(json.dumps({
        'best_epoch': best_epoch, 'best_validation_macro_f1': best,
        'best_validation_metrics': history[best_epoch - 1]['validation'],
        'epochs_run': len(history), 'test': 'NOT_EVALUATED',
        'checkpoint': str(a.output / 'ct_densenet121_best.pt'),
    }, indent=2))
    print('DONE; test set not evaluated', flush=True)


if __name__ == '__main__':
    main()
