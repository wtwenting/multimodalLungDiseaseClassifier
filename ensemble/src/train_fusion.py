#!/usr/bin/env python3
"""BIMCV paired X-ray + CT: smoke test and baseline training.

Patient-level manifest, one selected X-ray and CT per patient. Does not modify source files.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='dataAnalysis/output/paired_patient_manifest.csv')
    p.add_argument('--output', default='ensemble/output')
    p.add_argument('--mode', choices=['smoke', 'train'], default='smoke')
    p.add_argument('--model', choices=['fusion', 'xray', 'ct'], default='fusion')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--ct-shape', type=int, nargs=3, default=[48, 96, 96], metavar=('D', 'H', 'W'))
    p.add_argument('--xray-size', type=int, default=192)
    p.add_argument('--ct-intensity', choices=['percentile', 'hu'], default='percentile',
                   help='Use hu ONLY if CT voxel values are confirmed Hounsfield units')
    p.add_argument('--rebuild-splits', action='store_true', help='Rebuild split when manifest changes; invalidates prior comparison')
    return p.parse_args()


def load_rows(manifest):
    required = {'patient_id', 'covid_status', 'xray_path', 'ct_path'}
    with open(manifest, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {required - set(reader.fieldnames or [])}')
        rows = list(reader)
    eligible, missing, seen = [], 0, set()
    for r in rows:
        status = r['covid_status'].strip().lower()
        if status not in ('positive', 'negative'):
            continue
        pid = r['patient_id'].strip().removeprefix('sub-')
        key = (status, pid)
        if key in seen:
            raise ValueError(f'Duplicate patient row: {key}')
        seen.add(key)
        if not (Path(r['xray_path']).is_file() and Path(r['ct_path']).is_file()):
            missing += 1
            continue
        eligible.append({'patient_id': pid, 'status': status,
                         'label': int(status == 'positive'),
                         'xray_path': r['xray_path'], 'ct_path': r['ct_path']})
    if missing:
        print(f'WARNING: {missing} patients skipped: one or both files missing')
    if not eligible:
        raise ValueError('No locally available pairs in manifest')
    return eligible


def splits_for(rows, path, seed, rebuild):
    ids = {f"{r['status']}:{r['patient_id']}" for r in rows}
    if path.exists() and not rebuild:
        data = json.loads(path.read_text())
        old = set().union(*(set(v) for v in data.values()))
        if ids != old:
            raise ValueError('Available patients changed since split was saved. Use --rebuild-splits for a NEW experiment, or keep the same cohort.')
        return data
    rng = random.Random(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for status in ('negative', 'positive'):
        group = [f"{r['status']}:{r['patient_id']}" for r in rows if r['status'] == status]
        rng.shuffle(group)
        if len(group) < 3:
            raise ValueError(f'Need at least 3 patients in {status} class for stratified splits')
        n_test = max(1, round(len(group) * .15))
        n_val = max(1, round(len(group) * .15))
        if n_test + n_val >= len(group):
            raise ValueError(f'Insufficient {status} patients for train/val/test')
        splits['test'] += group[:n_test]
        splits['val'] += group[n_test:n_test+n_val]
        splits['train'] += group[n_test+n_val:]
    path.write_text(json.dumps(splits, indent=2))
    return splits


def make_dataset(rows, xray_size, ct_shape, ct_intensity):
    import torch
    import nibabel as nib
    from PIL import Image
    from scipy.ndimage import zoom
    from torch.utils.data import Dataset

    class PairedDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            with Image.open(r['xray_path']) as img:
                img = img.convert('L').resize((xray_size, xray_size))
                x = np.asarray(img, dtype=np.float32) / 255.0
            x = (x - .5) / .5
            nii = nib.as_closest_canonical(nib.load(r['ct_path']))
            vol = np.asarray(nii.dataobj, dtype=np.float32)
            if vol.ndim != 3 or min(vol.shape) < 2:
                raise ValueError(f'Invalid 3D CT shape {vol.shape}: {r["ct_path"]}')
            # Canonical volume axes (X,Y,Z) -> (Z,Y,X) for PyTorch Conv3D.
            vol = np.transpose(vol, (2, 1, 0))
            if ct_intensity == 'hu':
                vol = np.clip(vol, -1000, 400)
                vol = (vol + 1000) / 1400
            else:
                lo, hi = np.percentile(vol, [1, 99])
                vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
            scale = tuple(t / s for t, s in zip(ct_shape, vol.shape))
            vol = zoom(vol, scale, order=1, prefilter=False)
            # Avoid rounding mismatch in scipy zoom.
            fixed = np.zeros(ct_shape, dtype=np.float32)
            common = tuple(slice(0, min(a, b)) for a, b in zip(ct_shape, vol.shape))
            fixed[common] = vol[common]
            vol = fixed * 2 - 1
            return (torch.from_numpy(x[None].copy()),
                    torch.from_numpy(vol[None].copy()),
                    torch.tensor(float(r['label']), dtype=torch.float32))

    return PairedDataset()


def make_model(mode):
    import torch
    from torch import nn

    class Encoder2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU())
        def forward(self, x): return self.net(x)

    class Encoder3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 8, 3, 2, 1), nn.BatchNorm3d(8), nn.ReLU(),
                nn.Conv3d(8, 16, 3, 2, 1), nn.BatchNorm3d(16), nn.ReLU(),
                nn.Conv3d(16, 32, 3, 2, 1), nn.BatchNorm3d(32), nn.ReLU(),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(32, 64), nn.ReLU())
        def forward(self, x): return self.net(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.xray = Encoder2D() if mode != 'ct' else None
            self.ct = Encoder3D() if mode != 'xray' else None
            n = 128 if mode == 'fusion' else 64
            self.head = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Dropout(.25), nn.Linear(64, 1))
        def forward(self, xray, ct):
            features = []
            if self.xray is not None: features.append(self.xray(xray))
            if self.ct is not None: features.append(self.ct(ct))
            return self.head(torch.cat(features, dim=1)).squeeze(1)

    return Model()


def evaluate(model, loader, device):
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
    model.eval()
    truth, probs = [], []
    with torch.no_grad():
        for x, ct, y in loader:
            p = torch.sigmoid(model(x.to(device), ct.to(device))).cpu().numpy()
            truth.extend(y.numpy().astype(int).tolist())
            probs.extend(p.tolist())
    pred = [int(p >= .5) for p in probs]
    metrics = {'accuracy': accuracy_score(truth, pred),
               'balanced_accuracy': balanced_accuracy_score(truth, pred),
               'macro_f1': f1_score(truth, pred, average='macro', zero_division=0),
               'covid_recall': recall_score(truth, pred, pos_label=1, zero_division=0),
               'noncovid_recall': recall_score(truth, pred, pos_label=0, zero_division=0),
               'covid_precision': precision_score(truth, pred, pos_label=1, zero_division=0)}
    metrics['roc_auc'] = roc_auc_score(truth, probs) if len(set(truth)) == 2 else None
    return {k: None if v is None else float(v) for k, v in metrics.items()}


def main():
    args = parse_args()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from sklearn.utils.class_weight import compute_class_weight
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest)
    print('Available patient pairs:', len(rows), 'positive:', sum(r['label'] for r in rows),
          'negative:', sum(1-r['label'] for r in rows))
    if args.mode == 'smoke':
        subset = rows[:2]
        ds = make_dataset(subset, args.xray_size, args.ct_shape, args.ct_intensity)
        x, ct, y = next(iter(DataLoader(ds, batch_size=2)))
        model = make_model(args.model)
        logits = model(x, ct)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()
        print('SMOKE OK:', 'xray', tuple(x.shape), 'ct', tuple(ct.shape),
              'logits', tuple(logits.shape), 'loss', round(loss.item(), 4))
        return
    split_path = output / 'patient_splits.json'
    splits = splits_for(rows, split_path, args.seed, args.rebuild_splits)
    by_id = {f"{r['status']}:{r['patient_id']}": r for r in rows}
    loaders = {}
    for split, keys in splits.items():
        selected = [by_id[k] for k in keys]
        print(split, len(selected), 'positive:', sum(r['label'] for r in selected))
        ds = make_dataset(selected, args.xray_size, args.ct_shape, args.ct_intensity)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == 'train'),
                                    num_workers=args.workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:', device)
    model = make_model(args.model).to(device)
    labels = np.array([by_id[k]['label'] for k in splits['train']])
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_epoch = -1, 0
    checkpoint = output / f'{args.model}_best.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, ct, y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device), ct.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val = evaluate(model, loaders['val'], device)
        print(f'Epoch {epoch}: loss={total/len(splits["train"]):.4f} val_macro_f1={val["macro_f1"]:.4f}')
        if val['macro_f1'] > best:
            best, best_epoch = val['macro_f1'], epoch
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    result = {'model': args.model, 'best_epoch': best_epoch,
              'validation': evaluate(model, loaders['val'], device),
              'test': evaluate(model, loaders['test'], device),
              'note': 'Preliminary experiment. Date matching and CT series selection not verified.'}
    result_path = output / f'{args.model}_metrics.json'
    result_path.write_text(json.dumps(result, indent=2))
    print('Saved:', checkpoint, result_path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
