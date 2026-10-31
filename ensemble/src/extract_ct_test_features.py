#!/usr/bin/env python3
"""Frozen CT V6.1 mean-pooling encoder: test features only, no test scoring.
Run from ensemble/. Requires ct_v61_frozen_baseline.py in the same src folder.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ct_v61_frozen_baseline import load_checkpoint, make_views, resolve_tensor

CHECKPOINT = 'v61_13a378082055_best.pt'
QUARANTINE = {'S03074', 'S07090', 'S10766', 'S312604'}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('output/ct_v5_encoder_audit/ct_v5_encoder_eligible_manifest.csv'))
    p.add_argument('--checkpoint', type=Path, default=Path('models/ct_v61') / CHECKPOINT)
    p.add_argument('--out', type=Path, default=Path('output/ct_test_features'))
    p.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    p.add_argument('--smoke', action='store_true', help='Extract one test patient into a separate smoke output')
    args = p.parse_args()
    device = torch.device('mps' if args.device == 'auto' and torch.backends.mps.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS not available')
    with args.manifest.open(newline='', encoding='utf-8-sig') as f:
        data = list(csv.DictReader(f))
    if not data or not {'patient_id', 'split', 'tensor_path'}.issubset(data[0]):
        raise ValueError('Manifest must contain patient_id, split, tensor_path')
    selected = []
    all_ids = {}
    for r in data:
        pid = r['patient_id'].strip().upper()
        split = r['split'].strip().lower()
        if pid in all_ids:
            raise ValueError(f'Duplicate patient ID: {pid}')
        all_ids[pid] = split
        if pid in QUARANTINE:
            raise ValueError(f'Quarantined patient unexpectedly in eligible manifest: {pid}')
        if split == 'test':
            if r.get('eligibility_status', 'ELIGIBLE').strip().upper() not in ('ELIGIBLE', 'PASS', 'APPROVED'):
                raise ValueError(f'Test patient not eligible: {pid}')
            selected.append((pid, r['tensor_path']))
    selected.sort()
    if not selected:
        raise ValueError('No eligible test patients in manifest')
    if args.smoke:
        selected = selected[:1]
    model, cfg = load_checkpoint(args.checkpoint, device)
    if cfg['pooling'] != 'mean' or cfg['views'] != 12:
        raise ValueError('Wrong checkpoint: expected 12-view mean pooling')
    features = []
    index = []
    with torch.inference_mode():
        for i, (pid, tensor) in enumerate(selected, 1):
            path = resolve_tensor(tensor, args.manifest)
            if not path.is_file():
                raise FileNotFoundError(f'{pid}: {path}')
            vec, _ = model.features_and_logits(make_views(path).to(device))
            arr = vec[0].detach().cpu().numpy().astype(np.float32)
            if arr.shape != (1024,) or not np.isfinite(arr).all():
                raise ValueError(f'Invalid feature vector for {pid}')
            features.append(arr)
            index.append({'patient_id': pid, 'split': 'test'})
            if i % 20 == 0 or i == len(selected):
                print(f'CT test features {i}/{len(selected)}', flush=True)
    out = args.out / 'smoke' if args.smoke else args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'ct_b_test_features_1024d.npy', np.stack(features), allow_pickle=False)
    with (out / 'ct_b_test_index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['patient_id', 'split'])
        w.writeheader(); w.writerows(index)
    report = {'status': 'FEATURES_ONLY', 'test_patients': len(index), 'feature_dim': 1024,
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
              'device': str(device), 'smoke': args.smoke, 'test_labels_read': False,
              'test_predictions_generated': False, 'test_metrics_computed': False}
    (out / 'readiness.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
