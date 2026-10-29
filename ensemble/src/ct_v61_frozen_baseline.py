"""Frozen CT V6.1 evaluation + feature caching (Mac MPS/CPU).

Run from ensemble/:
 python src/ct_v61_frozen_baseline.py --manifest output/ct_v5_encoder_audit/ct_v5_encoder_eligible_manifest.csv --models models/ct_v61 --out output/ct_v61_frozen_baseline --smoke
 python src/ct_v61_frozen_baseline.py --manifest output/ct_v5_encoder_audit/ct_v5_encoder_eligible_manifest.csv --models models/ct_v61 --out output/ct_v61_frozen_baseline

No test samples are loaded. Does not train or modify checkpoints.
Only load checkpoints from a trusted source.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

CHECKPOINTS = {
    'A_topk': 'v61_4dc62ff23e78_best.pt',
    'B_mean': 'v61_13a378082055_best.pt',
}

class CTEncoder(nn.Module):
    def __init__(self, pooling, dropout):
        super().__init__()
        self.pooling = pooling
        self.backbone = models.densenet121(weights=None)
        dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.feature_dim = dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 2))

    def features_and_logits(self, x):
        b, v, c, h, w = x.shape
        f = self.backbone(x.reshape(b*v, c, h, w)).reshape(b, v, -1)
        if self.pooling == 'mean':
            pooled = f.mean(dim=1)
        elif self.pooling == 'topk':
            scores = self.head(f).softmax(-1)[..., 1]
            indices = scores.topk(k=min(3, v), dim=1).indices
            pooled = f.gather(1, indices[..., None].expand(-1, -1, f.shape[-1])).mean(dim=1)
        else:
            raise ValueError(f'Unexpected pooling: {self.pooling}')
        return pooled, self.head(pooled)

    def forward(self, x):
        return self.features_and_logits(x)[1]

def load_checkpoint(path, device):
    # Only trusted teammate checkpoints: weights_only=False can execute serialized code.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    if cfg['arch'] != 'densenet121' or cfg['views'] != 12:
        raise ValueError(f'Unexpected architecture/views in {path}: {cfg}')
    model = CTEncoder(cfg['pooling'], cfg['dropout'])
    model.load_state_dict(ckpt['model'], strict=True)
    if model.feature_dim != 1024:
        raise ValueError('Unexpected feature dimension')
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg

def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'patient_id', 'split', 'label', 'tensor_path'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest missing required fields: {required}')
    seen = set()
    selected = []
    for row in rows:
        if row.get('eligibility_status', 'ELIGIBLE').upper() not in ('ELIGIBLE', 'PASS', 'APPROVED'):
            continue
        split = row['split'].strip().lower()
        if split == 'test' or split not in ('train', 'val'):
            continue
        pid = row['patient_id'].strip().upper()
        if pid in seen:
            raise ValueError(f'Duplicate patient {pid}')
        seen.add(pid)
        label = row['label'].strip().lower()
        if label in ('positive', 'covidpositive', '1'):
            y = 1
        elif label in ('negative', 'covidnegative', '0'):
            y = 0
        else:
            raise ValueError(f'Unknown label {label} for {pid}')
        selected.append((pid, split, y, row['tensor_path']))
    return sorted(selected, key=lambda r: (r[1], r[0]))

def resolve_tensor(s, manifest_path):
    path = Path(s).expanduser()
    if path.is_absolute():
        return path
    for p in (Path.cwd()/path, manifest_path.parent/path, Path.cwd()/'ct_v5_corrected_preprocessed'/path):
        if p.exists():
            return p
    return Path.cwd()/path

def make_views(tensor_path):
    with np.load(tensor_path, allow_pickle=False) as data:
        key = 'volume' if 'volume' in data.files else 'tensor' if 'tensor' in data.files else None
        if key is None:
            raise ValueError(f'Expected volume/tensor key in {tensor_path}; got {data.files}')
        vol = np.asarray(data[key], dtype=np.float32)
    if vol.shape != (64, 192, 192):
        raise ValueError(f'Unexpected shape {vol.shape} in {tensor_path}')
    if not np.isfinite(vol).all() or vol.min() < -0.01 or vol.max() > 1.01:
        raise ValueError(f'Invalid normalized values in {tensor_path}')
    centers = torch.linspace(3, 60, 12).round().long().tolist()
    views = np.stack([vol[c-1:c+2] for c in centers], axis=0)
    return torch.from_numpy(views.copy()).unsqueeze(0)

def metrics(rows):
    y = np.array([r['label'] for r in rows], dtype=int)
    p = np.array([r['covid_probability'] for r in rows], dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        'patients': len(rows), 'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'positive_f1': float(f1_score(y, pred, zero_division=0)),
        'precision_positive': float(precision_score(y, pred, zero_division=0)),
        'recall_positive': float(recall_score(y, pred, zero_division=0)),
        'recall_negative': float(tn / max(1, tn + fp)),
        'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        'confusion_matrix_TN_FP_FN_TP': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--models', type=Path, default=Path('models/ct_v61'))
    ap.add_argument('--out', type=Path, default=Path('output/ct_v61_frozen_baseline'))
    ap.add_argument('--smoke', action='store_true', help='One training patient per model, no metrics')
    ap.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    args = ap.parse_args()
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; use --device cpu')
    rows = read_rows(args.manifest)
    if not rows:
        raise ValueError('No train/val eligible patients')
    if args.smoke:
        rows = [next((r for r in rows if r[1] == 'train'), rows[0])]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Device={device} | records={len(rows)} | smoke={args.smoke}', flush=True)
    report = {'device': str(device), 'smoke': args.smoke, 'models': {},
              'warning': 'Not an independent validation result until overlap with teammate encoder training patients is ruled out.'}
    for name, filename in CHECKPOINTS.items():
        model, cfg = load_checkpoint(args.models/filename, device)
        preds, feats, feature_index = [], [], []
        with torch.inference_mode():
            for i, (pid, split, label, source) in enumerate(rows, 1):
                path = resolve_tensor(source, args.manifest)
                if not path.exists():
                    raise FileNotFoundError(f'{pid}: {path}')
                x = make_views(path).to(device)
                feature, logits = model.features_and_logits(x)
                prob = float(logits.float().softmax(-1)[0,1].cpu())
                vec = feature[0].float().cpu().numpy()
                if vec.shape != (1024,) or not np.isfinite(vec).all() or not np.isfinite(prob):
                    raise RuntimeError(f'Invalid model output for {pid}')
                preds.append({'patient_id': pid, 'split': split, 'label': label,
                              'covid_probability': prob, 'predicted_label': int(prob >= 0.5)})
                feats.append(vec)
                feature_index.append({'patient_id': pid, 'split': split, 'label': label})
                if i % 25 == 0 or i == len(rows):
                    print(f'{name}: {i}/{len(rows)}', flush=True)
        write_csv(args.out/f'{name}_predictions.csv', preds)
        write_csv(args.out/f'{name}_feature_index.csv', feature_index)
        np.save(args.out/f'{name}_features_1024d.npy', np.stack(feats))
        result = {'checkpoint': filename, 'pooling': cfg['pooling'],
                  'feature_shape': [len(feats), 1024], 'views': cfg['views']}
        if not args.smoke:
            val = [r for r in preds if r['split'] == 'val']
            if val:
                result['validation'] = metrics(val)
        report['models'][name] = result
        print(name, json.dumps(result, indent=2), flush=True)
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    (args.out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'DONE: {args.out / "summary.json"}', flush=True)

if __name__ == '__main__':
    main()
