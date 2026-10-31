#!/usr/bin/env python3
"""Frozen Keras X-ray + MONAI PyTorch CT feature fusion for paired BIMCV patients.

Cache feature vectors per patient, then train a small PyTorch classifier.
The source encoders are never fine-tuned. Requires teammate preprocessing review.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

DEFAULT_MANIFEST = '/Users/wenting/Projects/BIMCV-COVID-19/dataAnalysis/output/paired_patient_manifest_ct_validated.csv'
DEFAULT_XRAY = '/Users/wenting/Projects/PsuprProj/xraydata/Models/ModelB_CovidvsNormal_27sep/model_B_v2_27sep.keras'
DEFAULT_CT = '/Users/wenting/Projects/PsuprProj/CT/model/ct_3d_covid_normal_densenet121.pt'
PREPROCESS_VERSION = 'xray_tf_pad_ct_trilinear_float32_v3'


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default=DEFAULT_MANIFEST)
    p.add_argument('--xray-model', default=DEFAULT_XRAY)
    p.add_argument('--ct-checkpoint', default=DEFAULT_CT)
    p.add_argument('--output', default='/Users/wenting/Projects/PsuprProj/ensemble/output/mixed_fusion')
    p.add_argument('--mode', choices=['inspect', 'extract', 'train', 'all'], default='inspect')
    p.add_argument('--model', choices=['xray', 'ct', 'fusion'], default='fusion', help='Cached features used by classifier')
    p.add_argument('--evaluate-test', action='store_true', help='Evaluate held-out test set only after model selection')
    p.add_argument('--no-standardize', action='store_true', help='Disable train-only feature scaling for legacy comparisons')
    p.add_argument('--xray-feature-layer', default=None, help='Layer name to extract; default: last suitable layer before final classifier')
    p.add_argument('--xray-size', type=int, nargs=2, default=None, metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--xray-scale', choices=['zero_one', 'minus_one_one', 'zero_255'], default='zero_255')
    p.add_argument('--ct-shape', type=int, nargs=3, default=[64, 128, 128])
    p.add_argument('--ct-intensity', choices=['percentile', 'hu'], default='percentile')
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--min-delta', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rebuild-splits', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='Only extract first N patients for a smoke test')
    return p.parse_args()


def read_manifest(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    needed = {'patient_id', 'covid_status', 'xray_path', 'ct_path'}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError(f'Manifest missing columns {needed}')
    result = []
    seen = set()
    for r in rows:
        status = r['covid_status'].strip().lower()
        if status not in ('positive', 'negative'):
            continue
        patient_id = r['patient_id'].strip().removeprefix('sub-')
        key = (status, patient_id)
        if key in seen:
            raise ValueError(f'Duplicate patient row: {key}')
        seen.add(key)
        if not Path(r['xray_path']).is_file() or not Path(r['ct_path']).is_file():
            continue
        result.append({'patient_id': patient_id, 'status': status,
                       'label': int(status == 'positive'), 'xray_path': r['xray_path'], 'ct_path': r['ct_path']})
    return result


def load_xray_encoder(args):
    import tensorflow as tf
    full = tf.keras.models.load_model(args.xray_model, compile=False)
    print('X-ray model input:', full.input_shape)
    print('X-ray model output:', full.output_shape)
    print('Available X-ray layers:')
    for layer in full.layers:
        try:
            shape = str(layer.output.shape)
        except (AttributeError, ValueError):
            shape = '?'
        print(f'  {layer.name:36s} {shape}')
    if args.xray_feature_layer:
        layer = full.get_layer(args.xray_feature_layer)
    else:
        # Prefer the known 128-dimensional penultimate Dense layer for Model B.
        try:
            layer = full.get_layer('dense')
        except ValueError:
            if len(full.layers) < 2:
                raise ValueError('Specify --xray-feature-layer for this architecture')
            layer = full.layers[-2]
        print('AUTO feature layer:', layer.name, '(VERIFY with X-ray teammate)')
    encoder = tf.keras.Model(inputs=full.inputs, outputs=layer.output)
    encoder.trainable = False
    shape = full.input_shape
    if isinstance(shape, list):
        raise ValueError('Multi-input X-ray models require a custom adapter')
    if len(shape) != 4:
        raise ValueError(f'Expected NHWC image input, got {shape}')
    return encoder, shape


def xray_features(path, encoder, shape, args):
    """Match teammate's tf.io.decode_image + tf.image.resize_with_pad pipeline.

    Note: 0-255 is the notebook's base preprocessing. Confirm any extra
    normalization inside Model B before relying on final evaluation.
    """
    import tensorflow as tf
    height = args.xray_size[0] if args.xray_size else shape[1]
    width = args.xray_size[1] if args.xray_size else shape[2]
    if not height or not width:
        raise ValueError('Dynamic image dimensions: pass --xray-size HEIGHT WIDTH')
    channels = shape[-1]
    if channels not in (1, 3):
        raise ValueError(f'Unsupported image channel count: {channels}')
    image_bytes = tf.io.read_file(str(path))
    image = tf.io.decode_image(image_bytes, channels=channels, expand_animations=False)
    image.set_shape([None, None, channels])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, height, width, antialias=True)
    if args.xray_scale == 'zero_one':
        image = image / 255.0
    elif args.xray_scale == 'minus_one_one':
        image = image / 127.5 - 1.0
    features = encoder(image[None], training=False).numpy()
    return features.reshape(-1).astype(np.float32)


def load_ct_encoder(checkpoint, device):
    import torch
    from monai.networks.nets import DenseNet121
    from torch import nn
    from torch.nn import functional as F
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)

    class Encoder(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.features = source.features
        def forward(self, x):
            z = F.relu(self.features(x))
            return F.adaptive_avg_pool3d(z, 1).flatten(1)

    enc = Encoder(model).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def ct_features(path, encoder, args):
    """Canonical NIfTI -> [D,H,W], percentile [0,1], torch trilinear resize."""
    import nibabel as nib
    import torch
    from torch.nn import functional as F
    nii = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    if vol.ndim != 3 or min(vol.shape) < 2:
        raise ValueError(f'Invalid CT volume: {vol.shape}')
    vol = np.transpose(vol, (2, 1, 0))
    if not np.isfinite(vol).all():
        raise ValueError('CT volume contains NaN or infinity')
    if args.ct_intensity == 'hu':
        vol = (np.clip(vol, -1000, 400) + 1000) / 1400
    else:
        lo, hi = np.percentile(vol, [1, 99])
        vol = np.clip(vol, lo, hi)
        if hi > lo:
            vol = (vol - lo) / (hi - lo)
        else:
            vol = np.zeros_like(vol)
    # Resize on CPU to avoid a second full-resolution copy on the accelerator.
    volume_tensor = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None].float()
    with torch.inference_mode():
        resized = F.interpolate(volume_tensor, size=tuple(args.ct_shape),
                                mode='trilinear', align_corners=False)
        resized = resized.to(device=args.device, dtype=next(encoder.parameters()).dtype)
        return encoder(resized).cpu().numpy().reshape(-1).astype(np.float32)


def fingerprint(args, layer_name):
    paths = [Path(args.xray_model), Path(args.ct_checkpoint)]
    payload = {'preprocessing_version': PREPROCESS_VERSION, 'models': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
               'xray_layer': layer_name, 'xray_size': args.xray_size, 'xray_scale': args.xray_scale,
               'ct_shape': args.ct_shape, 'ct_intensity': args.ct_intensity,
               'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def extract(args, rows, out):
    import torch
    xenc, shape = load_xray_encoder(args)
    ctenc = load_ct_encoder(args.ct_checkpoint, args.device)
    tag = fingerprint(args, xenc.layers[-1].name)
    cache = out / 'features' / tag
    cache.mkdir(parents=True, exist_ok=True)
    selected = rows[:args.limit] if args.limit else rows
    failures = []
    for index, r in enumerate(selected, 1):
        target = cache / f"{r['status']}_{r['patient_id']}.npz"
        if target.exists():
            print(f'[{index}/{len(selected)}] SKIP {target.name}', flush=True)
            continue
        try:
            xf = xray_features(r['xray_path'], xenc, shape, args)
            cf = ct_features(r['ct_path'], ctenc, args)
            if not np.all(np.isfinite(xf)) or not np.all(np.isfinite(cf)):
                raise ValueError('Nonfinite features')
            tmp = target.with_suffix('.partial.npz')
            np.savez_compressed(tmp, xray=xf, ct=cf, label=r['label'])
            tmp.replace(target)
            print(f'[{index}/{len(selected)}] OK {target.name}: X-ray {len(xf)}, CT {len(cf)}', flush=True)
        except Exception as exc:
            failures.append((r['patient_id'], str(exc)))
            print(f'[{index}/{len(selected)}] ERROR {r["patient_id"]}: {exc}', flush=True)
    (out / 'active_cache.json').write_text(json.dumps({'tag': tag, 'feature_layer': xenc.layers[-1].name,
        'xray_model': args.xray_model, 'ct_checkpoint': args.ct_checkpoint,
        'xray_scale': args.xray_scale, 'ct_shape': args.ct_shape, 'ct_intensity': args.ct_intensity,
                'manifest': str(Path(args.manifest).resolve())}, indent=2))
    with (out / 'feature_errors.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['patient_id', 'error']); writer.writerows(failures)
    print('Feature cache:', cache, 'Failures:', len(failures))


def split_patients(rows, path, seed, rebuild):
    ids = {f"{r['status']}:{r['patient_id']}" for r in rows}
    if path.exists() and not rebuild:
        data = json.loads(path.read_text())
        old = set().union(*(set(v) for v in data.values()))
        if old != ids:
            raise ValueError('Patient cohort changed. Use --rebuild-splits only for a NEW experiment.')
        return data
    rng = random.Random(seed)
    result = {'train': [], 'val': [], 'test': []}
    for status in ('negative', 'positive'):
        group = [f"{r['status']}:{r['patient_id']}" for r in rows if r['status'] == status]
        if len(group) < 3:
            raise ValueError(f'Insufficient {status} patients for split')
        rng.shuffle(group)
        n = max(1, round(len(group) * 0.15))
        result['test'] += group[:n]
        result['val'] += group[n:2*n]
        result['train'] += group[2*n:]
    path.write_text(json.dumps(result, indent=2))
    return result


def train(args, rows, out):
    """Train modality-specific head on frozen cached features; preserve patient split."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    import joblib

    if args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0 or args.min_delta < 0:
        raise ValueError('Invalid training hyperparameters')
    config_path = out / 'active_cache.json'
    if not config_path.exists():
        raise ValueError('No active feature cache. Run --mode extract first.')
    tag = json.loads(config_path.read_text())['tag']
    cache = out / 'features' / tag
    available = {}
    expected_dims = {'xray': 128, 'ct': 1024}
    patient_status = {}
    for r in rows:
        pid = r['patient_id']
        if pid in patient_status and patient_status[pid] != r['status']:
            raise ValueError(f'Patient appears in both cohorts: {pid}')
        patient_status[pid] = r['status']
        file = cache / f"{r['status']}_{pid}.npz"
        if not file.is_file():
            continue
        with np.load(file, allow_pickle=False) as data:
            xf = np.asarray(data['xray'], dtype=np.float32).reshape(-1)
            cf = np.asarray(data['ct'], dtype=np.float32).reshape(-1)
            if len(xf) != expected_dims['xray'] or len(cf) != expected_dims['ct']:
                raise ValueError(f'Unexpected encoder feature dimension: {file}')
            if not np.isfinite(xf).all() or not np.isfinite(cf).all():
                raise ValueError(f'Nonfinite features: {file}')
            if int(data['label']) != r['label']:
                raise ValueError(f'Cached label mismatch: {file}')
        features = {'xray': xf, 'ct': cf, 'fusion': np.concatenate((xf, cf))}[args.model]
        available[f"{r['status']}:{pid}"] = (features, r['label'])
    print(f'Cached patient pairs: {len(available)} | model: {args.model} | cache: {tag}')
    if not available:
        raise ValueError('No cached patients available')
    selected_rows = [r for r in rows if f"{r['status']}:{r['patient_id']}" in available]
    split_path = out / 'patient_splits.json'
    splits = split_patients(selected_rows, split_path, args.seed, args.rebuild_splits)
    sets = [set(splits[k]) for k in ('train', 'val', 'test')]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Patient leakage across saved splits')
    if set().union(*sets) != set(available):
        raise ValueError('Split keys do not match available cached patients')
    if args.rebuild_splits:
        print('WARNING: patient split was rebuilt; prior results may not be comparable')

    def array_for(split):
        items = [available[k] for k in splits[split]]
        return np.stack([v for v, _ in items]).astype(np.float32), np.array([y for _, y in items], dtype=np.float32)
    arrays = {name: array_for(name) for name in ('train', 'val', 'test')}
    experiment_dir = out / 'experiments' / args.model
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_standardize:
        scaler = StandardScaler()
        scaler.fit(arrays['train'][0])  # never fit on validation or test
        arrays = {name: (scaler.transform(x).astype(np.float32), y) for name, (x, y) in arrays.items()}
        joblib.dump(scaler, experiment_dir / 'train_feature_scaler.joblib')
    else:
        print('WARNING: feature standardization disabled')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    input_dim = arrays['train'][0].shape[1]
    model = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)).to(args.device)
    def loader(name):
        x, y = arrays[name]
        return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                          batch_size=args.batch_size, shuffle=(name == 'train'))
    loaders = {name: loader(name) for name in ('train', 'val', 'test')}
    for name in ('train', 'val', 'test'):
        y = arrays[name][1]
        print(f'{name}: {len(y)} patients; positive={int(y.sum())}; negative={int(len(y)-y.sum())}')
    y_train = arrays['train'][1]
    pos = int(y_train.sum()); neg = int(len(y_train) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], device=args.device))
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def evaluate(name):
        model.eval()
        truth, probs = [], []
        with torch.inference_mode():
            for x, y in loaders[name]:
                scores = torch.sigmoid(model(x.to(args.device))).flatten().cpu().numpy()
                probs.extend(scores.tolist())
                truth.extend(y.numpy().astype(int).tolist())
        pred = [int(p >= 0.5) for p in probs]
        return {'accuracy': float(accuracy_score(truth, pred)),
                'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
                'macro_f1': float(f1_score(truth, pred, average='macro', zero_division=0)),
                'covid_recall': float(recall_score(truth, pred, pos_label=1, zero_division=0)),
                'noncovid_recall': float(recall_score(truth, pred, pos_label=0, zero_division=0)),
                'roc_auc': float(roc_auc_score(truth, probs)) if len(set(truth)) == 2 else None}

    best, best_epoch, stalled = -1.0, 0, 0
    history = []
    checkpoint = experiment_dir / 'best.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loaders['train']:
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(args.device)).flatten(), y.to(args.device))
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(y)
        val = evaluate('val')
        train_loss = total_loss / len(splits['train'])
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_macro_f1': val['macro_f1']})
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} val_macro_f1={val["macro_f1"]:.4f}', flush=True)
        if best_epoch == 0 or val['macro_f1'] > best + args.min_delta:
            best, best_epoch, stalled = val['macro_f1'], epoch, 0
            torch.save(model.state_dict(), checkpoint)
            print(f'  Best checkpoint saved (epoch {epoch})', flush=True)
        else:
            stalled += 1
            if stalled >= args.patience:
                print(f'Early stopping at epoch {epoch}; best epoch {best_epoch}', flush=True)
                break
    (experiment_dir / 'history.json').write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    result = {'model': args.model, 'feature_cache_tag': tag, 'feature_dim': input_dim,
              'standardized_train_only': not args.no_standardize,
              'best_epoch': best_epoch, 'epochs_run': len(history),
              'learning_rate': args.learning_rate, 'patience': args.patience,
              'validation': evaluate('val'),
              'test': evaluate('test') if args.evaluate_test else 'NOT_EVALUATED',
              'note': 'Use validation to select modality; test only after final model choice.'}
    (experiment_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print('Saved experiment:', experiment_dir)


def main():
    args = arguments()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    print('Paired patients:', len(rows))
    if args.mode in ('inspect', 'extract', 'all'):
        if args.mode == 'inspect':
            load_xray_encoder(args)
            print('INSPECT OK. Confirm selected feature layer and preprocessing with teammate.')
            return
        extract(args, rows, out)
    if args.mode in ('train', 'all'):
        train(args, rows, out)


if __name__ == '__main__':
    main()
