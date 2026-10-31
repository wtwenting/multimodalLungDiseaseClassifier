#!/usr/bin/env python3
"""Frozen Model B v2 adaptive-preprocessing benchmark. No test pixels loaded."""
import argparse
import csv
import hashlib
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

METHODS = ('direct', 'adaptive', 'selective_clahe')
DEFAULT_MODEL = Path('/Users/wenting/Projects/PsuprProj/xraydata/Models/ModelB_CovidvsNormal_27sep/model_B_v2_27sep.keras')


def log(*args):
    print(*args, flush=True)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_image(path):
    with Image.open(path) as im:
        arr = np.asarray(im)
        if arr.ndim == 3:
            arr = np.asarray(im.convert('L'))
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D image: {path}')
    return arr.astype(np.float32)


def direct(a):
    if a.max() > 255:
        a = np.floor(a / 256.0)
    return np.clip(a, 0, 255).astype(np.uint8)


def robust(a):
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return direct(a)
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def process(a, method, decision):
    if method == 'direct':
        return direct(a)
    if method == 'adaptive':
        return robust(a) if decision['adaptive_decision'] == 'robust_1_99' else direct(a)
    rule = decision['selective_clahe_decision']
    if rule == 'clahe_after_robust':
        import cv2
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(robust(a))
    if rule == 'robust_1_99':
        return robust(a)
    if rule == 'direct':
        return direct(a)
    raise ValueError(f'Unknown decision {rule}')


def make_tensor(a):
    import tensorflow as tf
    x = tf.convert_to_tensor(a[..., None], dtype=tf.float32)
    x = tf.image.resize_with_pad(x, 224, 224, antialias=False)
    x = tf.repeat(x, 3, axis=-1) / 255.0
    if tuple(x.shape) != (224, 224, 3):
        raise ValueError('Invalid CNN input shape')
    return x


def patient_predictions(rows, probabilities):
    groups, labels = defaultdict(list), {}
    for row, prob in zip(rows, probabilities):
        groups[row['key']].append(float(prob))
        labels[row['key']] = int(row['label'])
    keys = sorted(groups)
    return keys, np.asarray([labels[k] for k in keys]), np.asarray([np.mean(groups[k]) for k in keys])


def scores(y, p, threshold):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 recall_score, precision_score, roc_auc_score, confusion_matrix)
    pred = (p >= threshold).astype(int)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'covid_recall': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'noncovid_recall': float(recall_score(y, pred, pos_label=0, zero_division=0)),
        'covid_precision': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, default=Path('output/xray_overnight/splits'))
    parser.add_argument('--audit', type=Path, default=Path('output/xray_adaptive_audit'))
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=Path('output/xray_adaptive_benchmark'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--baseline-threshold', type=float, default=0.425)
    parser.add_argument('--c', type=float, default=0.01)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    decisions = read_csv(args.audit / 'preprocessing_decisions.csv')
    decision_map = {r['path']: r for r in decisions}
    if len(decision_map) != len(decisions):
        raise ValueError('Duplicate paths in audit decisions')
    thresholds = json.loads((args.audit / 'thresholds_train_only.json').read_text())
    log('STAGE 1: read existing splits and audit decisions')
    all_rows = []
    for split in ('train', 'val', 'test'):
        rows = read_csv(args.splits / f'{split}_manifest.csv')
        for r in rows:
            if r['split'] != split:
                raise ValueError(f'Incorrect split field: {r}')
            if r['view'] not in ('AP', 'PA'):
                raise ValueError(f'Unexpected view: {r}')
            if r['path'] not in decision_map:
                raise ValueError(f'Missing audit decision: {r["path"]}')
            d = decision_map[r['path']]
            if d['key'] != r['key'] or d['split'] != split or d['view'] != r['view']:
                raise ValueError(f'Audit/manifest mismatch: {r["path"]}')
            all_rows.append(r)
    by_patient = defaultdict(set)
    by_label = defaultdict(set)
    for r in all_rows:
        pid = r['key'].split(':')[-1]
        by_patient[pid].add(r['split'])
        by_label[pid].add(int(r['label']))
    if any(len(x) != 1 for x in by_patient.values()) or any(len(x) != 1 for x in by_label.values()):
        raise ValueError('Patient split overlap or label conflict')
    train_rows = sorted([r for r in all_rows if r['split'] == 'train'], key=lambda r: (r['key'], r['path']))
    val_rows = sorted([r for r in all_rows if r['split'] == 'val'], key=lambda r: (r['key'], r['path']))
    log('Cohort:', dict(Counter(r['split'] for r in all_rows)), 'patients:', len(by_patient))
    log('TEST NOT LOADED: test paths checked in manifests only; test pixels never decoded')
    if args.smoke:
        # Cover both classes if available, with different patients.
        samples = []
        for label in (0, 1):
            for r in train_rows:
                if int(r['label']) == label:
                    samples.append(r)
                    break
        train_rows = samples
        val_rows = []
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    full = tf.keras.models.load_model(str(args.model), compile=False)
    if tuple(full.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f'Unexpected encoder input {full.input_shape}')
    encoder = tf.keras.Model(full.inputs, full.get_layer('dense').output)
    encoder.trainable = False
    log('STAGE 2: frozen encoder', args.model, 'input:', full.input_shape)
    model_hash = hashlib.sha256()
    with args.model.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            model_hash.update(chunk)
    signature = hashlib.sha256(json.dumps({
        'model_sha256': model_hash.hexdigest(),
        'thresholds': thresholds,
        'manifest_paths': [(r['path'], r['bytes'], r['mtime_ns'], r['split']) for r in train_rows + val_rows],
        'audit_decisions': [(r['path'], r['adaptive_decision'], r['selective_clahe_decision']) for r in decisions if r['split'] in ('train', 'val')],
        'tensor': 'tf_resize_with_pad_antialias_false_0to1',
        'methods': METHODS,
    }, sort_keys=True).encode()).hexdigest()[:16]
    log('Feature cache tag:', signature)
    results = []
    for method in METHODS:
        log('METHOD', method)
        feat = []
        cohort = train_rows + val_rows
        for i, r in enumerate(cohort, 1):
            path = Path(r['path'])
            stat = path.stat()
            if stat.st_size != int(r['bytes']) or stat.st_mtime_ns != int(r['mtime_ns']):
                raise RuntimeError(f'Source image changed since manifest: {path}')
            cache = args.output / 'features' / signature / method / (hashlib.sha256(str(path).encode()).hexdigest()[:24] + '.npz')
            if cache.exists():
                with np.load(cache) as z:
                    if int(z['bytes']) != stat.st_size or int(z['mtime_ns']) != stat.st_mtime_ns:
                        raise RuntimeError(f'Stale feature cache: {cache}')
                    vec = z['features'].astype(np.float32)
            else:
                raw = read_image(path)
                out = process(raw, method, decision_map[str(path)])
                x = make_tensor(out)
                vec = np.asarray(encoder(x[None, ...], training=False)).reshape(-1).astype(np.float32)
                if i <= 4 and r['split'] == 'train':
                    preview = args.output / 'previews' / method / (r['key'].replace(':', '_') + '_' + str(i) + '.png')
                    preview.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.clip(np.rint(np.asarray(x) * 255), 0, 255).astype(np.uint8)).save(preview)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, features=vec, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if vec.shape != (128,) or not np.isfinite(vec).all():
                raise ValueError(f'Invalid features for {path}: {vec.shape}')
            feat.append(vec)
            if i % 25 == 0 or args.smoke:
                log(f'  {i}/{len(cohort)} {r["key"]}')
        if args.smoke:
            continue
        ntrain = len(train_rows)
        Xtrain, Xval = np.stack(feat[:ntrain]), np.stack(feat[ntrain:])
        ytrain = np.array([int(r['label']) for r in train_rows])
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        counts = Counter(r['key'] for r in train_rows)
        weights = np.array([1 / counts[r['key']] for r in train_rows])
        weights *= len(weights) / weights.sum()
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.c, max_iter=2000, random_state=42))
        clf.fit(Xtrain, ytrain, logisticregression__sample_weight=weights)
        probabilities = clf.predict_proba(Xval)[:, 1]
        keys, y, p = patient_predictions(val_rows, probabilities)
        if len(keys) != 98:
            log('WARNING: Validation patient count is', len(keys), '(expected 98 from previous run)')
        fixed = scores(y, p, args.baseline_threshold)
        rec = {'method': method, 'head': 'logistic', 'C': args.c, 'threshold': args.baseline_threshold,
               'mode': 'fixed_baseline', 'train_images': ntrain, 'val_images': len(val_rows),
               'val_patients': len(keys), **fixed}
        results.append(rec)
        save_csv(args.output / 'reports' / 'validation_comparison.csv', results)
        log('FIXED RESULT', json.dumps(rec))
        best = None
        for threshold in np.arange(0.2, 0.801, 0.025):
            m = scores(y, p, float(threshold))
            rank = (m['macro_f1'], m['balanced_accuracy'])
            if best is None or rank > best[0]:
                best = (rank, float(threshold), m)
        tuned = {'method': method, 'head': 'logistic', 'C': args.c, 'threshold': best[1],
                 'mode': 'validation_tuned', 'train_images': ntrain, 'val_images': len(val_rows),
                 'val_patients': len(keys), **best[2]}
        results.append(tuned)
        save_csv(args.output / 'reports' / 'validation_comparison.csv', results)
        log('TUNED RESULT', json.dumps(tuned))
        # Validation predictions are for diagnostic review only; no test predictions.
        save_csv(args.output / 'reports' / f'validation_predictions_{method}.csv', [
            {'key': k, 'label': int(yy), 'prob_covid': float(pp)} for k, yy, pp in zip(keys, y, p)])
    if args.smoke:
        log('SMOKE PASS: 3 preprocessing methods x 2 TRAIN images; TEST NEVER LOADED')
        return
    fixed_results = [r for r in results if r['mode'] == 'fixed_baseline']
    best_fixed = max(fixed_results, key=lambda r: (r['macro_f1'], r['balanced_accuracy']))
    summary = {'dataset_images': len(all_rows), 'train_images': len(train_rows), 'val_images': len(val_rows),
               'test_images_inventory_only': sum(r['split'] == 'test' for r in all_rows),
               'patients': len(by_patient), 'model': str(args.model), 'model_sha256': model_hash.hexdigest(),
               'feature_cache_tag': signature, 'baseline_reference': {'accuracy': 0.7346938775510204,
               'macro_f1': 0.7306553911205074, 'method': 'direct', 'head': 'logistic', 'C': 0.01, 'threshold': 0.425},
               'best_fixed_validation': best_fixed, 'test': 'NOT_LOADED_OR_EVALUATED',
               'notes': ['Thresholds from prior training-only audit; no rules recalculated.',
                         'Fixed settings isolate preprocessing; tuned results are exploratory.',
                         'Test manifest read for overlap checks, test image pixels never decoded.',
                         'Audit preview uses PIL resize, this benchmark uses original TensorFlow resize_with_pad to match prior benchmark.']}
    save_json(args.output / 'reports' / 'experiment_summary.json', summary)
    log('DONE', args.output.resolve())


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
