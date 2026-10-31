#!/usr/bin/env python3
"""Read-only AP/PA preprocessing benchmark using teammate's frozen Keras encoder.

No test-set evaluation. Split membership comes exclusively from existing JSON.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEFAULT_ROOT = Path('/Users/wenting/Projects/BIMCV-COVID-19')
DEFAULT_MODEL = Path('/Users/wenting/Projects/PsuprProj/xraydata/Models/ModelB_CovidvsNormal_27sep/model_B_v2_27sep.keras')
VIEW = re.compile(r'(?:^|_)vp-(ap|pa)(?:_|\.)', re.I)
METHODS = ('normalized', 'histogram', 'clahe', 'histogram_clahe')


def key(row):
    return f"{row['covid_status'].strip().lower()}:{row['patient_id'].strip().upper()}".upper()


def split_ids(path):
    obj = json.loads(path.read_text())
    if 'splits' in obj and isinstance(obj['splits'], dict):
        obj = obj['splits']
    result = {}
    for name in ('train', 'val', 'test'):
        items = obj.get(name, obj.get('validation', []) if name == 'val' else [])
        if isinstance(items, dict):
            items = items.get('patient_ids', items.get('patients', []))
        result[name] = {str(x.get('patient_id', x.get('id', '')) if isinstance(x, dict) else x).strip().upper() for x in items}
    if not result['train'] or not result['val']:
        raise ValueError('Split JSON must include nonempty train and val lists')
    if any(result[a] & result[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Patient leakage: overlapping split identifiers')
    return result


def read_rows(manifest, splits):
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        source = list(csv.DictReader(f))
    if not source or not {'patient_id', 'covid_status', 'xray_path'} <= set(source[0]):
        raise ValueError('Manifest requires patient_id, covid_status, xray_path')
    rows = []
    seen = set()
    for r in source:
        path = Path(r['xray_path'])
        if not VIEW.search(path.name) or not path.is_file() or r['covid_status'].strip().lower() not in ('positive', 'negative'):
            continue
        k = key(r)
        if k in seen:
            raise ValueError(f'Multiple manifest rows for {k}: select one X-ray per patient first')
        seen.add(k)
        subset = next((s for s, ids in splits.items() if k in ids), None)
        if subset:
            rows.append({'key': k, 'path': path, 'split': subset, 'label': int(r['covid_status'].lower().strip() == 'positive')})
    return rows


def gray(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        if a.dtype != np.uint8:
            raise ValueError(f'Unexpected multichannel {a.dtype} image; review conversion')
        a = np.asarray(Image.fromarray(a).convert('L'))
    if a.ndim != 2:
        raise ValueError(f'Unexpected image shape {a.shape}')
    return a


def normalize(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        raise ValueError('Flat intensity image')
    return np.rint(255 * np.clip((a - lo) / (hi - lo), 0, 1)).astype(np.uint8)


def transform(base, ref, method):
    if method in ('histogram', 'histogram_clahe'):
        from skimage.exposure import match_histograms
        base = np.clip(np.rint(match_histograms(base, ref, channel_axis=None)), 0, 255).astype(np.uint8)
    if method in ('clahe', 'histogram_clahe'):
        import cv2
        base = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base)
    return base


def encoder_from_model(path, layer_name):
    import tensorflow as tf
    full = tf.keras.models.load_model(str(path), compile=False)
    if layer_name not in [x.name for x in full.layers]:
        raise ValueError(f'Layer {layer_name!r} missing. Available: {[x.name for x in full.layers]}')
    layer = full.get_layer(layer_name)
    if len(full.input_shape) != 4 or full.input_shape[-1] not in (1, 3):
        raise ValueError(f'Unexpected input shape: {full.input_shape}')
    try:
        model = tf.keras.Model(inputs=full.inputs, outputs=layer.output)
    except ValueError:
        # Sequential models may not have a graph built for intermediate outputs yet.
        full(tf.zeros((1,) + tuple(int(x or 224) for x in full.input_shape[1:])), training=False)
        model = tf.keras.Model(inputs=full.inputs, outputs=layer.output)
    model.trainable = False
    return model, full.input_shape


def image_input(a, shape, scale):
    import tensorflow as tf
    h, w, c = shape[1:]
    if not h or not w:
        raise ValueError('Encoder must have fixed H/W input')
    im = tf.convert_to_tensor(a[:, :, None], dtype=tf.float32)
    im = tf.image.resize_with_pad(im, h, w, antialias=True)
    if c == 3:
        im = tf.repeat(im, 3, axis=-1)
    if scale == 'zero_one':
        im = im / 255.0
    elif scale == 'minus_one_one':
        im = im / 127.5 - 1.0
    return im[None]


def evaluate(y, probs, threshold=0.5):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    pred = (probs >= threshold).astype(int)
    return {'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'covid_recall': float(recall_score(y, pred, pos_label=1, zero_division=0)),
            'noncovid_recall': float(recall_score(y, pred, pos_label=0, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probs)) if len(set(y)) == 2 else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DEFAULT_ROOT / 'dataAnalysis/output/paired_patient_manifest_ct_validated.csv')
    p.add_argument('--split-json', type=Path, default=Path('/Users/wenting/Projects/PsuprProj/ensemble/output/mixed_fusion/patient_splits.json'))
    p.add_argument('--xray-model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--feature-layer', default='dense')
    p.add_argument('--xray-scale', choices=['zero_255', 'zero_one', 'minus_one_one'], default='zero_255')
    p.add_argument('--reference', type=Path, help='Explicit AP/PA training reference; otherwise first sorted training image')
    p.add_argument('--output', type=Path, default=Path('/Users/wenting/Projects/PsuprProj/ensemble/output/xray_preprocessing_benchmark'))
    p.add_argument('--smoke', action='store_true', help='Extract 2 TRAIN samples; no classifier fitting')
    p.add_argument('--limit', type=int, default=0, help='Debug only: cap samples PER SPLIT (not for final comparisons)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    splits = split_ids(args.split_json)
    rows = read_rows(args.manifest, splits)
    train = sorted((r for r in rows if r['split'] == 'train'), key=lambda r: r['key'])
    if not train:
        p.error('No eligible train X-rays')
    if args.reference:
        refpath = args.reference.resolve()
        if not any(r['path'].resolve() == refpath for r in train):
            p.error('Reference must be an eligible TRAIN image in manifest')
    else:
        refpath = train[0]['path']
    reference = normalize(gray(refpath))
    if args.smoke:
        rows = train[:2]
    elif args.limit:
        rows = [r for s in ('train', 'val') for r in sorted((x for x in rows if x['split'] == s), key=lambda x: x['key'])[:args.limit]]
    else:
        rows = [r for r in rows if r['split'] in ('train', 'val')]
    print('Reference (review quality):', refpath, flush=True)
    print('Cohort:', dict(Counter(r['split'] for r in rows)), 'TEST NEVER LOADED', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    enc, shape = encoder_from_model(args.xray_model, args.feature_layer)
    print('Encoder input:', shape, 'feature layer:', args.feature_layer, flush=True)
    signature = hashlib.sha256(json.dumps({'model': str(args.xray_model.resolve()), 'model_size': args.xray_model.stat().st_size,
        'model_mtime': args.xray_model.stat().st_mtime_ns, 'reference': str(refpath.resolve()),
        'reference_mtime': refpath.stat().st_mtime_ns, 'layer': args.feature_layer,
        'scale': args.xray_scale, 'split': str(args.split_json.resolve()), 'manifest': str(args.manifest.resolve()),
        'preprocess': 'percentile1_99_matching_clahe2_8_v1'}, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = args.output / 'features' / signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for method in METHODS:
        features, labels, subsets, failures = [], [], [], []
        for n, r in enumerate(rows, 1):
            filename = cache_dir / method / (r['key'].replace(':', '_') + '.npz')
            try:
                if filename.is_file():
                    with np.load(filename) as data:
                        if str(data['path']) != str(r['path']) or int(data['mtime']) != r['path'].stat().st_mtime_ns:
                            raise ValueError('Stale cache: source changed. Remove cache file and rerun.')
                        vec = data['feature'].astype(np.float32)
                else:
                    base = normalize(gray(r['path']))
                    processed = transform(base, reference, method)
                    vec = np.asarray(enc(image_input(processed, shape, args.xray_scale), training=False)).reshape(-1).astype(np.float32)
                    if not np.isfinite(vec).all():
                        raise ValueError('Non-finite encoder features')
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(filename, feature=vec, path=str(r['path']), mtime=r['path'].stat().st_mtime_ns)
                features.append(vec)
                labels.append(r['label'])
                subsets.append(r['split'])
                if args.smoke or n % 25 == 0:
                    print(f'{method}: {n}/{len(rows)} OK {r["key"]} features={len(vec)}', flush=True)
            except Exception as exc:
                failures.append({'method': method, 'patient': r['key'], 'error': str(exc)})
                print(f'{method}: ERROR {r["key"]}: {exc}', flush=True)
        if failures:
            with (args.output / f'{method}_failures.json').open('w') as f:
                json.dump(failures, f, indent=2)
            raise RuntimeError(f'{method}: {len(failures)} failures; stop to keep identical cohorts')
        if args.smoke:
            continue
        x = np.stack(features)
        y = np.asarray(labels)
        subset = np.asarray(subsets)
        if len(set(y[subset == 'train'])) < 2 or len(set(y[subset == 'val'])) < 2:
            raise ValueError('Train/validation must contain both classes; do not use --limit for full comparisons')
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        best = None
        for c in (0.001, 0.01, 0.1, 1.0, 10.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=args.seed))
            clf.fit(x[subset == 'train'], y[subset == 'train'])
            prob = clf.predict_proba(x[subset == 'val'])[:, 1]
            metrics = evaluate(y[subset == 'val'], prob)
            if best is None or (metrics['macro_f1'], metrics['roc_auc'] or 0) > (best['validation']['macro_f1'], best['validation']['roc_auc'] or 0):
                best = {'method': method, 'C': c, 'validation': metrics, 'train_n': int(sum(subset == 'train')),
                        'val_n': int(sum(subset == 'val')), 'feature_dim': int(x.shape[1])}
        results.append(best)
        print('RESULT', json.dumps(best), flush=True)
    summary = {'feature_cache_tag': signature, 'reference': str(refpath), 'smoke': args.smoke,
               'test': 'NOT_LOADED_OR_EVALUATED', 'methods': results,
               'notes': ['Frozen teammate encoder; identical patient cohort across preprocessing methods.',
                         'Logistic-regression head with train-only standardization and validation-selected C.',
                         'This is NOT the same MLP as the earlier 0.6514 baseline; compare only within this run.',
                         'Source images never modified; validate teammate input scale and reference image quality.']}
    (args.output / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if results:
        with (args.output / 'validation_results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'C', 'train_n', 'val_n', 'feature_dim', *results[0]['validation']])
            writer.writeheader()
            for r in results:
                writer.writerow({**{k: r[k] for k in ('method', 'C', 'train_n', 'val_n', 'feature_dim')}, **r['validation']})
    print('Saved:', args.output / 'benchmark_summary.json', flush=True)


if __name__ == '__main__':
    main()
