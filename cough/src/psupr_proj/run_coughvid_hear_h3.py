#!/usr/bin/env python3
"""H3: regularization study on COUGHVID frozen-HeAR embeddings.

Research question: does stronger L2, dropout, or a smaller MLP improve
out-of-sample discrimination compared with the existing H0 baseline?

Design safeguards:
* Binary healthy/COVID-19 only; symptomatic samples are excluded.
* No audio augmentation, no HeAR re-extraction, and no test-set access.
* Fit ANOVA feature selection and scaling on INTERNAL training only.
* Select hyperparameters and thresholds on an INTERNAL tuning subset.
* Evaluate only a few internally selected finalists on the fixed validation set.
* Keep original H0 results unchanged in a separate directory.

Run from cough repository root:
  python src/psupr_proj/run_coughvid_hear_h3.py --stages l2 dropout

Dependencies: numpy, scipy, scikit-learn, joblib. TensorFlow is additionally
required for --stages dropout. No GPU is required for these small networks.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path, split):
    with np.load(path, allow_pickle=False) as data:
        x = data['embeddings'].astype('float32')
        labels = data['label'].astype(str)
        ids = data['uuid'].astype(str)
        if 'split' in data and not np.all(data['split'].astype(str) == split):
            raise ValueError(f'Unexpected split in {path}')
    mask = np.isin(labels, ['healthy', 'COVID-19'])
    x, labels, ids = x[mask], labels[mask], ids[mask]
    if x.ndim != 2 or x.shape[1] != 512 or not np.isfinite(x).all():
        raise ValueError('Expected finite 512-D embeddings')
    if len(ids) != len(set(ids)) or set(labels) != {'healthy', 'COVID-19'}:
        raise ValueError('Duplicate IDs or missing class')
    return x, (labels == 'COVID-19').astype(int), ids


def metrics(y, score, threshold):
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'covid_f1': float(f1_score(y, pred, zero_division=0)),
            'covid_precision': float(precision_score(y, pred, zero_division=0)),
            'covid_recall': float(recall_score(y, pred, zero_division=0)),
            'specificity': float(tn / (tn + fp)), 'accuracy': float((pred == y).mean()),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'roc_auc': float(roc_auc_score(y, score)),
            'average_precision': float(average_precision_score(y, score))}


def best_threshold(y, score):
    options = [metrics(y, score, t) for t in np.arange(0.05, 0.951, 0.05)]
    return max(options, key=lambda r: (r['macro_f1'], r['covid_recall'], r['specificity']))


def build_keras(d, layers, alpha, dropout, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(d,))
    z = inputs
    for units in layers:
        z = tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(alpha))(z)
        if dropout:
            z = tf.keras.layers.Dropout(dropout)(z)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(z)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model


def main(args):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import f_classif
    from sklearn.neural_network import MLPClassifier
    import joblib

    x, y, ids = load_npz(args.train_npz, 'train')
    xv, yv, val_ids = load_npz(args.validation_npz, 'validation')
    if set(ids) & set(val_ids):
        raise ValueError('Train/validation recording overlap')
    # Each UUID is treated as a recording group. If participant identifiers
    # exist elsewhere, split by participant instead to avoid subject leakage.
    it, ii = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=args.seed)
    scaler = StandardScaler().fit(x[it])
    xt, xi, xval = (scaler.transform(a) for a in (x[it], x[ii], xv))
    scores, _ = f_classif(xt, y[it])
    ranking = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind='stable')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    signature = {'train_sha256': file_hash(args.train_npz), 'validation_sha256': file_hash(args.validation_npz),
                 'seed': args.seed, 'internal_tuning_fraction': 0.2,
                 'feature_counts': args.features, 'stages': args.stages}
    state_path = out / 'h3_summary.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['signature'] != signature:
            raise ValueError('Configuration changed; use a new output directory')
    else:
        state = {'signature': signature, 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'counts': {'train': len(it), 'internal_tuning': len(ii), 'validation': len(yv)},
                 'test_set_used': False, 'runs': {}, 'finalists': {}}
        atomic_json(state_path, state)
    joblib.dump({'scaler': scaler, 'ranking': ranking, 'internal_train_indices': it,
                 'internal_tuning_indices': ii}, out / 'internal_preprocessing.joblib')
    for k in args.features:
        cols = ranking[:k]
        for stage in args.stages:
            if stage == 'l2':
                configs = [{'alpha': a, 'dropout': 0.0, 'layers': (128, 32)} for a in (0.0001, 0.001, 0.01, 0.1)]
            else:
                # 0% dropout is essential as a framework-controlled baseline.
                configs = [{'alpha': 0.001, 'dropout': d, 'layers': layers}
                           for layers in ((64, 32), (128, 32)) for d in (0.0, 0.1, 0.2, 0.3)]
            for config in configs:
                key = f"{stage}_f{k}_layers{'-'.join(map(str, config['layers']))}_l2{config['alpha']}_drop{config['dropout']}"
                if key in state['runs'] and 'internal_tuning' in state['runs'][key]:
                    print('[SKIP]', key, flush=True)
                    continue
                print('[RUN]', key, flush=True)
                try:
                    t0 = time.perf_counter()
                    if stage == 'l2':
                        model = MLPClassifier(hidden_layer_sizes=config['layers'], alpha=config['alpha'],
                                              max_iter=200, early_stopping=False, random_state=args.seed)
                        weights = len(it) / (2 * np.bincount(y[it], minlength=2)[y[it]])
                        import inspect
                        if 'sample_weight' not in inspect.signature(model.fit).parameters:
                            raise RuntimeError('Requires scikit-learn MLPClassifier.fit(sample_weight)')
                        model.fit(xt[:, cols], y[it], sample_weight=weights)
                        prob = model.predict_proba(xi[:, cols])[:, 1]
                        model_path = out / 'models' / (key + '.joblib')
                        model_path.parent.mkdir(exist_ok=True)
                        joblib.dump(model, model_path)
                    else:
                        import tensorflow as tf
                        model = build_keras(k, config['layers'], config['alpha'], config['dropout'], args.seed)
                        weights = len(it) / (2 * np.bincount(y[it], minlength=2))
                        model.fit(xt[:, cols], y[it], epochs=100, batch_size=64, verbose=0,
                                  class_weight={0: float(weights[0]), 1: float(weights[1])},
                                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12,
                                                                               restore_best_weights=True)])
                        prob = model.predict(xi[:, cols], verbose=0).ravel()
                        model_path = out / 'models' / (key + '.keras')
                        model_path.parent.mkdir(exist_ok=True)
                        model.save(model_path)
                    state['runs'][key] = {'stage': stage, 'features': k, 'config': config,
                                          'internal_tuning': best_threshold(y[ii], prob),
                                          'fit_seconds': time.perf_counter() - t0,
                                          'model_path': str(model_path)}
                except Exception as exc:
                    state['runs'][key] = {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
                    print('[ERROR]', key, exc, flush=True)
                atomic_json(state_path, state)
    # Select ONE finalist per stage from internal tuning data. Only finalists
    # are evaluated on the fixed validation set; never select thresholds there.
    for stage in args.stages:
        eligible = [(key, r) for key, r in state['runs'].items() if r.get('stage') == stage and 'internal_tuning' in r]
        if not eligible:
            continue
        key, winner = max(eligible, key=lambda pair: pair[1]['internal_tuning']['macro_f1'])
        cols = ranking[:winner['features']]
        if stage == 'l2':
            model = joblib.load(winner['model_path'])
            prob = model.predict_proba(xval[:, cols])[:, 1]
        else:
            import tensorflow as tf
            model = tf.keras.models.load_model(winner['model_path'])
            prob = model.predict(xval[:, cols], verbose=0).ravel()
        threshold = winner['internal_tuning']['threshold']
        state['finalists'][stage] = {'run_key': key, 'threshold_from_internal_tuning': threshold,
                                     'fixed_validation': metrics(yv, prob, threshold)}
        atomic_json(state_path, state)
    rows = []
    for key, r in state['runs'].items():
        if 'internal_tuning' in r:
            rows.append({'run': key, 'stage': r['stage'], 'features': r['features'],
                         'internal_macro_f1': r['internal_tuning']['macro_f1'],
                         'internal_covid_recall': r['internal_tuning']['covid_recall'],
                         'internal_threshold': r['internal_tuning']['threshold']})
    if rows:
        with (out / 'h3_internal_comparison.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('Saved:', state_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-npz', type=Path, default=Path('data/coughvid_hear_embeddings/hear_train.npz'))
    p.add_argument('--validation-npz', type=Path, default=Path('data/coughvid_hear_embeddings/hear_validation.npz'))
    p.add_argument('--output-dir', type=Path, default=Path('data/coughvid_hear_h3_regularization'))
    p.add_argument('--stages', nargs='+', choices=['l2', 'dropout'], default=['l2', 'dropout'])
    p.add_argument('--features', nargs='+', type=int, choices=[64, 256], default=[64, 256])
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
