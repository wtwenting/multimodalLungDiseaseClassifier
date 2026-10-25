"""CT B + X-ray frozen-feature MLP, train-only CV and one validation evaluation.
Run from ensemble/. No test-set data is loaded. Requires numpy, scikit-learn.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_table(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def load_features(root, feature_name, index_name, expected_dim):
    arr = np.load(root / feature_name, allow_pickle=False)
    rows = read_table(root / index_name)
    if arr.ndim != 2 or arr.shape != (len(rows), expected_dim) or not np.isfinite(arr).all():
        raise ValueError(f'Invalid feature matrix/index: {root / feature_name}, shape={arr.shape}')
    mapping = {}
    for i, row in enumerate(rows):
        patient = row['patient_id'].strip().upper()
        if patient in mapping:
            raise ValueError(f'Duplicate patient: {patient}')
        split = row['split'].strip().lower()
        if split not in ('train', 'val'):
            continue  # Do not include test/unassigned patients
        raw = row['label'].strip().lower()
        if raw not in ('0', '1', 'positive', 'negative', 'covidpositive', 'covidnegative'):
            raise ValueError(f'Unknown label for {patient}: {raw}')
        label = int(raw in ('1', 'positive', 'covidpositive'))
        mapping[patient] = (split, label, arr[i].astype(np.float32))
    return mapping


def make_pipeline(hidden, alpha, seed, max_iter):
    return Pipeline([
        ('scale', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=hidden, activation='relu', solver='adam',
                              alpha=alpha, learning_rate_init=0.0005, batch_size=32,
                              max_iter=max_iter, early_stopping=True, validation_fraction=0.15,
                              n_iter_no_change=15, random_state=seed)),
    ])


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': int(len(y)), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'positive_f1': float(f1_score(y, pred, zero_division=0)),
            'precision_positive': float(precision_score(y, pred, zero_division=0)),
            'recall_positive': float(recall_score(y, pred, zero_division=0)),
            'recall_negative': float(tn / max(1, tn + fp)),
            'roc_auc': float(roc_auc_score(y, p)),
            'confusion_matrix_TN_FP_FN_TP': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ct', type=Path, default=Path('output/ct_v61_frozen_baseline'))
    ap.add_argument('--xray', type=Path, default=Path('output/xray_patient_feature_aggregation'))
    ap.add_argument('--out', type=Path, default=Path('output/fusion_mlp'))
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--max-iter', type=int, default=180)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    ct = load_features(args.ct, 'B_mean_features_1024d.npy', 'B_mean_feature_index.csv', 1024)
    xr = load_features(args.xray, 'patient_features_256d.npy', 'patient_aggregation_index.csv', 256)
    ids = sorted(set(ct) & set(xr))
    for pid in ids:
        if ct[pid][:2] != xr[pid][:2]:
            raise ValueError(f'CT/X-ray split or label mismatch: {pid}: {ct[pid][:2]} vs {xr[pid][:2]}')
    train_ids = [pid for pid in ids if ct[pid][0] == 'train']
    val_ids = [pid for pid in ids if ct[pid][0] == 'val']
    y_train = np.array([ct[pid][1] for pid in train_ids], dtype=int)
    y_val = np.array([ct[pid][1] for pid in val_ids], dtype=int)
    if len(train_ids) < 30 or len(val_ids) < 10 or len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise RuntimeError('Insufficient matched train/val data')
    def matrix(pids):
        return np.stack([np.concatenate((ct[p][2], xr[p][2])) for p in pids])
    x_train, x_val = matrix(train_ids), matrix(val_ids)
    audit = {'matched_train': len(train_ids), 'matched_val': len(val_ids),
             'train_classes': dict(Counter(map(str, y_train))), 'val_classes': dict(Counter(map(str, y_val))),
             'feature_dim': x_train.shape[1], 'test_set_read': False,
             'caution': 'Check pretrained encoder patient overlap and CT/X-ray session alignment separately.'}
    (args.out / 'cohort_audit.json').write_text(json.dumps(audit, indent=2))
    print('COHORT:', json.dumps(audit), flush=True)
    if args.dry_run:
        print('DRY RUN COMPLETE', flush=True)
        return
    if args.folds < 2 or args.folds > min(Counter(y_train).values()):
        raise ValueError('Invalid number of CV folds')
    # Compare a small, predefined grid; all model selection uses training-only CV.
    configs = [(64,), (128,), (128, 32)]
    alphas = [0.01, 1.0]
    seeds = [42, 123, 2026]
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    results = []
    for hidden in configs:
        for alpha in alphas:
            fold_scores = []
            for seed in seeds:
                for tr, va in cv.split(x_train, y_train):
                    pipe = make_pipeline(hidden, alpha, seed, args.max_iter)
                    pipe.fit(x_train[tr], y_train[tr])
                    probs = pipe.predict_proba(x_train[va])[:, 1]
                    fold_scores.append(f1_score(y_train[va], probs >= 0.5, average='macro', zero_division=0))
            record = {'hidden': list(hidden), 'alpha': alpha,
                      'train_cv_macro_f1_mean': float(np.mean(fold_scores)),
                      'train_cv_macro_f1_std': float(np.std(fold_scores)),
                      'n_cv_runs': len(fold_scores)}
            results.append(record)
            print('CV', record, flush=True)
            (args.out / 'cv_results.json').write_text(json.dumps(results, indent=2))
    best = sorted(results, key=lambda r: (-r['train_cv_macro_f1_mean'], r['train_cv_macro_f1_std'], len(r['hidden'])))[0]
    # Seed stability is reported on the held-out validation cohort; no seed is chosen using validation results.
    val_results = []
    for seed in seeds:
        pipe = make_pipeline(tuple(best['hidden']), best['alpha'], seed, args.max_iter)
        pipe.fit(x_train, y_train)
        p = pipe.predict_proba(x_val)[:, 1]
        result = {'seed': seed, 'validation': metrics(y_val, p)}
        val_results.append(result)
        with (args.out / f'validation_predictions_seed_{seed}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['patient_id', 'label', 'probability_positive', 'predicted'])
            writer.writeheader()
            writer.writerows({'patient_id': pid, 'label': int(y), 'probability_positive': float(prob),
                              'predicted': int(prob >= 0.5)} for pid, y, prob in zip(val_ids, y_val, p))
        print(f'VALIDATION seed={seed} macro_f1={result["validation"]["macro_f1"]:.4f} auc={result["validation"]["roc_auc"]:.4f}', flush=True)
    summary = {'cohort': audit, 'selection': 'training-only stratified CV macro F1',
               'best_config': best, 'validation_by_seed': val_results,
               'validation_macro_f1_mean': float(np.mean([r['validation']['macro_f1'] for r in val_results])),
               'validation_macro_f1_std': float(np.std([r['validation']['macro_f1'] for r in val_results])),
               'validation_auc_mean': float(np.mean([r['validation']['roc_auc'] for r in val_results])),
               'threshold': 0.5, 'test_set_evaluated': False,
               'elapsed_seconds': round(time.time() - start, 1)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('DONE:', args.out / 'summary.json', flush=True)

if __name__ == '__main__':
    main()
