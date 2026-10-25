"""Separate CT/X-ray PCA fusion experiment, train-only CV, untouched test.

Run from ensemble/: python src/run_fusion_separate_pca.py --dry-run
Then: python src/run_fusion_separate_pca.py --out output/fusion_separate_pca
Requires numpy, scipy, scikit-learn. Uses existing frozen feature caches.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def label_value(value):
    value = str(value).strip().lower()
    if value in ('1', 'positive', 'covidpositive'):
        return 1
    if value in ('0', 'negative', 'covidnegative'):
        return 0
    raise ValueError('Unknown label: ' + value)


def load_features(folder, stem, index_name, expected_dim):
    matrix = np.load(folder / stem, allow_pickle=False)
    rows = read_csv(folder / index_name)
    if matrix.ndim != 2 or matrix.shape != (len(rows), expected_dim) or not np.isfinite(matrix).all():
        raise ValueError(f'Invalid feature matrix/index: {stem}: {matrix.shape}')
    result = {}
    for i, row in enumerate(rows):
        pid = row['patient_id'].strip().upper()
        split = row['split'].strip().lower()
        if pid in result:
            raise ValueError('Duplicate patient: ' + pid)
        # Do not load test patients into any fitted or evaluated array.
        if split in ('train', 'val'):
            result[pid] = (split, label_value(row['label']), matrix[i].astype(np.float64))
    return result


class SeparatePCA(BaseEstimator, TransformerMixin):
    """Fit CT and X-ray scaler/PCA independently INSIDE each CV fold."""
    def __init__(self, ct_components=64, xray_components=32, seed=42):
        self.ct_components = ct_components
        self.xray_components = xray_components
        self.seed = seed

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1280:
            raise ValueError('Expected concatenated CT(1024) + Xray(256) features')
        if max(self.ct_components, self.xray_components) >= X.shape[0]:
            raise ValueError('PCA components must be smaller than training fold size')
        self.ct_scale_ = StandardScaler().fit(X[:, :1024])
        self.xr_scale_ = StandardScaler().fit(X[:, 1024:])
        self.ct_pca_ = PCA(n_components=self.ct_components, svd_solver='randomized', random_state=self.seed).fit(self.ct_scale_.transform(X[:, :1024]))
        self.xr_pca_ = PCA(n_components=self.xray_components, svd_solver='full').fit(self.xr_scale_.transform(X[:, 1024:]))
        return self

    def transform(self, X):
        X = np.asarray(X)
        a = self.ct_pca_.transform(self.ct_scale_.transform(X[:, :1024]))
        b = self.xr_pca_.transform(self.xr_scale_.transform(X[:, 1024:]))
        return np.concatenate((a, b), axis=1)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'positive_f1': float(f1_score(y, pred, zero_division=0)),
            'precision_positive': float(precision_score(y, pred, zero_division=0)),
            'recall_positive': float(recall_score(y, pred, zero_division=0)),
            'recall_negative': float(tn / max(1, tn + fp)),
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'confusion_matrix_TN_FP_FN_TP': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def make_pipeline(ct_n, xr_n, kind, seed, c=0.1, alpha=0.01):
    if kind == 'LR':
        clf = LogisticRegression(C=c, solver='liblinear', max_iter=3000, random_state=seed)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=alpha, max_iter=350,
                            early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=18, random_state=seed)
    return Pipeline([('separate_pca', SeparatePCA(ct_n, xr_n)),
                     ('post_scale', StandardScaler()), ('clf', clf)])


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct', type=Path, default=Path('output/ct_v61_frozen_baseline'))
    parser.add_argument('--xray', type=Path, default=Path('output/xray_patient_feature_aggregation'))
    parser.add_argument('--out', type=Path, default=Path('output/fusion_separate_pca'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--folds', type=int, default=3)
    args = parser.parse_args()
    start = time.time()
    ct = load_features(args.ct, 'B_mean_features_1024d.npy', 'B_mean_feature_index.csv', 1024)
    xr = load_features(args.xray, 'patient_features_256d.npy', 'patient_aggregation_index.csv', 256)
    matched = sorted(set(ct) & set(xr))
    for pid in matched:
        if ct[pid][:2] != xr[pid][:2]:
            raise RuntimeError(f'CT/X-ray split or label mismatch: {pid}')
    train_ids = [p for p in matched if ct[p][0] == 'train']
    val_ids = [p for p in matched if ct[p][0] == 'val']
    ytrain = np.array([ct[p][1] for p in train_ids], dtype=int)
    yval = np.array([ct[p][1] for p in val_ids], dtype=int)
    if len(set(ytrain)) != 2 or len(set(yval)) != 2 or min(Counter(ytrain).values()) < args.folds:
        raise RuntimeError('Insufficient class coverage for stratified CV')
    audit = {'matched_train': len(train_ids), 'matched_val': len(val_ids),
             'train_classes': dict(Counter(map(str, ytrain))), 'val_classes': dict(Counter(map(str, yval))),
             'input_dim': 1280, 'test_set_read': False,
             'note': 'Patient-level match does not establish same clinical encounter; verify upstream encoder independence.'}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cohort_audit.json').write_text(json.dumps(audit, indent=2))
    print('COHORT', json.dumps(audit), flush=True)
    if args.dry_run:
        print('DRY RUN COMPLETE; no PCA or classifiers fitted', flush=True)
        return
    def stack(ids):
        return np.stack([np.concatenate((ct[p][2], xr[p][2])) for p in ids])
    Xtrain, Xval = stack(train_ids), stack(val_ids)
    configs = []
    for a, b in ((32, 16), (64, 32), (128, 64)):
        for c in (0.01, 0.1, 1.0):
            configs.append({'kind': 'LR', 'ct': a, 'xray': b, 'C': c})
        for alpha in (0.01, 1.0):
            configs.append({'kind': 'MLP', 'ct': a, 'xray': b, 'alpha': alpha})
    seeds = (42, 123, 2026)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    results = []
    for config in configs:
        scores = []
        model_seeds = (42,) if config['kind'] == 'LR' else seeds
        for seed in model_seeds:
            for train_idx, hold_idx in cv.split(Xtrain, ytrain):
                model = make_pipeline(config['ct'], config['xray'], config['kind'], seed,
                                      c=config.get('C', 0.1), alpha=config.get('alpha', 0.01))
                model.fit(Xtrain[train_idx], ytrain[train_idx])
                pred = model.predict(Xtrain[hold_idx])
                scores.append(float(f1_score(ytrain[hold_idx], pred, average='macro', zero_division=0)))
        result = {**config, 'cv_macro_f1_mean': float(np.mean(scores)),
                  'cv_macro_f1_std': float(np.std(scores)), 'cv_fits': len(scores)}
        results.append(result)
        print('CV', result, flush=True)
        (args.out / 'cv_results.json').write_text(json.dumps(results, indent=2))
    # Select the best configuration PER classifier using training CV ONLY.
    summary = {'cohort': audit, 'selection': 'train-only stratified CV macro F1',
               'threshold': 0.5, 'test_set_evaluated': False, 'models': {}}
    for kind in ('LR', 'MLP'):
        chosen = max((r for r in results if r['kind'] == kind), key=lambda r: r['cv_macro_f1_mean'])
        per_seed = []
        for seed in ((42,) if kind == 'LR' else seeds):
            model = make_pipeline(chosen['ct'], chosen['xray'], kind, seed,
                                  c=chosen.get('C', 0.1), alpha=chosen.get('alpha', 0.01))
            model.fit(Xtrain, ytrain)
            p = model.predict_proba(Xval)[:, 1]
            per_seed.append({'seed': seed, 'validation': metrics(yval, p),
                             'ct_explained_variance': float(model.named_steps['separate_pca'].ct_pca_.explained_variance_ratio_.sum()),
                             'xray_explained_variance': float(model.named_steps['separate_pca'].xr_pca_.explained_variance_ratio_.sum())})
            save_csv(args.out / f'{kind}_validation_predictions_seed_{seed}.csv',
                     [{'patient_id': pid, 'label': int(y), 'probability_positive': float(prob),
                       'predicted': int(prob >= 0.5)} for pid, y, prob in zip(val_ids, yval, p)])
        summary['models'][kind] = {'selected_config': chosen, 'validation_by_seed': per_seed,
                                   'validation_macro_f1_mean': float(np.mean([r['validation']['macro_f1'] for r in per_seed])),
                                   'validation_auc_mean': float(np.mean([r['validation']['roc_auc'] for r in per_seed]))}
        print('FINAL', kind, summary['models'][kind]['validation_macro_f1_mean'], flush=True)
    summary['elapsed_seconds'] = round(time.time() - start, 2)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('DONE', args.out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
