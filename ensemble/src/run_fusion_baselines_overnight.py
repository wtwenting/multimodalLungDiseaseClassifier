"""Read-only feature-fusion baselines: CT A, CT B, X-ray, CT A+X-ray, CT B+X-ray.
Run from ensemble/. Does not read test rows or retrain encoders.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def label_value(x):
    s = str(x).strip().lower()
    if s in ('1', 'positive', 'covidpositive'):
        return 1
    if s in ('0', 'negative', 'covidnegative'):
        return 0
    raise ValueError(f'Unrecognized label: {x}')


def load_feature_table(matrix, index, name):
    rows = read_csv(index)
    arr = np.load(matrix, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[0] != len(rows) or not np.isfinite(arr).all():
        raise ValueError(f'{name}: invalid matrix/index alignment or nonfinite values')
    mapping = {}
    for i, r in enumerate(rows):
        pid = r['patient_id'].strip().upper()
        split = r['split'].strip().lower()
        if pid in mapping:
            raise ValueError(f'{name}: duplicate patient {pid}')
        # Never include test or unassigned patients, even in data matching.
        if split not in ('train', 'val'):
            continue
        mapping[pid] = (split, label_value(r['label']), arr[i].astype(np.float32))
    return mapping, arr.shape[1]


def score(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'n': int(len(y)), 'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'positive_f1': float(f1_score(y, pred, zero_division=0)),
        'precision_positive': float(precision_score(y, pred, zero_division=0)),
        'recall_positive': float(recall_score(y, pred, zero_division=0)),
        'recall_negative': float(tn / max(1, tn + fp)),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'confusion_matrix_TN_FP_FN_TP': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ct', type=Path, default=Path('output/ct_v61_frozen_baseline'))
    ap.add_argument('--xray', type=Path, default=Path('output/xray_patient_feature_aggregation'))
    ap.add_argument('--out', type=Path, default=Path('output/fusion_lr_overnight'))
    ap.add_argument('--dry-run', action='store_true', help='Validate alignment; no training')
    ap.add_argument('--cv-folds', type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    xr, xr_dim = load_feature_table(args.xray/'patient_features_256d.npy', args.xray/'patient_aggregation_index.csv', 'Xray')
    ct = {}
    for name in ('A_topk', 'B_mean'):
        ct[name], dim = load_feature_table(args.ct/f'{name}_features_1024d.npy', args.ct/f'{name}_feature_index.csv', name)
        if dim != 1024:
            raise ValueError(f'{name} has {dim} features, expected 1024')
    if xr_dim != 256:
        raise ValueError(f'Xray has {xr_dim} features, expected 256')

    matched = sorted(set(xr) & set(ct['A_topk']) & set(ct['B_mean']))
    conflicts = []
    for pid in matched:
        facts = [(xr[pid][0], xr[pid][1]), (ct['A_topk'][pid][0], ct['A_topk'][pid][1]), (ct['B_mean'][pid][0], ct['B_mean'][pid][1])]
        if len(set(facts)) != 1:
            conflicts.append({'patient_id': pid, 'xray': str(facts[0]), 'ct_a': str(facts[1]), 'ct_b': str(facts[2])})
    if conflicts:
        write_csv(args.out/'patient_conflicts.csv', conflicts)
        raise RuntimeError(f'{len(conflicts)} patient label/split conflicts; see patient_conflicts.csv. No training performed.')

    train_ids = [p for p in matched if xr[p][0] == 'train']
    val_ids = [p for p in matched if xr[p][0] == 'val']
    train_y = np.array([xr[p][1] for p in train_ids], dtype=int)
    val_y = np.array([xr[p][1] for p in val_ids], dtype=int)
    audit = {
        'matched_train': len(train_ids), 'matched_val': len(val_ids),
        'train_class_counts': dict(Counter(map(str, train_y))),
        'val_class_counts': dict(Counter(map(str, val_y))),
        'ct_a_train_val': len(ct['A_topk']), 'ct_b_train_val': len(ct['B_mean']),
        'xray_train_val': len(xr),
        'warning': 'Same-patient CT/X-ray features are not proof of same-session or temporal alignment. Independently verify encoder pretraining patient overlap.',
        'test_set_read': False,
    }
    (args.out/'cohort_audit.json').write_text(json.dumps(audit, indent=2))
    print('COHORT AUDIT:', json.dumps(audit, indent=2), flush=True)
    if not train_ids or not val_ids or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        raise RuntimeError('Insufficient matched train/validation patients or missing class; no training performed.')
    write_csv(args.out/'matched_patient_index.csv', [{'patient_id': p, 'split': xr[p][0], 'label': xr[p][1]} for p in matched])
    if args.dry_run:
        print('DRY RUN COMPLETE: no models fitted', flush=True)
        return

    def matrix(mapping, ids):
        return np.stack([mapping[p][2] for p in ids]).astype(np.float64)

    datasets = {
        'CT_A_only': (matrix(ct['A_topk'], train_ids), matrix(ct['A_topk'], val_ids)),
        'CT_B_only': (matrix(ct['B_mean'], train_ids), matrix(ct['B_mean'], val_ids)),
        'Xray_only': (matrix(xr, train_ids), matrix(xr, val_ids)),
        'CT_A_plus_Xray': (np.concatenate([matrix(ct['A_topk'], train_ids), matrix(xr, train_ids)], axis=1), np.concatenate([matrix(ct['A_topk'], val_ids), matrix(xr, val_ids)], axis=1)),
        'CT_B_plus_Xray': (np.concatenate([matrix(ct['B_mean'], train_ids), matrix(xr, train_ids)], axis=1), np.concatenate([matrix(ct['B_mean'], val_ids), matrix(xr, val_ids)], axis=1)),
    }
    results = {'cohort': audit, 'models': {}, 'selection_metric': 'training CV macro_f1', 'validation_threshold': 0.5, 'test_set_evaluated': False}
    min_class = min(Counter(train_y).values())
    folds = min(args.cv_folds, min_class)
    if folds < 2:
        raise RuntimeError('Not enough training examples per class for stratified CV')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for name, (xtrain, xval) in datasets.items():
        print(f'RUN {name}: train={xtrain.shape} val={xval.shape}', flush=True)
        pipe = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=3000, solver='liblinear', random_state=42))])
        search = GridSearchCV(pipe, {'clf__C': [0.001, 0.01, 0.1, 1, 10], 'clf__class_weight': [None, 'balanced']}, scoring='f1_macro', cv=cv, n_jobs=1, refit=True, error_score='raise')
        search.fit(xtrain, train_y)
        p = search.predict_proba(xval)[:, 1]
        result = {'feature_dim': xtrain.shape[1], 'best_params': search.best_params_, 'train_cv_macro_f1': float(search.best_score_), 'validation': score(val_y, p)}
        results['models'][name] = result
        write_csv(args.out/f'{name}_validation_predictions.csv', [{'patient_id': pid, 'label': int(y), 'probability_positive': float(prob), 'predicted': int(prob >= 0.5)} for pid, y, prob in zip(val_ids, val_y, p)])
        (args.out/'summary.json').write_text(json.dumps(results, indent=2))
        print(f'COMPLETE {name}: val_macro_f1={result["validation"]["macro_f1"]:.4f} val_auc={result["validation"]["roc_auc"]:.4f}', flush=True)
    results['elapsed_seconds'] = round(time.time()-started, 2)
    (args.out/'summary.json').write_text(json.dumps(results, indent=2))
    print('DONE:', args.out/'summary.json', flush=True)

if __name__ == '__main__':
    main()
