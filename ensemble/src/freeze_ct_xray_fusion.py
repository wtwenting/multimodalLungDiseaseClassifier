"""Freeze CT B + X-ray Logistic Regression using TRAIN only; verify against saved VAL predictions.
Run from ensemble/. Does not access test data, tune hyperparameters, or alter input features.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def label(x):
    x = str(x).strip().lower()
    if x in ('1', 'positive', 'covidpositive'):
        return 1
    if x in ('0', 'negative', 'covidnegative'):
        return 0
    raise ValueError(f'Unrecognized label: {x}')


def load_features(folder, stem, expected_dim):
    arr = np.load(folder / f'{stem}.npy', allow_pickle=False)
    index = csv_rows(folder / ('B_mean_feature_index.csv' if stem.startswith('B_mean') else 'patient_aggregation_index.csv'))
    if arr.shape != (len(index), expected_dim) or not np.isfinite(arr).all():
        raise ValueError(f'Invalid feature matrix {stem}: {arr.shape} / {len(index)} rows')
    result = {}
    for i, row in enumerate(index):
        split = row['split'].strip().lower()
        if split not in ('train', 'val'):
            continue  # never access test feature values
        pid = row['patient_id'].strip().upper()
        if pid in result:
            raise ValueError(f'Duplicate patient in {stem}: {pid}')
        result[pid] = (split, label(row['label']), arr[i].astype(np.float64))
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct', type=Path, default=Path('output/ct_v61_frozen_baseline'))
    p.add_argument('--xray', type=Path, default=Path('output/xray_patient_feature_aggregation'))
    p.add_argument('--reference', type=Path, default=Path('output/fusion_lr_overnight/CT_B_plus_Xray_validation_predictions.csv'))
    p.add_argument('--out', type=Path, default=Path('output/fusion_frozen_final'))
    p.add_argument('--tolerance', type=float, default=1e-5, help='Max absolute difference in validation probabilities')
    args = p.parse_args()
    ct = load_features(args.ct, 'B_mean_features_1024d', 1024)
    xr = load_features(args.xray, 'patient_features_256d', 256)
    ids = sorted(set(ct) & set(xr))
    for pid in ids:
        if ct[pid][:2] != xr[pid][:2]:
            raise ValueError(f'CT/X-ray split or label mismatch for {pid}')
    train = [pid for pid in ids if ct[pid][0] == 'train']
    val = [pid for pid in ids if ct[pid][0] == 'val']
    if len(train) != 446 or len(val) != 97:
        raise RuntimeError(f'Cohort changed: train={len(train)} val={len(val)}; expected 446/97. Stop.')

    def make_xy(patients):
        x = np.stack([np.concatenate((ct[pid][2], xr[pid][2])) for pid in patients])
        y = np.array([ct[pid][1] for pid in patients], dtype=int)
        return x, y

    xtrain, ytrain = make_xy(train)
    xval, yval = make_xy(val)
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, solver='liblinear', class_weight=None,
                                   max_iter=3000, random_state=42)),
    ])
    pipe.fit(xtrain, ytrain)
    probabilities = pipe.predict_proba(xval)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(yval, predictions, labels=[0, 1]).tolist()
    metrics = {'macro_f1': float(f1_score(yval, predictions, average='macro')),
               'roc_auc': float(roc_auc_score(yval, probabilities)),
               'accuracy': float(accuracy_score(yval, predictions)),
               'confusion_matrix': cm}

    if not args.reference.exists():
        raise FileNotFoundError(f'Reference predictions required: {args.reference}')
    reference = csv_rows(args.reference)
    refmap = {}
    for row in reference:
        pid = row['patient_id'].strip().upper()
        if pid in refmap:
            raise ValueError(f'Duplicate reference patient: {pid}')
        refmap[pid] = (label(row['label']), float(row['probability_positive']))
    if set(refmap) != set(val):
        raise RuntimeError('Reference validation patient IDs do not match; refusing to freeze.')
    differences = []
    for pid, y, prob in zip(val, yval, probabilities):
        if refmap[pid][0] != y:
            raise RuntimeError(f'Reference label mismatch for {pid}')
        differences.append(abs(prob - refmap[pid][1]))
    max_diff = float(max(differences))
    if max_diff > args.tolerance:
        raise RuntimeError(f'Validation probability reproduction FAILED: max difference={max_diff:.8g} > {args.tolerance}. No model saved.')

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / 'ct_b_xray_logistic_frozen.joblib'
    joblib.dump(pipe, model_path)
    config = {
        'model': 'CT_B_plus_Xray', 'status': 'FROZEN', 'train_patients': len(train),
        'validation_patients': len(val), 'feature_order': ['CT_B_mean_1024d', 'Xray_256d'],
        'feature_dim': 1280, 'classifier': {'C': 0.1, 'solver': 'liblinear',
        'class_weight': None, 'max_iter': 3000, 'random_state': 42},
        'scaler': 'StandardScaler fitted on 446 training patients', 'threshold': 0.5,
        'ct_checkpoint': 'v61_13a378082055_best.pt',
        'validation': metrics, 'max_abs_probability_difference_from_reference': max_diff,
        'reproduction_tolerance': args.tolerance, 'test_set_accessed': False,
        'input_files_sha256': {
            'ct_features': sha256(args.ct / 'B_mean_features_1024d.npy'),
            'ct_index': sha256(args.ct / 'B_mean_feature_index.csv'),
            'xray_features': sha256(args.xray / 'patient_features_256d.npy'),
            'xray_index': sha256(args.xray / 'patient_aggregation_index.csv'),
            'reference_predictions': sha256(args.reference),
        },
        'model_sha256': sha256(model_path),
        'limitations': ['Patient-level CT/X-ray matching does not establish same-session timing.',
                        'Independent encoder training patient overlap must be documented.',
                        'Test features not generated or evaluated.'],
    }
    (args.out / 'freeze_report.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with (args.out / 'validation_reproduced.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'label', 'probability_positive', 'predicted'])
        for pid, y, prob, pred in zip(val, yval, probabilities, predictions):
            writer.writerow([pid, int(y), float(prob), int(pred)])
    print('FREEZE SUCCESS:', model_path)
    print('VALIDATION:', json.dumps(metrics))
    print('Max probability difference from original:', max_diff)
    print('TEST SET NOT ACCESSED')


if __name__ == '__main__':
    main()
