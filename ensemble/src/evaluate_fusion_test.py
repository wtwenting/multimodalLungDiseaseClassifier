#!/usr/bin/env python3
"""One-time locked CT B + X-ray held-out test evaluation.

Run from ensemble/. First use --check-only (no labels or predictions).
Then run --evaluate once, after approving the cohort and frozen model.
Never refits scaler, classifier, or encoders.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_block(folder, array_name, index_name, dim):
    matrix_path = folder / array_name
    index_path = folder / index_name
    matrix = np.load(matrix_path, allow_pickle=False)
    idx = pd.read_csv(index_path, dtype={'patient_id': str, 'split': str})
    if matrix.ndim != 2 or matrix.shape != (len(idx), dim):
        raise ValueError(f'{array_name}: expected ({len(idx)},{dim}), got {matrix.shape}')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{array_name}: non-finite features')
    if not {'patient_id', 'split'}.issubset(idx.columns):
        raise ValueError(f'{index_name}: patient_id/split columns required')
    idx['patient_id'] = idx.patient_id.str.strip().str.upper()
    if idx.patient_id.isna().any() or idx.patient_id.eq('').any() or idx.patient_id.duplicated().any():
        raise ValueError(f'{index_name}: missing/duplicate patient IDs')
    if not idx['split'].str.strip().str.lower().eq('test').all():
        raise ValueError(f'{index_name}: contains non-test patients')
    return {pid: matrix[i] for i, pid in enumerate(idx.patient_id)}, {
        'features_sha256': digest(matrix_path), 'index_sha256': digest(index_path)
    }


def label_to_int(v):
    v = str(v).strip().lower()
    if v in ('1', 'positive', 'covidpositive'):
        return 1
    if v in ('0', 'negative', 'covidnegative'):
        return 0
    raise ValueError(f'Unknown test label: {v!r}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct', type=Path, default=Path('output/ct_test_features'))
    p.add_argument('--xray', type=Path, default=Path('output/xray_test_features_reviewed'))
    p.add_argument('--frozen', type=Path, default=Path('output/fusion_frozen_final'))
    p.add_argument('--test-manifest', type=Path, default=Path('output/xray_overnight/splits/test_manifest.csv'))
    p.add_argument('--out', type=Path, default=Path('output/fusion_final_test'))
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true', help='No labels, predictions or test metrics')
    mode.add_argument('--evaluate', action='store_true', help='Run final one-time test scoring')
    args = p.parse_args()

    ct, ct_hashes = load_block(args.ct, 'ct_b_test_features_1024d.npy', 'ct_b_test_index.csv', 1024)
    xr, xr_hashes = load_block(args.xray, 'xray_test_features_256d.npy', 'xray_test_index.csv', 256)
    if set(ct) != set(xr):
        raise RuntimeError(f'Patient mismatch: CT only={len(set(ct)-set(xr))}, X-ray only={len(set(xr)-set(ct))}')
    patients = sorted(ct)
    if not patients:
        raise RuntimeError('Empty test cohort')

    report_path = args.frozen / 'freeze_report.json'
    model_path = args.frozen / 'ct_b_xray_logistic_frozen.joblib'
    report = json.loads(report_path.read_text())
    if report.get('status') != 'FROZEN' or report.get('feature_order') != ['CT_B_mean_1024d', 'Xray_256d']:
        raise RuntimeError('Frozen report does not match CT B then X-ray feature order')
    if report.get('feature_dim') != 1280 or report.get('threshold') != 0.5:
        raise RuntimeError('Frozen feature dimension or threshold changed')
    if digest(model_path) != report.get('model_sha256'):
        raise RuntimeError('Saved model SHA-256 differs from freeze report')
    clf_cfg = report.get('classifier', {})
    if clf_cfg.get('C') != 0.1 or clf_cfg.get('solver') != 'liblinear' or clf_cfg.get('class_weight') is not None:
        raise RuntimeError('Unexpected frozen classifier configuration')
    ct_ready = json.loads((args.ct / 'readiness.json').read_text())
    xr_ready = json.loads((args.xray / 'readiness.json').read_text())
    if ct_ready.get('checkpoint_sha256') is None or xr_ready.get('manual_review_applied') != 15:
        raise RuntimeError('Missing expected CT checkpoint hash or approved X-ray reviews')
    if any(r.get('test_metrics_computed') or r.get('test_predictions_generated') for r in (ct_ready, xr_ready)):
        raise RuntimeError('Input feature readiness indicates test scoring; investigate')
    if xr_ready.get('feature_dim') != 256 or not xr_ready.get('model_b_first') or not xr_ready.get('model_c_second'):
        raise RuntimeError('X-ray feature order or dimensions invalid')

    audit = {
        'status': 'PREFLIGHT_PASS', 'matched_test_patients': len(patients),
        'ct_dim': 1024, 'xray_dim': 256, 'fusion_dim': 1280,
        'feature_order': report['feature_order'], 'threshold': 0.5,
        'model_sha256': digest(model_path), 'ct_inputs': ct_hashes, 'xray_inputs': xr_hashes,
        'ct_checkpoint_sha256': ct_ready['checkpoint_sha256'],
        'xray_model_b_sha256': xr_ready.get('model_b_sha256'),
        'xray_model_c_sha256': xr_ready.get('model_c_sha256'),
        'manual_review_sha256': xr_ready.get('manual_review_file_sha256'),
        'limitation': 'Patient-level matching does not establish same-session alignment.'
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        audit['test_labels_read'] = False
        audit['test_predictions_generated'] = False
        (args.out / 'preflight.json').write_text(json.dumps(audit, indent=2))
        print('PREFLIGHT PASS (no labels or predictions):', json.dumps(audit, indent=2))
        return

    # The only section that reads held-out labels. Do not use labels for feature selection.
    test_df = pd.read_csv(args.test_manifest)
    if not {'key', 'label'}.issubset(test_df.columns):
        raise ValueError('Test manifest must contain key and label')
    test_df['patient_id'] = test_df['key'].astype(str).str.split(':').str[-1].str.strip().str.upper()
    if 'split' in test_df and not test_df['split'].astype(str).str.lower().eq('test').all():
        raise ValueError('Test manifest contains non-test records')
    test_df['binary_label'] = test_df['label'].map(label_to_int)
    if (test_df.groupby('patient_id')['binary_label'].nunique() > 1).any():
        raise ValueError('Conflicting test labels for a patient')
    labels = test_df.groupby('patient_id')['binary_label'].first().to_dict()
    if set(patients) != set(labels):
        raise RuntimeError(f'Test manifest coverage mismatch: missing labels={len(set(patients)-set(labels))}, extra manifest patients={len(set(labels)-set(patients))}')

    # Trusted, locally generated model only: joblib deserialization can execute code.
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, 'predict_proba') or not hasattr(pipeline, 'named_steps'):
        raise RuntimeError('Unexpected frozen pipeline type')
    if list(pipeline.named_steps) != ['scale', 'clf']:
        raise RuntimeError('Unexpected scaler/classifier pipeline steps')
    estimator = pipeline.named_steps['clf']
    if estimator.C != 0.1 or estimator.solver != 'liblinear' or estimator.class_weight is not None:
        raise RuntimeError('Loaded classifier settings differ from freeze report')
    if list(estimator.classes_) != [0, 1] or pipeline.named_steps['scale'].n_features_in_ != 1280:
        raise RuntimeError('Unexpected classes or fitted scaler dimensions')
    x = np.stack([np.concatenate([ct[pid], xr[pid]]) for pid in patients]).astype(np.float64)
    y = np.asarray([labels[pid] for pid in patients], dtype=int)
    prob = pipeline.predict_proba(x)[:, 1]
    if not np.isfinite(prob).all():
        raise RuntimeError('Non-finite predictions')
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    metrics = {
        'n': len(patients), 'class_counts': {'negative': int((y == 0).sum()), 'positive': int((y == 1).sum())},
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'positive_f1': float(f1_score(y, pred, pos_label=1, zero_division=0)),
        'precision_positive': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'recall_positive': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'recall_negative': float(recall_score(y, pred, pos_label=0, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, prob)) if len(set(y)) == 2 else None,
        'confusion_matrix_TN_FP_FN_TP': cm,
    }
    pd.DataFrame({'patient_id': patients, 'label': y, 'probability_positive': prob,
                  'predicted': pred}).to_csv(args.out / 'test_patient_predictions.csv', index=False)
    audit['status'] = 'FINAL_TEST_EVALUATED'
    audit['test_labels_read'] = True
    audit['test_predictions_generated'] = True
    audit['test_manifest_sha256'] = digest(args.test_manifest)
    audit['metrics'] = metrics
    (args.out / 'final_test_summary.json').write_text(json.dumps(audit, indent=2))
    print('FINAL HELD-OUT TEST:', json.dumps(metrics, indent=2))
    print('Results:', args.out.resolve())


if __name__ == '__main__':
    main()
