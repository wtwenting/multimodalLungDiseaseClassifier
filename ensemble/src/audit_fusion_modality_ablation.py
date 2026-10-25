"""Patient-level modality block permutation audit for locked CT B + X-ray LR.
Refits the already-selected LR configuration on training rows because the prior
baseline runner did not save the fitted estimator. No tuning or test-set reads.
Run from ensemble/. Requires numpy, scikit-learn.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score


def read_table(matrix, index, expected):
    x = np.load(matrix, allow_pickle=False)
    with open(index, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if x.ndim != 2 or x.shape != (len(rows), expected) or not np.isfinite(x).all():
        raise ValueError(f'Invalid feature matrix or row alignment: {matrix}')
    result = {}
    for i, r in enumerate(rows):
        pid = r['patient_id'].strip().upper()
        split = r['split'].strip().lower()
        if split not in ('train', 'val'):
            continue
        label = str(r['label']).strip().lower()
        if label not in ('0', '1', 'positive', 'negative', 'covidpositive', 'covidnegative'):
            raise ValueError(f'Invalid label {label} for {pid}')
        y = int(label in ('1', 'positive', 'covidpositive'))
        if pid in result:
            raise ValueError(f'Duplicate patient {pid}')
        result[pid] = (split, y, x[i].astype(np.float64))
    return result


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)),
            'accuracy': float(accuracy_score(y, pred)),
            'recall_positive': float(tp / max(1, tp + fn)),
            'recall_negative': float(tn / max(1, tn + fp)),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ct', type=Path, default=Path('output/ct_v61_frozen_baseline'))
    ap.add_argument('--xray', type=Path, default=Path('output/xray_patient_feature_aggregation'))
    ap.add_argument('--baseline-summary', type=Path, default=Path('output/fusion_lr_overnight/summary.json'))
    ap.add_argument('--out', type=Path, default=Path('output/fusion_modality_ablation'))
    ap.add_argument('--iterations', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.iterations < 10:
        ap.error('--iterations must be >= 10')
    baseline = json.loads(args.baseline_summary.read_text())
    selected = baseline['models']['CT_B_plus_Xray']['best_params']
    c = float(selected['clf__C'])
    class_weight = selected['clf__class_weight']
    ct = read_table(args.ct/'B_mean_features_1024d.npy', args.ct/'B_mean_feature_index.csv', 1024)
    xr = read_table(args.xray/'patient_features_256d.npy', args.xray/'patient_aggregation_index.csv', 256)
    matched = sorted(set(ct) & set(xr))
    for pid in matched:
        if ct[pid][:2] != xr[pid][:2]:
            raise ValueError(f'Split/label mismatch: {pid}')
    train = [pid for pid in matched if ct[pid][0] == 'train']
    val = [pid for pid in matched if ct[pid][0] == 'val']
    if len(train) != 446 or len(val) != 97:
        raise ValueError(f'Cohort mismatch: train={len(train)} val={len(val)}; expected 446/97')
    def make_x(ids):
        return np.stack([np.r_[ct[p][2], xr[p][2]] for p in ids])
    xtrain, xval = make_x(train), make_x(val)
    ytrain = np.array([ct[p][1] for p in train]); yval = np.array([ct[p][1] for p in val])
    pipe = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(C=c, class_weight=class_weight, max_iter=3000, solver='liblinear', random_state=42))])
    pipe.fit(xtrain, ytrain)
    original_p = pipe.predict_proba(xval)[:, 1]
    original = metrics(yval, original_p)
    previous = baseline['models']['CT_B_plus_Xray']['validation']
    for key in ('macro_f1', 'roc_auc'):
        if abs(original[key] - previous[key]) > 1e-6:
            raise RuntimeError(f'Reconstructed model differs from saved baseline: {key} {original[key]} vs {previous[key]}')
    rng = np.random.default_rng(args.seed)
    raw = {'CT': [], 'Xray': []}
    for _ in range(args.iterations):
        for modality, sl in [('CT', slice(0, 1024)), ('Xray', slice(1024, 1280))]:
            shuffled = xval.copy()
            shuffled[:, sl] = xval[rng.permutation(len(val)), sl]
            p = pipe.predict_proba(shuffled)[:, 1]
            raw[modality].append(metrics(yval, p))
    result = {'method': 'Validation-set patient-level block permutation of one modality at a time; fixed model, no retraining or tuning',
              'model': 'CT_B_plus_Xray', 'baseline_config': {'C': c, 'class_weight': class_weight, 'scaler': 'StandardScaler', 'solver': 'liblinear'},
              'matched_train': len(train), 'matched_val': len(val), 'iterations': args.iterations, 'seed': args.seed,
              'original': original, 'baseline_reproduced': True, 'test_set_read': False,
              'limitations': ['Permutation creates unnatural modality combinations and measures predictive reliance, not causal or clinical importance.', 'Validation set previously used for model selection; exploratory analysis only.', 'Same patient does not establish same imaging session.'],
              'permutations': {}}
    for name, scores in raw.items():
        out = {}
        for metric in ('macro_f1', 'roc_auc', 'accuracy'):
            vals = np.array([m[metric] for m in scores])
            drops = original[metric] - vals
            out[metric] = {'permuted_mean': float(vals.mean()), 'permuted_sd': float(vals.std()),
                           'mean_drop': float(drops.mean()), 'drop_95_percentile_interval': [float(v) for v in np.quantile(drops, [.025, .975])]}
        result['permutations'][name] = out
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'modality_ablation_summary.json').write_text(json.dumps(result, indent=2))
    with (args.out/'modality_ablation_iterations.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['modality', 'iteration', 'macro_f1', 'roc_auc', 'accuracy'])
        for name, scores in raw.items():
            for i, m in enumerate(scores, 1):
                writer.writerow([name, i, m['macro_f1'], m['roc_auc'], m['accuracy']])
    print('Baseline reproduced:', original)
    for name, data in result['permutations'].items():
        print(f"{name} permutation: mean F1 drop={data['macro_f1']['mean_drop']:.4f}, mean AUC drop={data['roc_auc']['mean_drop']:.4f}")
    print('DONE:', args.out/'modality_ablation_summary.json')


if __name__ == '__main__':
    main()
