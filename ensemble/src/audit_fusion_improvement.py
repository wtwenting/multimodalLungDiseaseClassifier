"""Paired patient bootstrap for CT B + X-ray fusion vs unimodal baselines.

Run from ensemble/: python src/audit_fusion_improvement.py
Uses saved validation predictions only. Never reads test data or refits models.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

MODELS = ('CT_B_only', 'Xray_only', 'CT_B_plus_Xray')


def read_predictions(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for r in rows:
        pid = r['patient_id'].strip().upper()
        if not pid or pid in result:
            raise ValueError(f'Empty/duplicate patient ID in {path}: {pid!r}')
        label = int(r['label'])
        probability = float(r['probability_positive'])
        if label not in (0, 1) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f'Invalid label/probability for {pid} in {path}')
        result[pid] = (label, probability)
    if not result:
        raise ValueError(f'No predictions in {path}')
    return result


def metric(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', type=Path, default=Path('output/fusion_lr_overnight'))
    ap.add_argument('--out', type=Path, default=Path('output/fusion_paired_bootstrap'))
    ap.add_argument('--iterations', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.iterations < 100:
        ap.error('--iterations must be >= 100')
    maps = {name: read_predictions(args.predictions / f'{name}_validation_predictions.csv') for name in MODELS}
    ids = sorted(maps[MODELS[0]])
    if any(set(m) != set(ids) for m in maps.values()):
        raise ValueError('Prediction files have different patient sets. Aborting paired analysis.')
    y = np.asarray([maps[MODELS[0]][pid][0] for pid in ids], dtype=int)
    if any(maps[name][pid][0] != int(label) for name in MODELS for pid, label in zip(ids, y)):
        raise ValueError('Labels disagree across prediction files')
    if len(np.unique(y)) != 2:
        raise ValueError('Validation must contain both classes')
    probs = {name: np.asarray([maps[name][pid][1] for pid in ids]) for name in MODELS}
    comparisons = [('CT_B_plus_Xray', 'CT_B_only'), ('CT_B_plus_Xray', 'Xray_only')]
    rng = np.random.default_rng(args.seed)
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    observed = {name: metric(y, p) for name, p in probs.items()}
    diffs = {f'{a}_vs_{b}': {'macro_f1': [], 'roc_auc': []} for a, b in comparisons}
    for _ in range(args.iterations):
        ix = np.concatenate((rng.choice(neg, size=len(neg), replace=True), rng.choice(pos, size=len(pos), replace=True)))
        yi = y[ix]
        sampled = {name: metric(yi, p[ix]) for name, p in probs.items()}
        for a, b in comparisons:
            for k in ('macro_f1', 'roc_auc'):
                diffs[f'{a}_vs_{b}'][k].append(sampled[a][k] - sampled[b][k])
    results = {}
    for a, b in comparisons:
        key = f'{a}_vs_{b}'
        results[key] = {}
        for metric_name, samples in diffs[key].items():
            samples = np.asarray(samples)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            results[key][metric_name] = {
                'observed_difference': round(observed[a][metric_name] - observed[b][metric_name], 6),
                'bootstrap_95_percentile_interval': [round(float(lo), 6), round(float(hi), 6)],
                'bootstrap_fraction_difference_positive': round(float(np.mean(samples > 0)), 4),
            }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'paired_bootstrap_summary.json').write_text(json.dumps({
        'n_validation_patients': len(ids), 'class_counts': {'negative': len(neg), 'positive': len(pos)},
        'iterations': args.iterations, 'seed': args.seed,
        'method': 'Stratified paired patient bootstrap, percentile 95% intervals, threshold 0.5',
        'model_metrics': observed, 'comparisons': results,
        'limitations': ['Exploratory uncertainty intervals on an already-used validation cohort; not a fresh test.',
                        'Does not validate CT/X-ray session timing or upstream encoder patient independence.',
                        'No model fitting, test-set reading, or hyperparameter selection.'],
    }, indent=2) + '\n', encoding='utf-8')
    with (args.out / 'paired_patient_errors.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'label', 'ct_b_probability', 'fusion_probability', 'ct_b_predicted', 'fusion_predicted', 'ct_b_correct', 'fusion_correct'])
        for i, pid in enumerate(ids):
            pb, pf = probs['CT_B_only'][i], probs['CT_B_plus_Xray'][i]
            cb, cf = int(pb >= 0.5), int(pf >= 0.5)
            w.writerow([pid, int(y[i]), round(float(pb), 6), round(float(pf), 6), cb, cf, int(cb == y[i]), int(cf == y[i])])
    print(json.dumps(results, indent=2))
    print(f'Saved: {args.out / "paired_bootstrap_summary.json"}')
    print(f'Saved: {args.out / "paired_patient_errors.csv"}')


if __name__ == '__main__':
    main()
