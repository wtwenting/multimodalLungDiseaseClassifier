#!/usr/bin/env python3
"""Threshold sweep and five-seed stability for existing image/patient 256D MLPs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

SEEDS = [42, 123, 456, 789, 2026]


def score(y, p, t):
    q = (p >= t).astype(int)
    return dict(threshold=round(float(t), 4), accuracy=float(accuracy_score(y, q)),
                macro_f1=float(f1_score(y, q, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, q)),
                covid_recall=float(recall_score(y, q, pos_label=1, zero_division=0)),
                noncovid_recall=float(recall_score(y, q, pos_label=0, zero_division=0)),
                covid_precision=float(precision_score(y, q, pos_label=1, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)),
                tn=int(confusion_matrix(y, q, labels=[0,1])[0,0]),
                fp=int(confusion_matrix(y, q, labels=[0,1])[0,1]),
                fn=int(confusion_matrix(y, q, labels=[0,1])[1,0]),
                tp=int(confusion_matrix(y, q, labels=[0,1])[1,1]))


def read_predictions(path, model):
    d = pd.read_csv(path)
    if 'label' not in d or 'probability' not in d or 'key' not in d:
        raise ValueError(f'{path} must contain key,label,probability')
    if d.key.duplicated().any() or len(d) != 97 or set(d.label) != {0,1}:
        raise ValueError(f'Unexpected validation cohort in {path}')
    if not d.probability.between(0, 1).all():
        raise ValueError('Invalid probabilities')
    return d[['key','label','probability']].rename(columns={'probability': model})


def sweep(root):
    im = read_predictions(Path('output/xray_mlp_256d/validation_patient_predictions.csv'), 'image_mlp')
    pa = pd.read_csv('output/xray_patient_feature_aggregation/validation_patient_predictions.csv')
    if not {'key','label','mlp_probability'} <= set(pa.columns):
        raise ValueError('Patient MLP prediction columns missing')
    pa = pa[['key','label','mlp_probability']].rename(columns={'mlp_probability':'patient_mlp'})
    if pa.key.duplicated().any():
        raise ValueError('Duplicate patient keys')
    merged = im.merge(pa, on=['key','label'], validate='one_to_one')
    if len(merged) != 97:
        raise ValueError('Image and patient MLP cohorts differ')
    rows = []
    for model in ['image_mlp','patient_mlp']:
        for t in np.arange(.30, .6001, .025):
            rows.append(dict(model=model, **score(merged.label.to_numpy(), merged[model].to_numpy(), t)))
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / 'threshold_sweep.csv', index=False)
    print('THRESHOLD SWEEP DONE', root / 'threshold_sweep.csv', flush=True)
    for model in ['image_mlp','patient_mlp']:
        candidates = [r for r in rows if r['model']==model]
        best = max(candidates, key=lambda r: (r['macro_f1'], r['balanced_accuracy'], -abs(r['threshold']-.5)))
        print('EXPLORATORY BEST', json.dumps(best), flush=True)


def seeds(root, smoke=False):
    configs = [('image_mlp', 'src/train_mlp_256d.py'),
               ('patient_mlp', 'src/benchmark_patient_aggregation.py')]
    for name, src in configs:
        if not Path(src).is_file():
            raise FileNotFoundError(f'Missing existing script {src}')
    if smoke:
        for name, src in configs:
            subprocess.run([sys.executable, src, '--smoke'], check=True)
        print('SMOKE PASS: both feature pipelines validated; no training')
        return
    records = []
    for name, src in configs:
        for seed in SEEDS:
            target = root / 'seeds' / name / str(seed)
            target.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', src, '--seed', str(seed), '--output', str(target)]
            print('RUN', name, 'seed', seed, flush=True)
            with (target / 'training.log').open('w') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} seed={seed} failed; see {target / "training.log"}')
            summary = json.loads((target / 'experiment_summary.json').read_text())
            fixed = summary.get('fixed_validation') or summary['models']['patient_mlp']['fixed']
            records.append(dict(model=name, seed=seed, **{k:v for k,v in fixed.items() if k != 'confusion_matrix'}))
            pd.DataFrame(records).to_csv(root / 'seed_results.csv', index=False)
            print('RESULT', name, seed, 'macro_f1', fixed['macro_f1'], 'auc', fixed['roc_auc'], flush=True)
    df = pd.DataFrame(records)
    cols = ['accuracy','macro_f1','balanced_accuracy','covid_recall','noncovid_recall','roc_auc']
    stats = df.groupby('model')[cols].agg(['mean','std']).round(5)
    stats.to_csv(root / 'seed_summary.csv')
    print('FIVE-SEED STABILITY DONE', root / 'seed_summary.csv', flush=True)
    print(stats.to_string(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['thresholds','seeds'])
    p.add_argument('--output', default='output/xray_final_checks')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    root = Path(a.output)
    if a.stage == 'thresholds':
        sweep(root)
    else:
        seeds(root, smoke=a.smoke)

if __name__ == '__main__':
    main()
