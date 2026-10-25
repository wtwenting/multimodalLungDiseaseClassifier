#!/usr/bin/env python3
"""Train a small MLP on aligned cached B+C X-ray features; no test loading."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return dict(threshold=round(float(t), 6), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                covid_recall=float(recall_score(y, pred, pos_label=1, zero_division=0)),
                noncovid_recall=float(recall_score(y, pred, pos_label=0, zero_division=0)),
                covid_precision=float(precision_score(y, pred, pos_label=1, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', type=Path, default=Path('output/xray_modelc_cleaned_v2'))
    ap.add_argument('--comparison', type=Path, default=Path('output/xray_modelb_vs_modelc_cleaned'))
    ap.add_argument('--output', type=Path, default=Path('output/xray_mlp_256d'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--threshold', type=float, default=0.425)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    clean = pd.read_csv(a.clean / 'clean_manifest.csv')
    idx = pd.read_csv(a.clean / 'modelC_features_index.csv')
    cols = ['image_id', 'key', 'patient_id', 'label', 'split', 'png_path']
    if any(c not in clean.columns or c not in idx.columns for c in cols):
        raise ValueError('Missing required manifest/index columns')
    if not clean[cols].reset_index(drop=True).equals(idx[cols].reset_index(drop=True)):
        raise ValueError('Feature index and cleaned manifest row order differ')
    if clean.image_id.duplicated().any() or set(clean.split.unique()) != {'train', 'val'}:
        raise ValueError('Duplicate images or unexpected split; test must not be loaded')
    if clean.groupby('patient_id').split.nunique().max() != 1 or clean.groupby('patient_id').label.nunique().max() != 1:
        raise ValueError('Patient leakage or conflicting labels')
    final = json.loads((a.clean / 'final_summary.json').read_text())
    if final.get('review_pending', -1) != 0:
        raise ValueError('Cleaning not finalized')
    xc = np.load(a.clean / 'modelC_features.npy', allow_pickle=False)
    comp = json.loads((a.comparison / 'experiment_summary.json').read_text())
    if comp.get('cohort_images') != len(clean) or comp.get('cohort_patients') != clean.key.nunique():
        raise ValueError('Model B comparison used different cohort')
    cache = a.comparison / 'features' / comp['model_b_cache_tag']
    xb = np.stack([np.load(cache / (str(i) + '.npy'), allow_pickle=False) for i in clean.image_id])
    if xc.shape != (len(clean), 128) or xb.shape != (len(clean), 128):
        raise ValueError('Expected aligned 128D features from both encoders')
    X = np.concatenate([xb, xc], axis=1).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite feature values')
    tr = clean.split.eq('train').to_numpy()
    va = clean.split.eq('val').to_numpy()
    ytr = clean.loc[tr, 'label'].to_numpy(dtype=np.float32)
    val_frame = clean.loc[va, ['key', 'image_id', 'label']].copy().reset_index(drop=True)
    yval_patient = val_frame.groupby('key', sort=True).label.first().to_numpy(dtype=int)
    if len(np.unique(ytr)) != 2 or len(np.unique(yval_patient)) != 2:
        raise ValueError('Both classes required in train and validation')
    print('COHORT', json.dumps(dict(train_images=int(tr.sum()), val_images=int(va.sum()),
                                    train_patients=int(clean.loc[tr, 'key'].nunique()),
                                    val_patients=int(clean.loc[va, 'key'].nunique()),
                                    features=256, test='NEVER LOADED')), flush=True)
    if a.smoke:
        print('SMOKE PASS: features aligned; no fitting or test loading', flush=True)
        return

    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    scaler = StandardScaler()
    xtr = scaler.fit_transform(X[tr]).astype(np.float32)
    xval = scaler.transform(X[va]).astype(np.float32)
    counts = clean.loc[tr].groupby('key').image_id.transform('size').to_numpy()
    weights = (1.0 / counts).astype(np.float32)
    weights *= len(weights) / weights.sum()

    reg = tf.keras.regularizers.l2(1e-4)
    net = tf.keras.Sequential([
        tf.keras.Input(shape=(256,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=a.learning_rate),
                loss='binary_crossentropy')

    a.output.mkdir(parents=True, exist_ok=True)
    from joblib import dump
    history = []
    best_score = -1.0
    best_epoch = 0
    bad_epochs = 0
    # Use fixed threshold for epoch selection, avoiding per-epoch threshold optimization.
    for epoch in range(1, a.epochs + 1):
        h = net.fit(xtr, ytr, sample_weight=weights, batch_size=a.batch_size,
                    epochs=1, verbose=0, shuffle=True)
        val_frame['probability'] = net.predict(xval, batch_size=64, verbose=0).reshape(-1)
        pat = val_frame.groupby('key', sort=True).agg(label=('label', 'first'),
                                                        probability=('probability', 'mean'))
        m = metrics(pat.label.to_numpy(), pat.probability.to_numpy(), a.threshold)
        history.append(dict(epoch=epoch, train_loss=float(h.history['loss'][0]), **{k:v for k,v in m.items() if k!='confusion_matrix'}))
        print(f"epoch={epoch:03d} loss={history[-1]['train_loss']:.4f} val_macro_f1={m['macro_f1']:.4f} val_auc={m['roc_auc']:.4f}", flush=True)
        if m['macro_f1'] > best_score + 1e-8:
            best_score, best_epoch, bad_epochs = m['macro_f1'], epoch, 0
            net.save(a.output / 'best_mlp.keras')
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f'EARLY STOP: no fixed-threshold macro F1 improvement for {a.patience} epochs', flush=True)
                break
    net = tf.keras.models.load_model(a.output / 'best_mlp.keras')
    val_frame['probability'] = net.predict(xval, batch_size=64, verbose=0).reshape(-1)
    pat = val_frame.groupby('key', sort=True).agg(label=('label', 'first'),
        probability=('probability', 'mean'), images=('image_id', 'size')).reset_index()
    y = pat.label.to_numpy(dtype=int)
    prob = pat.probability.to_numpy()
    fixed = metrics(y, prob, a.threshold)
    tuned = max((metrics(y, prob, t) for t in np.arange(0.2, 0.80001, 0.025)),
                key=lambda m: (m['macro_f1'], m['balanced_accuracy'], -abs(m['threshold'] - .5)))
    val_frame.to_csv(a.output / 'validation_image_predictions.csv', index=False)
    pat.to_csv(a.output / 'validation_patient_predictions.csv', index=False)
    pd.DataFrame(history).to_csv(a.output / 'training_history.csv', index=False)
    dump(scaler, a.output / 'train_fitted_scaler.joblib')
    summary = dict(model='B+C frozen 256D -> 64/32 MLP', best_epoch=best_epoch,
                   train_images=int(tr.sum()), val_images=int(va.sum()),
                   train_patients=int(clean.loc[tr, 'key'].nunique()),
                   val_patients=len(pat), fixed_validation=fixed,
                   tuned_validation_exploratory=tuned, test='NEVER LOADED_OR_EVALUATED',
                   note='Validation used for early stopping and threshold exploration; final test remains untouched.')
    (a.output / 'experiment_summary.json').write_text(json.dumps(summary, indent=2))
    print('FIXED RESULT', json.dumps(fixed), flush=True)
    print('TUNED RESULT exploratory', json.dumps(tuned), flush=True)
    print('DONE', a.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
