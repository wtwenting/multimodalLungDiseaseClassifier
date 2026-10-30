#!/usr/bin/env python3
"""Coswara-only HeAR embedding extraction and training-only exploratory analysis.

Dependencies: tensorflow, librosa, numpy, pandas, scikit-learn, matplotlib.
No model fitting or weight updates are performed.
"""
import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

LABELS = {'healthy': 0, 'COVID-19': 1}
SPLITS = ('train', 'validation', 'test')
SAMPLE_RATE = 16000
CLIP_SAMPLES = 32000
EMBED_DIM = 512


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True, help='Existing Coswara heavy-cough split CSV')
    p.add_argument('--model-path', type=Path, required=True, help='HeAR SavedModel directory')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16, help='Number of 2-second clips per inference batch')
    p.add_argument('--top-db', type=float, default=60.0)
    p.add_argument('--limit', type=int, default=0, help='Smoke test: at most N records per split (0=all)')
    p.add_argument('--skip-analysis', action='store_true', help='Only extract and save embeddings')
    return p.parse_args()


def read_records(path, limit):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'uuid', 'status', 'audio_path', 'split'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Metadata is empty or missing required columns: ' + str(required))
    groups = {s: [] for s in SPLITS}
    seen = set()
    for row in rows:
        if row['status'].strip() not in LABELS:
            raise ValueError('Unexpected class: ' + row['status'])
        split = row['split'].strip()
        if split not in groups:
            raise ValueError('Unexpected split: ' + split)
        uid = row['uuid'].strip()
        if uid in seen:
            raise ValueError('Duplicate participant UUID across metadata: ' + uid)
        seen.add(uid)
        if not limit or len(groups[split]) < limit:
            groups[split].append(row)
    for split, group in groups.items():
        if not group:
            raise ValueError('No rows for split ' + split)
    return groups


def audio_clips(path, top_db):
    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError('Empty or nonfinite waveform')
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if y.size == 0 or np.max(np.abs(y)) == 0:
        raise ValueError('Empty or silent waveform after trimming')
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = np.asarray(y, dtype=np.float32)
    n_clips = max(1, math.ceil(len(y) / CLIP_SAMPLES))
    padded = np.pad(y, (0, n_clips * CLIP_SAMPLES - len(y)))
    return padded.reshape(n_clips, CLIP_SAMPLES), len(y) / SAMPLE_RATE


def encode(clips, signature, batch_size):
    vectors = []
    for start in range(0, len(clips), batch_size):
        batch = tf.convert_to_tensor(clips[start:start + batch_size], dtype=tf.float32)
        result = signature(x=batch)['output_0'].numpy()
        if result.ndim != 2 or result.shape[1] != EMBED_DIM:
            raise ValueError('Unexpected HeAR output shape: ' + str(result.shape))
        if not np.isfinite(result).all():
            raise ValueError('HeAR produced NaN/Inf embeddings')
        vectors.append(result)
    return np.concatenate(vectors, axis=0)


def extract(groups, signature, args):
    all_ids = {}
    failures = []
    stats = {}
    for split, records in groups.items():
        embeddings, labels, uuids, clip_counts, durations = [], [], [], [], []
        start_time = time.monotonic()
        for index, row in enumerate(records, 1):
            try:
                clips, duration = audio_clips(Path(row['audio_path']), args.top_db)
                clip_embeddings = encode(clips, signature, args.batch_size)
                pooled = np.mean(clip_embeddings, axis=0)
                embeddings.append(pooled)
                labels.append(LABELS[row['status'].strip()])
                uuids.append(row['uuid'].strip())
                clip_counts.append(len(clips))
                durations.append(duration)
            except Exception as exc:
                failures.append({'split': split, 'uuid': row['uuid'], 'error': str(exc)})
            if index % 50 == 0 or index == len(records):
                print(f'{split}: {index}/{len(records)} processed; {len(embeddings)} successful', flush=True)
        if not embeddings:
            raise RuntimeError('No successful embeddings for split ' + split)
        X = np.stack(embeddings).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        ids = np.asarray(uuids, dtype=str)
        np.savez_compressed(args.output_dir / f'hear_{split}.npz', X=X, y=y, uuid=ids,
                            clip_count=np.asarray(clip_counts, dtype=np.int32),
                            trimmed_duration_seconds=np.asarray(durations, dtype=np.float32))
        all_ids[split] = set(uuids)
        stats[split] = {'requested': len(records), 'processed': len(embeddings),
                        'failed': len(records) - len(embeddings),
                        'class_counts': dict(Counter(row['status'] for row in records)),
                        'successful_label_counts': {'healthy': int((y == 0).sum()),
                                                    'COVID-19': int((y == 1).sum())},
                        'embedding_shape': list(X.shape), 'clips_processed': int(sum(clip_counts)),
                        'elapsed_seconds': round(time.monotonic() - start_time, 2)}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            if all_ids[first] & all_ids[second]:
                raise ValueError('Participant leakage between ' + first + ' and ' + second)
    validation = {'splits': stats, 'total_failed': len(failures), 'failures': failures,
                  'participant_overlap': 0, 'embedding_dimensions': EMBED_DIM,
                  'all_finite': True}
    (args.output_dir / 'hear_feature_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    return validation


def analysis(output_dir):
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    data = np.load(output_dir / 'hear_train.npz')
    X, y = data['X'], data['y']
    if len(np.unique(y)) < 2:
        raise ValueError('Need both classes for exploratory comparisons')
    summary_rows, separation_rows = [], []
    for index in range(X.shape[1]):
        a, b = X[y == 0, index].astype(float), X[y == 1, index].astype(float)
        for name, values in [('healthy', a), ('COVID-19', b)]:
            summary_rows.append({'dimension': index, 'class': name, 'count': len(values),
                                 'mean': float(values.mean()), 'std': float(values.std(ddof=1)),
                                 'median': float(np.median(values)),
                                 'p25': float(np.percentile(values, 25)),
                                 'p75': float(np.percentile(values, 75))})
        pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
        d = (b.mean() - a.mean()) / math.sqrt(pooled_var) if pooled_var > 0 else 0.0
        separation_rows.append({'dimension': index, 'cohens_d_covid_minus_healthy': float(d),
                                'abs_cohens_d': abs(float(d))})
    separation_rows.sort(key=lambda row: row['abs_cohens_d'], reverse=True)
    for filename, rows in [('hear_feature_summary.csv', summary_rows),
                           ('hear_feature_separability.csv', separation_rows)]:
        with (output_dir / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(X)
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, name in [(0, 'Healthy'), (1, 'COVID-19')]:
        mask = y == label
        ax.scatter(points[mask, 0], points[mask, 1], alpha=0.45, s=13, label=name)
    ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
           ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
           title='Coswara HeAR training embeddings: PCA')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / 'hear_pca_train.png', dpi=160)
    plt.close(fig)
    # Silhouette is exploratory, not a downstream classification metric.
    sil = float(silhouette_score(points, y))
    metrics = {'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
               'pca_2d_silhouette_score': sil,
               'top_10_dimensions_by_absolute_cohens_d': separation_rows[:10]}
    (output_dir / 'hear_exploratory_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def report(args, validation, metrics):
    lines = [
        '# Coswara Stage 2B — HeAR Feature Extraction', '',
        '**Scope:** Coswara heavy cough; Healthy vs COVID-19 only.', '',
        '## Method', '',
        f'- SavedModel: `{args.model_path}`',
        f'- Original mono waveform; leading/trailing trim at `top_db={args.top_db}`; resample to 16 kHz.',
        '- Split into non-overlapping 2-second clips (32,000 samples); zero-pad the final partial clip.',
        '- Frozen HeAR `serving_default(x=...)` returns 512-dimensional clip embeddings.',
        '- Mean-pool clip embeddings into one 512-dimensional vector per recording.',
        '- Preserve existing participant-level train/validation/test splits.', '',
        '## Extraction validation', '',
        '| Split | Requested | Successful | Failed | Clips |',
        '|---|---:|---:|---:|---:|',
    ]
    for split, s in validation['splits'].items():
        lines.append(f"| {split} | {s['requested']} | {s['processed']} | {s['failed']} | {s['clips_processed']} |")
    lines.extend(['', f"- Cross-split participant overlap: {validation['participant_overlap']}",
                  f"- Total failures: {validation['total_failed']}", '',
                  '## Exploratory feature analysis (training split only)', ''])
    if metrics:
        lines.extend([f"- PCA variance explained: PC1 {metrics['pca_explained_variance_ratio'][0]:.2%}; PC2 {metrics['pca_explained_variance_ratio'][1]:.2%}.",
                      f"- Silhouette score in 2D PCA: {metrics['pca_2d_silhouette_score']:.4f}.",
                      '- Top embedding dimensions by absolute Cohen d (COVID minus Healthy):', ''])
        for item in metrics['top_10_dimensions_by_absolute_cohens_d'][:5]:
            lines.append(f"  - Dimension {item['dimension']}: d={item['cohens_d_covid_minus_healthy']:.3f}")
    else:
        lines.append('- Skipped via `--skip-analysis`.')
    lines.extend(['', '## Interpretation and limitations', '',
                  '- PCA and univariate effect sizes are exploratory; neither proves diagnostic validity.',
                  '- Learned embedding dimensions have no guaranteed direct physiological interpretation.',
                  '- Mean pooling loses clip ordering and may dilute short cough events.',
                  '- Dataset confounders (recording conditions, devices, demographics) may affect apparent class differences.',
                  '- No classifier was trained; do not interpret this report as classification performance.',
                  '- Select features and hyperparameters using training/validation data, not the held-out test set.', '',
                  '## Stage 3 handoff', '',
                  'Train a downstream classifier on `hear_train.npz`, select using `hear_validation.npz`, and evaluate the final locked model on `hear_test.npz`.',
                  'Compare with the existing Coswara Log-Mel CNN using identical participant splits and Macro F1.', '',
                  '## Files', '',
                  '- `hear_train.npz`, `hear_validation.npz`, `hear_test.npz` (X, y, uuid, clip_count, trimmed_duration_seconds)',
                  '- `hear_feature_config.json`, `hear_feature_validation.json`',
                  '- `hear_feature_summary.csv`, `hear_feature_separability.csv` (when analysis enabled)',
                  '- `hear_exploratory_metrics.json`, `plots/hear_pca_train.png` (when analysis enabled)', ''])
    (args.output_dir / 'hear_stage2_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    args = arguments()
    if args.batch_size < 1 or args.limit < 0:
        raise ValueError('batch-size must be positive and limit nonnegative')
    if not (args.model_path / 'saved_model.pb').exists():
        raise FileNotFoundError('Missing saved_model.pb in model directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = read_records(args.metadata, args.limit)
    config = {'metadata': str(args.metadata), 'model_path': str(args.model_path),
              'model_signature': 'serving_default', 'input_key': 'x', 'output_key': 'output_0',
              'sample_rate_hz': SAMPLE_RATE, 'trim_top_db': args.top_db,
              'clip_duration_seconds': 2, 'clip_samples': CLIP_SAMPLES,
              'clip_overlap': 0, 'final_clip': 'zero_pad', 'aggregation': 'mean',
              'embedding_dimensions': EMBED_DIM, 'batch_size': args.batch_size,
              'limit_per_split': args.limit, 'train_only_exploratory_analysis': True,
              'tensorflow_version': tf.__version__, 'librosa_version': librosa.__version__}
    (args.output_dir / 'hear_feature_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print('Loading HeAR SavedModel (inference only)...', flush=True)
    model = tf.saved_model.load(str(args.model_path))
    signature = model.signatures['serving_default']
    print('Model loaded. Extracting Coswara embeddings...', flush=True)
    validation = extract(groups, signature, args)
    metrics = None if args.skip_analysis else analysis(args.output_dir)
    report(args, validation, metrics)
    print('Complete. Results saved to:', args.output_dir)
    if validation['total_failed']:
        print('WARNING: Some recordings failed; see hear_feature_validation.json')


if __name__ == '__main__':
    main()
