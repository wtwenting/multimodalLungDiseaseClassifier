#!/usr/bin/env python3
"""Stage 5C: training-only CT acquisition/run and anatomical-preview audit.

Read-only. Summarizes inventory metadata for ALL training files, and optionally
renders a reproducible sample of geometry-valid 3D volumes for human review.
Does NOT infer acquisition plane or anatomical correctness automatically.
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def token(v):
    return str(v or '').strip() or 'none'


def group_key(row):
    return f"acq-{token(row.get('acquisition'))}__run-{token(row.get('run'))}"


def make_preview(row, output, maxdim):
    import nibabel as nib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(row['path'])
    nii = nib.load(str(path))
    if len(nii.shape) != 3:
        raise ValueError('Not 3D')
    nii = nib.as_closest_canonical(nii)
    # RAS after canonical conversion; take sparse sample for preview only.
    step = max(1, math.ceil(max(nii.shape) / maxdim))
    a = np.asarray(nii.dataobj[::step, ::step, ::step], dtype=np.float32)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError('Invalid voxel values')
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        raise ValueError('No intensity range')
    a = np.clip((a-lo)/(hi-lo), 0, 1)
    # Display physical aspect using affine-derived spacing.
    sp = np.linalg.norm(nii.affine[:3, :3], axis=0) * step
    ex, ey, ez = np.asarray(a.shape) * sp
    x, y, z = a.shape
    views = [
        (a[:, :, z//2].T, (ex, ey), 'Axial'),
        (a[:, y//2, :].T, (ex, ez), 'Coronal'),
        (a[x//2, :, :].T, (ey, ez), 'Sagittal'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (im, (w,h), title) in zip(axs, views):
        ax.imshow(im, cmap='gray', origin='lower', extent=[0,w,0,h], vmin=0, vmax=1)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('mm')
    fig.suptitle(f"{row['patient_id']} | {group_key(row)} | {path.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=125)
    plt.close(fig)
    return str(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--types', type=Path, required=True, help='Stage 3B ct_image_type_distribution.csv')
    p.add_argument('--manifest', type=Path, required=True, help='Stage 4 V3 selected manifest')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preview-per-group', type=int, default=2)
    p.add_argument('--max-previews', type=int, default=24)
    p.add_argument('--max-preview-dimension', type=int, default=384)
    p.add_argument('--no-previews', action='store_true', help='Metadata report only; no voxel arrays loaded')
    args = p.parse_args()
    if args.preview_per_group < 0 or args.max_previews < 0:
        p.error('Preview counts must be non-negative')
    args.output.mkdir(parents=True, exist_ok=True)
    allrows = read_rows(args.types)
    train = [r for r in allrows if r.get('split') == 'train']
    selected = {r['patient_id']: r for r in read_rows(args.manifest) if r.get('split') == 'train' and r.get('ct_path')}
    for r in train:
        r['_group'] = group_key(r)
        r['_selected'] = selected.get(r['patient_id'], {}).get('ct_path') == r.get('path')
    groups = defaultdict(list)
    for r in train:
        groups[r['_group']].append(r)
    group_rows = []
    for group, rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        group_rows.append({
            'acq_run_group': group, 'files': len(rows),
            'patients': len(set(r['patient_id'] for r in rows)),
            'covid_positive_files': sum(r.get('class_name') == 'covidPositive' for r in rows),
            'covid_negative_files': sum(r.get('class_name') == 'covidNegative' for r in rows),
            'original_candidate_files': sum(r.get('original_status') == 'CANDIDATE_3D' for r in rows),
            'header_pass_files': sum(r.get('geometry_flag') == 'HEADER_PASS' for r in rows),
            'selected_v3_files': sum(r['_selected'] for r in rows),
            'orientations': json.dumps(dict(Counter(r.get('orientation','') for r in rows)), sort_keys=True),
            'resolution_groups': json.dumps(dict(Counter(r.get('resolution_group','') for r in rows)), sort_keys=True),
            'geometry_flags': json.dumps(dict(Counter(r.get('geometry_flag','') for r in rows)), sort_keys=True),
        })
    write_rows(args.output/'ct_stage5c_acq_run_groups.csv', group_rows, list(group_rows[0]) if group_rows else ['acq_run_group'])
    # Patient-level overview: a group is a metadata label, not proof of a separate acquisition.
    patients = defaultdict(list)
    for r in train:
        patients[r['patient_id']].append(r)
    patient_rows = []
    for pid, rows in sorted(patients.items()):
        patient_rows.append({
            'patient_id': pid, 'class_name': rows[0]['class_name'],
            'ct_files': len(rows), 'distinct_sessions': len(set(r.get('session','') for r in rows)),
            'distinct_acq_run_labels': len(set(r['_group'] for r in rows)),
            'header_pass_candidates': sum(r.get('original_status') == 'CANDIDATE_3D' and r.get('geometry_flag') == 'HEADER_PASS' for r in rows),
            'selected_v3': selected.get(pid, {}).get('ct_path', ''),
            'selected_acq_run': next((r['_group'] for r in rows if r['_selected']), ''),
        })
    write_rows(args.output/'ct_stage5c_patient_groups.csv', patient_rows, list(patient_rows[0]) if patient_rows else ['patient_id'])
    # Only training candidates with valid metadata can be previewed.
    eligible = [r for r in train if r.get('original_status') == 'CANDIDATE_3D' and r.get('geometry_flag') == 'HEADER_PASS' and r.get('header_check') == 'OK']
    # Prefer selected V3 examples for each group, then deterministic filename order.
    buckets = defaultdict(list)
    for r in eligible:
        buckets[r['_group']].append(r)
    for g in buckets:
        buckets[g].sort(key=lambda r: (not r['_selected'], r['patient_id'], r['path']))
    picks = []
    seen_paths = set()
    for g in sorted(buckets, key=lambda x: (-len(buckets[x]), x)):
        n = 0
        for r in buckets[g]:
            if n >= args.preview_per_group or len(picks) >= args.max_previews:
                break
            if r['path'] not in seen_paths:
                picks.append(r)
                seen_paths.add(r['path'])
                n += 1
    preview_dir = args.output/'previews'
    preview_rows = []
    if not args.no_previews:
        preview_dir.mkdir(exist_ok=True)
    for i, r in enumerate(picks if not args.no_previews else []):
        safe_pid = re.sub(r'[^A-Za-z0-9_-]', '_', r['patient_id'])
        dest = preview_dir/f'{i+1:03d}_{safe_pid}_{r["_group"]}.png'
        status, err = 'OK', ''
        try:
            make_preview(r, dest, args.max_preview_dimension)
        except Exception as exc:
            status, err = 'FAILED', f'{type(exc).__name__}: {exc}'[:250]
        preview_rows.append({'patient_id':r['patient_id'], 'class_name':r['class_name'],
                             'acq_run_group':r['_group'], 'selected_v3':r['_selected'],
                             'file_path':r['path'], 'preview_path':str(dest) if status=='OK' else '',
                             'status':status, 'error':err, 'manual_anatomy_review':'PENDING',
                             'manual_notes':''})
        print(f'[{i+1}/{len(picks)}] {r["patient_id"]} {r["_group"]}: {status}', flush=True)
    preview_fields = ['patient_id','class_name','acq_run_group','selected_v3','file_path','preview_path','status','error','manual_anatomy_review','manual_notes']
    write_rows(args.output/'ct_stage5c_preview_review.csv', preview_rows, preview_fields)
    summary = {'scope':'TRAIN_ONLY', 'training_files':len(train), 'training_patients':len(patients),
               'acq_run_groups':len(groups), 'geometry_valid_candidate_files':len(eligible),
               'selected_training_patients':len(selected), 'previews_requested':len(picks) if not args.no_previews else 0,
               'previews_successful':sum(r['status']=='OK' for r in preview_rows),
               'previews_failed':sum(r['status']=='FAILED' for r in preview_rows),
               'warnings':['Acq/run filename tokens are NOT confirmed CT reconstruction types.',
                           'Stored NIfTI orientation does NOT prove acquisition plane.',
                           'Manual image review is required; do not auto-exclude by group.',
                           'Preview intensities are per-volume percentile normalized for display only.',
                           'Do not use validation/test files to choose preprocessing rules.',
                           'No training or original-file modifications.']}
    (args.output/'ct_stage5c_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
