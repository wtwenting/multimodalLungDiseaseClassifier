#!/usr/bin/env python3
"""Stage 5E: training-only, non-destructive CT technical quality screening.

Heuristic flags are triage signals, NOT validated anatomy or clinical judgments.
Uses V3 patient manifest; never changes selection, splits, or original CTs.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-scans', type=int, default=0, help='0 = all selected training CTs')
    p.add_argument('--preview-count', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--minimum-si-mm', type=float, default=250)
    p.add_argument('--large-si-mm', type=float, default=650)
    p.add_argument('--max-axis-spacing-mm', type=float, default=10)
    p.add_argument('--min-slices', type=int, default=32)
    p.add_argument('--max-slice-cv', type=float, default=2.5,
                   help='Heuristic variation of adjacent slice medians (review only)')
    return p.parse_args()


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'patient_id', 'split', 'ct_path', 'selection_status'}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f'Manifest missing columns: {sorted(required-set(rows[0]))}')
    rows = [r for r in rows if r['split'].strip().lower() == 'train' and r['ct_path'].strip()]
    ids = [r['patient_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate training patient rows: stop to prevent patient overrepresentation')
    return sorted(rows, key=lambda r: r['patient_id'])


def technical_metrics(row, a):
    import nibabel as nib
    path = Path(row['ct_path'])
    if not path.is_file():
        raise FileNotFoundError(path)
    nii = nib.load(str(path))
    if len(nii.shape) != 3:
        raise ValueError(f'Not 3D: {nii.shape}')
    zooms = np.asarray(nii.header.get_zooms()[:3], dtype=float)
    affine_zooms = np.linalg.norm(nii.affine[:3, :3], axis=0)
    if not np.all(np.isfinite(zooms)) or np.any(zooms <= 0):
        raise ValueError('Invalid header voxel spacing')
    if not np.all(np.isfinite(affine_zooms)) or np.any(affine_zooms <= 0):
        raise ValueError('Invalid affine voxel spacing')
    spacing_conflict = not np.allclose(zooms, affine_zooms, rtol=0.01, atol=0.01)
    canonical = nib.as_closest_canonical(nii)
    # nibabel's dataobj applies stored slope/intercept; do not scale manually.
    data = np.asarray(canonical.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError('Nonfinite intensity values')
    # canonical RAS has X,Y,Z axes; Z is superior/inferior for axis-aligned volumes.
    z = int(data.shape[2]); nz = min(15, z)
    idx = np.unique(np.linspace(0, z-1, nz).astype(int))
    # Downsample each axial slice for cheap robust metrics; no model preprocessing.
    slices = data[::max(1, data.shape[0]//128), ::max(1, data.shape[1]//128), :][:, :, idx]
    p01, p50, p99 = np.percentile(slices, [1, 50, 99])
    span = max(float(p99-p01), 1e-6)
    norm = np.clip((slices-p01)/span, 0, 1)
    medians = np.median(norm, axis=(0, 1))
    diffs = np.abs(np.diff(medians))
    slice_cv = float(np.max(diffs) / max(float(np.median(diffs)), 0.01)) if len(diffs) else 0.0
    # Descriptive signals only: intensity/background prevalence and slice variability.
    low_floor_fraction = float(np.mean(slices <= p01 + 1e-5))
    high_fraction = float(np.mean(norm >= 0.99))
    axial_energy = np.mean(np.abs(np.diff(norm, axis=0)), axis=(0, 1))
    median_edge = float(np.median(axial_energy))
    si_mm = float(np.linalg.norm(canonical.affine[:3, 2]) * canonical.shape[2])
    flags = []
    if spacing_conflict: flags.append('HEADER_AFFINE_CONFLICT')
    if np.max(zooms) > a.max_axis_spacing_mm: flags.append('LARGE_VOXEL_SPACING')
    if min(data.shape) < a.min_slices: flags.append('LOW_AXIS_DEPTH')
    if si_mm < a.minimum_si_mm: flags.append('LOW_SI_COVERAGE')
    if si_mm > a.large_si_mm: flags.append('VERY_LARGE_SI_COVERAGE')
    if slice_cv > a.max_slice_cv: flags.append('SLICE_INTENSITY_DISCONTINUITY')
    if low_floor_fraction > 0.4: flags.append('DOMINANT_LOW_INTENSITY_FLOOR')
    if high_fraction > 0.25: flags.append('DOMINANT_HIGH_INTENSITY_VALUES')
    result = {
        'patient_id': row['patient_id'], 'class_name': row.get('class_name',''),
        'split': row['split'], 'ct_path': str(path),
        'selection_status_v3': row['selection_status'],
        'original_orientation': ''.join(nib.aff2axcodes(nii.affine)),
        'canonical_orientation': ''.join(nib.aff2axcodes(canonical.affine)),
        'canonical_shape': 'x'.join(map(str, canonical.shape)),
        'si_extent_mm': round(si_mm, 3),
        'header_affine_spacing_conflict': spacing_conflict,
        'max_axis_spacing_mm': round(float(np.max(zooms)), 4),
        'sample_p01': round(float(p01), 3),
        'sample_p50': round(float(p50), 3),
        'sample_p99': round(float(p99), 3),
        'low_floor_fraction': round(low_floor_fraction, 5),
        'high_fraction': round(high_fraction, 5),
        'slice_discontinuity_ratio': round(slice_cv, 3),
        'median_axial_edge': round(median_edge, 5),
        'flags': ';'.join(flags),
        'triage': 'REVIEW' if flags else 'NO_HEURISTIC_FLAG',
        'manual_status': 'PENDING',
        'manual_notes': ''
    }
    return result, canonical, data


def save_preview(path, nii, data, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    low, high = np.percentile(data[::4, ::4, ::max(1, data.shape[2]//50)], [1, 99])
    image = np.clip((data-low)/max(high-low, 1e-6), 0, 1)
    # RAS X,Y,Z -> D,H,W (Z,Y,X); display aspect from physical voxel dimensions.
    dhw = np.transpose(image, (2,1,0))
    sx, sy, sz = np.linalg.norm(nii.affine[:3,:3], axis=0)
    views = [
        (dhw[dhw.shape[0]//2], 'Axial', sy/sx),
        (dhw[:,dhw.shape[1]//2], 'Coronal', sz/sx),
        (dhw[:,:,dhw.shape[2]//2], 'Sagittal', sz/sy)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, (img, title, aspect) in zip(axes, views):
        ax.imshow(img, cmap='gray', origin='lower', aspect=aspect, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(result['patient_id']+' | '+result['triage']+' | '+result['flags'][:100], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    a = args_parser()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(a.manifest)
    total_eligible = len(rows)
    if a.max_scans > 0: rows = rows[:a.max_scans]
    print(f'Training-only: processing {len(rows)} of {total_eligible} selected CTs', flush=True)
    results, failures, previews = [], [], []
    for i, row in enumerate(rows, 1):
        try:
            result, nii, data = technical_metrics(row, a)
            results.append(result)
            if len(previews) < a.preview_count:
                # Collect first flagged scans; fallback to unflagged samples after scan.
                if result['flags']:

                    preview_dir = a.output/'previews'
                    preview_dir.mkdir(exist_ok=True)
                    save_preview(preview_dir/(result['patient_id']+'_stage5e.png'), nii, data, result)
                    previews.append(result['patient_id'])
            if i % 10 == 0 or i == len(rows):
                print(f'{i}/{len(rows)} processed; flagged={sum(bool(r["flags"]) for r in results)}; errors={len(failures)}', flush=True)
        except Exception as e:
            failures.append({'patient_id': row['patient_id'], 'ct_path': row['ct_path'], 'error': f'{type(e).__name__}: {e}'})
    preview_dir = a.output/'previews'
    preview_dir.mkdir(exist_ok=True)
    fields = ['patient_id','class_name','split','ct_path','selection_status_v3',
              'original_orientation','canonical_orientation','canonical_shape','si_extent_mm',
              'header_affine_spacing_conflict','max_axis_spacing_mm','sample_p01','sample_p50',
              'sample_p99','low_floor_fraction','high_fraction','slice_discontinuity_ratio',
              'median_axial_edge','flags','triage','manual_status','manual_notes']
    write_csv(a.output/'ct_stage5e_screening.csv', results, fields)
    write_csv(a.output/'ct_stage5e_failures.csv', failures, ['patient_id','ct_path','error'])
    summary = {
        'mode':'TRAIN_ONLY_HEURISTIC_SCREENING', 'total_training_selected_in_manifest':total_eligible,
        'attempted':len(rows), 'processed':len(results), 'failures':len(failures),
        'triage_counts':dict(Counter(r['triage'] for r in results)),
        'flag_counts':dict(Counter(f for r in results for f in r['flags'].split(';') if f)),
        'previews_written':len(previews),
        'parameters': {'minimum_si_mm':a.minimum_si_mm, 'large_si_mm':a.large_si_mm,
                       'max_axis_spacing_mm':a.max_axis_spacing_mm,
                       'min_slices':a.min_slices,'max_slice_cv':a.max_slice_cv},
        'cautions': [
            'NO_HEURISTIC_FLAG does not mean anatomically correct or clinically acceptable.',
            'Orientation mislabeling cannot be reliably detected by these heuristic metrics.',
            'Thresholds are exploratory and require visual validation on training patients.',
            'Only training CTs are decoded; no validation/test image data used for decisions.',
            'No CT files, splits, or V3 manifest changed; no model training performed.'
        ]}
    (a.output/'ct_stage5e_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
