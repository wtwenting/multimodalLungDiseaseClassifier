#!/usr/bin/env python3
"""Stage 5B: compare four original CT reconstructions for ONE patient.

Read-only local QA. Checks NIfTI geometry and produces orientation-aware
orthogonal views plus a slice montage. No model training or selection changes.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUNS = {
    'run-1': 'sub-S03048_ses-E20387_run-1_bp-chest_ct.nii.gz',
    'run-2': 'sub-S03048_ses-E20387_run-2_bp-chest_ct.nii.gz',
    'acq-2_run-4': 'sub-S03048_ses-E20387_acq-2_run-4_bp-chest_ct.nii.gz',
    'acq-2_run-5': 'sub-S03048_ses-E20387_acq-2_run-5_bp-chest_ct.nii.gz',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct-dir', type=Path, required=True, help='Directory containing the original S03048 NIfTI files')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-display-voxels', type=int, default=512, help='Maximum dimension for preview-only downsampling')
    return p.parse_args()


def safe_preview(a, maxdim):
    # Preview-only decimation; original data is never altered.
    step = max(1, int(np.ceil(max(a.shape) / maxdim)))
    return a[::step, ::step, ::step]


def scaled(a):
    low, high = np.percentile(a, [1, 99])
    if not np.isfinite(low + high) or high <= low:
        raise ValueError('Invalid intensity range')
    return np.clip((a - low) / (high - low), 0, 1), float(low), float(high)


def load_one(name, path, maxdim):
    import nibabel as nib
    nii = nib.load(str(path))
    shape = tuple(int(x) for x in nii.shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError(f'Not a usable 3D volume: {shape}')
    header_spacing = np.array(nii.header.get_zooms()[:3], dtype=float)
    affine_spacing = np.linalg.norm(nii.affine[:3, :3], axis=0)
    consistent = bool(np.allclose(header_spacing, affine_spacing, rtol=0.01, atol=0.01))
    orientation = ''.join(nib.aff2axcodes(nii.affine))
    canonical = nib.as_closest_canonical(nii)
    canonical_orientation = ''.join(nib.aff2axcodes(canonical.affine))
    canonical_spacing = np.linalg.norm(canonical.affine[:3, :3], axis=0)
    # Nibabel's dataobj applies NIfTI scaling; do not manually apply slope/intercept.
    vol = np.asarray(canonical.dataobj, dtype=np.float32)
    preview = safe_preview(vol, maxdim)
    preview, p01, p99 = scaled(preview)
    qform, qcode = nii.get_qform(coded=True)
    sform, scode = nii.get_sform(coded=True)
    qs_conflict = bool(qcode > 0 and scode > 0 and not np.allclose(qform, sform, rtol=1e-3, atol=1e-2))
    record = dict(run=name, file=str(path), original_shape=str(shape), original_orientation=orientation,
                  canonical_shape=str(tuple(canonical.shape)), canonical_orientation=canonical_orientation,
                  header_spacing_mm=str(tuple(np.round(header_spacing, 5))),
                  affine_spacing_mm=str(tuple(np.round(affine_spacing, 5))),
                  spacing_consistent=consistent, qform_code=int(qcode), sform_code=int(scode),
                  qform_sform_conflict=qs_conflict,
                  canonical_extent_mm=str(tuple(np.round(np.array(canonical.shape)*canonical_spacing, 2))),
                  preview_p01=p01, preview_p99=p99)
    return record, preview, canonical_spacing * np.array(canonical.shape)  # xyz


def plot_views(items, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(items), 3, figsize=(13, 4*len(items)), squeeze=False)
    for row, (name, arr, extent) in enumerate(items):
        x, y, z = arr.shape
        ex, ey, ez = extent
        # Canonical RAS XYZ. Transpose to show anatomical axes in expected directions.
        views = [
            (arr[:, :, z//2].T, [-ex/2, ex/2, -ey/2, ey/2], 'Axial (A vs R)'),
            (arr[:, y//2, :].T, [-ex/2, ex/2, -ez/2, ez/2], 'Coronal (S vs R)'),
            (arr[x//2, :, :].T, [-ey/2, ey/2, -ez/2, ez/2], 'Sagittal (S vs A)'),
        ]
        for col, (sl, ext, title) in enumerate(views):
            ax = axs[row, col]
            ax.imshow(sl, cmap='gray', origin='lower', extent=ext, vmin=0, vmax=1, interpolation='nearest')
            ax.set_aspect('equal')
            ax.set_title(f'{name} | {title}', fontsize=10)
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
    fig.suptitle('S03048 — Original NIfTI canonical RAS views (preview only)', fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_axial_montage(items, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fractions = [0.15, 0.3, 0.5, 0.7, 0.85]
    fig, axs = plt.subplots(len(items), len(fractions), figsize=(15, 3.3*len(items)), squeeze=False)
    for i, (name, arr, _) in enumerate(items):
        for j, fraction in enumerate(fractions):
            k = min(arr.shape[2]-1, int((arr.shape[2]-1)*fraction))
            ax = axs[i, j]
            ax.imshow(arr[:, :, k].T, cmap='gray', origin='lower', vmin=0, vmax=1)
            ax.set_title(f'{name} | Z {fraction:.0%}', fontsize=9)
            ax.axis('off')
    fig.suptitle('S03048 — Axial slice progression (inferior to superior)', fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, items, failures = [], [], []
    for name, filename in RUNS.items():
        path = args.ct_dir / filename
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            row, preview, extent = load_one(name, path, args.max_display_voxels)
            rows.append(row)
            items.append((name, preview, extent))
            print(f'OK {name}: {row["original_shape"]}, {row["original_orientation"]} -> {row["canonical_orientation"]}, spacing consistent={row["spacing_consistent"]}', flush=True)
        except Exception as e:
            failures.append({'run': name, 'path': str(path), 'error': str(e)})
            print(f'ERROR {name}: {e}', flush=True)
    if rows:
        with (args.output / 'ct_stage5b_geometry.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_views(items, args.output / 'S03048_four_runs_orthogonal.png')
        plot_axial_montage(items, args.output / 'S03048_four_runs_axial_montage.png')
    with (args.output / 'ct_stage5b_failures.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run','path','error'])
        writer.writeheader()
        writer.writerows(failures)
    summary = {'patient': 'S03048', 'requested': len(RUNS), 'processed': len(rows),
               'failures': len(failures), 'no_training': True, 'no_source_modifications': True,
               'note': 'Metadata agreement does not establish correct anatomical orientation. Inspect the orthogonal and axial montage images.'}
    (args.output / 'ct_stage5b_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
