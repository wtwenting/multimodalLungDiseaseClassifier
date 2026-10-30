"""BIMCV CT V5: lung-guided, physical-space CT preparation (NO TRAINING).
Run from BIMCV/CT or its parent. Source NIfTI files are read-only.
Dependencies: numpy, scipy, nibabel, SimpleITK, lungmask, torch, matplotlib.

Output shape (Z,Y,X)=(64,192,192), fixed RAS physical grid at 5x2x2 mm.
Every source-to-output affine and QC measurement is recorded for fusion reuse.
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

VERSION = 'ct_v5_lungmask_ras_grid_1'
SHAPE = (64, 192, 192)  # Z,Y,X
SPACING_ZYX = np.array([5., 2., 2.])
WINDOW = (-1000., 400.)
MIN_RETENTION = .985  # estimated from downsampled segmentation, not ground truth


def csv_read(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(path, records):
    if not records:
        return
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    os.replace(tmp, path)


def locate_root(start):
    start = Path(start).resolve()
    for p in (start, start / 'CT'):
        if (p / 'ct_chest_v3_preprocessed' / 'patient_manifest.csv').exists():
            return p
    raise FileNotFoundError('Cannot locate CT/ct_chest_v3_preprocessed/patient_manifest.csv')


def source_candidates(root, row, series):
    # Start with previously selected study, then other previously screened candidates.
    label, patient = row['label'], row['patient']
    paths = [Path(row['source_path'])]
    extra = [r for r in series.get((label, patient), []) if r.get('status') == 'chest_candidate']
    extra.sort(key=lambda r: float(r.get('quality') or 0), reverse=True)
    paths += [Path(r['path']) for r in extra]
    seen = set()
    for p in paths:
        if not p.exists() or str(p).lower() in seen:
            continue
        seen.add(str(p).lower())
        yield p


def process_source(path, model):
    import SimpleITK as sitk
    img = nib.as_closest_canonical(nib.load(str(path)))
    if len(img.shape) != 3:
        raise ValueError('not_3d')
    source_spacing = np.asarray(img.header.get_zooms()[:3], float)
    if np.any(~np.isfinite(source_spacing)) or np.any(source_spacing <= 0):
        raise ValueError('invalid_spacing')
    # Bound segmentation memory; preserve affine/world mapping via index scaling.
    factors_xyz = np.minimum(1., source_spacing / np.array([2., 2., 2.5]))
    small_shape = np.maximum(1, np.rint(np.array(img.shape) * factors_xyz).astype(int))
    if np.prod(small_shape) > 25_000_000:
        raise ValueError('segmentation_volume_too_large')
    raw = np.asarray(img.dataobj, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite_source')
    # NIfTI get_fdata/dataobj applies slope/intercept when present.
    small_xyz = ndi.zoom(raw, small_shape / np.array(raw.shape), order=1, prefilter=False)
    small_zyx = np.transpose(small_xyz, (2, 1, 0))
    del small_xyz
    seg_spacing_xyz = source_spacing * np.array(img.shape) / np.array(small_shape)
    itk = sitk.GetImageFromArray(small_zyx)
    itk.SetSpacing(tuple(float(x) for x in seg_spacing_xyz))
    mask = model.apply(itk)
    if mask.shape != small_zyx.shape:
        raise ValueError('mask_shape_mismatch')
    # Map sampled mask voxels to RAS world coordinates (mask Z,Y,X -> NIfTI X,Y,Z).
    coords = np.argwhere(mask > 0)
    if len(coords) < 1000:
        raise ValueError('no_lung_mask')
    labels = np.unique(mask[mask > 0])
    if len(labels) < 2:
        raise ValueError('missing_bilateral_mask')
    # Use all mask voxels for crop retention, subsample only to estimate crop centre.
    xyz = coords[:, ::-1].astype(np.float64)
    xyz *= (np.array(img.shape) / np.array(small_shape))[None, :]
    world = nib.affines.apply_affine(img.affine, xyz)
    # Robust center: bbox midpoint, not average of differently sized lungs.
    center = (world.min(axis=0) + world.max(axis=0)) / 2
    # Fixed physical FOV, output is Z,Y,X; output voxel centers in RAS.
    # Output index [z,y,x] -> RAS [x,y,z].
    out_to_world = np.array([[0,0,SPACING_ZYX[2]],
                             [0,SPACING_ZYX[1],0],
                             [SPACING_ZYX[0],0,0]], dtype=np.float64)
    offset_world = center - out_to_world @ ((np.array(SHAPE)-1)/2)
    out_to_source = np.linalg.inv(img.affine)[:3,:3] @ out_to_world
    source_offset = (np.linalg.inv(img.affine) @ np.r_[offset_world, 1])[:3]
    # Fixed grid directly from source; avoids intermediate center crop and resizing.
    output = ndi.affine_transform(raw, out_to_source, offset=source_offset,
                                  output_shape=SHAPE, order=1, mode='constant',
                                  cval=-1024., prefilter=False)
    del raw
    normalized = ((np.clip(output, *WINDOW) - WINDOW[0]) /
                  (WINDOW[1] - WINDOW[0])).astype(np.float32)
    del output
    # World -> output grid; check how much segmented anatomy fits fixed physical FOV.
    output_idx = (world - offset_world) @ np.linalg.inv(out_to_world).T
    inside = np.all((output_idx >= 0) & (output_idx <= (np.array(SHAPE)-1)), axis=1)
    retention = float(inside.mean())
    edge_frac = float(np.mean(np.any((output_idx < 2) |
                                       (output_idx > np.array(SHAPE)-3), axis=1)))
    affine_out_zyx = np.eye(4)
    affine_out_zyx[:3,:3] = out_to_world
    affine_out_zyx[:3,3] = offset_world
    qc = {'mask_retention_estimate': round(retention, 5),
          'mask_boundary_fraction': round(edge_frac, 5),
          'source_shape': list(img.shape),
          'source_spacing_xyz_mm': source_spacing.tolist(),
          'segmented_volume_ml': round(len(coords) * np.prod(seg_spacing_xyz) / 1000, 1),
          'output_spacing_zyx_mm': SPACING_ZYX.tolist(),
          'output_shape_zyx': list(SHAPE),
          'output_zyx_to_ras_affine': affine_out_zyx.tolist(),
          'source_ras_affine': img.affine.tolist(),
          'output_zyx_to_source_xyz_matrix': out_to_source.tolist(),
          'output_zyx_to_source_xyz_offset': source_offset.tolist(),
          'normalized_std': round(float(normalized.std(dtype=np.float64)), 6)}
    qc['status'] = ('PASS' if retention >= MIN_RETENTION and edge_frac < .04
                    and qc['normalized_std'] > .03 else 'REVIEW')
    return normalized, qc


def preview(path, volume, qc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 5, figsize=(14, 6))
    for ax, q in zip(axs.flat, np.linspace(.08, .92, 10)):
        z = int(round(q * (len(volume)-1)))
        ax.imshow(volume[z], cmap='gray', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'z={z}')
        ax.axis('off')
    fig.suptitle(f"{qc['status']} | retention={qc['mask_retention_estimate']:.3f} | {path.stem}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='BIMCV/CT or BIMCV folder')
    p.add_argument('--limit', type=int, default=0, help='First N patients for smoke test')
    p.add_argument('--preview-every', type=int, default=25)
    args = p.parse_args()
    root = locate_root(args.root)
    out = root / 'ct_v5_corrected_preprocessed'
    (out/'tensors').mkdir(parents=True, exist_ok=True)
    (out/'previews').mkdir(exist_ok=True)
    (out/'records').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                        handlers=[logging.FileHandler(out/'progress.log', encoding='utf-8'), logging.StreamHandler()])
    manifest = csv_read(root/'ct_chest_v3_preprocessed'/'patient_manifest.csv')
    manifest = [r for r in manifest if r.get('status') == 'accepted']
    series = defaultdict(list)
    sf = root/'ct_chest_v3_preprocessed'/'series_audit.csv'
    if sf.exists():
        for r in csv_read(sf):
            series[(r['label'],r['patient'])].append(r)
    if args.limit:
        manifest = manifest[:args.limit]
    config = {'version': VERSION, 'output_shape_zyx': SHAPE,
              'spacing_zyx_mm': SPACING_ZYX.tolist(), 'hu_window': WINDOW,
              'normalization': 'clip_then_minmax_0_1',
              'interpolation': 'linear', 'outside_fov_hu': -1024,
              'segmentation': 'lungmask R231', 'min_mask_retention': MIN_RETENTION,
              'note': 'Estimated retention from segmentation; independent visual QC still required.'}
    atomic_json(out/'preprocessing_config.json', config)
    import lungmask
    from lungmask import LMInferer
    model = LMInferer(modelname='R231', fillmodel=None, tqdm_disable=True)
    logging.info('START %s | %s | patients=%s', VERSION, root, len(manifest))
    records = []
    for i, row in enumerate(manifest, 1):
        label, patient = row['label'], row['patient']
        key = f'{label}_{patient}'
        record_path = out/'records'/f'{key}.json'
        if record_path.exists():
            try:
                old = json.loads(record_path.read_text(encoding='utf-8'))
                source = Path(old['source_path'])
                if (old['version'] == VERSION and source.exists() and
                    old['source_size'] == source.stat().st_size and
                    old['source_mtime_ns'] == source.stat().st_mtime_ns and
                    (old['status'] != 'PASS' or Path(old['tensor_path']).exists())):
                    records.append(old)
                    continue
            except (KeyError, ValueError, OSError):
                pass
        selected = None
        failures = []
        for path in source_candidates(root, row, series):
            try:
                vol, qc = process_source(path, model)
                if qc['status'] != 'PASS':
                    failures.append(f'{path.name}: retention={qc["mask_retention_estimate"]}')
                    continue
                tensor_path = out/'tensors'/f'{key}.npz'
                temp = tensor_path.with_suffix('.npz.tmp')
                with temp.open('wb') as f:
                    np.savez_compressed(f, volume=vol.astype(np.float16))
                os.replace(temp, tensor_path)
                selected = {'version': VERSION, 'label': label, 'patient': patient,
                            'status': 'PASS', 'source_path': str(path),
                            'tensor_path': str(tensor_path), 'source_size': path.stat().st_size,
                            'source_mtime_ns': path.stat().st_mtime_ns, **qc}
                if i <= 20 or i % max(args.preview_every, 1) == 0:
                    preview(out/'previews'/f'{key}.png', vol, qc)
                break
            except Exception as exc:
                failures.append(f'{path.name}: {type(exc).__name__}: {exc}')
        if selected is None:
            selected = {'version': VERSION, 'label': label, 'patient': patient,
                        'status': 'REVIEW', 'source_path': row['source_path'],
                        'tensor_path': '', 'source_size': '', 'source_mtime_ns': '',
                        'reason': '; '.join(failures)[:2000] or 'no_candidates'}
        atomic_json(record_path, selected)
        records.append(selected)
        if i % 10 == 0 or i == len(manifest):
            logging.info('%d/%d | %s', i, len(manifest), dict(Counter(r['status'] for r in records)))
            atomic_csv(out/'patient_manifest.csv', records)
            atomic_json(out/'summary.json', {'version': VERSION, 'processed': len(records),
                         'total': len(manifest), 'statuses': dict(Counter(r['status'] for r in records)),
                         'by_label': {label: dict(Counter(r['status'] for r in records if r['label']==label))
                                      for label in ('Positive','Negative')},
                         'training_approved': False})
    logging.info('COMPLETE %s', dict(Counter(r['status'] for r in records)))


if __name__ == '__main__':
    main()
