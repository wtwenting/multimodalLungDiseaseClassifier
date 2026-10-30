#!/usr/bin/env python3
"""Stage 5G: fast exploratory lung-air coverage triage (NOT validated segmentation).
Reads only training candidate CTs; never changes original files or manifests.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HIGH_RISK = ['S03048', 'S03074', 'S03259', 'S03834', 'S04583', 'S05295']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--patient-ids', nargs='+', default=HIGH_RISK)
    p.add_argument('--max-per-patient', type=int, default=4)
    p.add_argument('--max-side', type=int, default=160)
    p.add_argument('--preview-count', type=int, default=12)
    return p.parse_args()


def analyze(path, max_side, preview=False):
    import nibabel as nib
    from scipy import ndimage as ndi
    nii = nib.as_closest_canonical(nib.load(str(path)))
    if len(nii.shape) != 3:
        raise ValueError('Not a 3D volume')
    shape = np.asarray(nii.shape)
    # Read strided voxels to reduce memory; keep the affine-consistent orientation.
    stride = np.maximum(1, np.ceil(shape / max_side).astype(int))
    v = np.asarray(nii.dataobj[::stride[0], ::stride[1], ::stride[2]], dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite intensity values')
    spacing = np.linalg.norm(nii.affine[:3, :3], axis=0) * stride
    # In RAS, array axes are X (left-right), Y (posterior-anterior), Z (inferior-superior).
    # Use a robust body silhouette from intensity, then seek air pockets inside it.
    # This assumes approximate HU and is only a heuristic, not a validated lung mask.
    air = (v >= -1000) & (v <= -300)
    body = v > -500
    body = ndi.binary_closing(body, iterations=2)
    body = ndi.binary_fill_holes(body)
    # Per axial slice: fill external body contour, then detect internal low-density regions.
    internal_air = np.zeros(v.shape, dtype=bool)
    for k in range(v.shape[2]):
        silhouette = ndi.binary_fill_holes(ndi.binary_closing(body[:, :, k], iterations=3))
        internal_air[:, :, k] = air[:, :, k] & silhouette
    # Restrict to central torso region; keep this as a conservative proxy, not a diagnosis.
    x0, x1 = int(v.shape[0]*.12), int(v.shape[0]*.88)
    y0, y1 = int(v.shape[1]*.12), int(v.shape[1]*.88)
    internal_air[:x0] = False; internal_air[x1:] = False
    internal_air[:, :y0] = False; internal_air[:, y1:] = False
    slice_voxels = internal_air.sum(axis=(0, 1))
    # At least 0.5% of the in-plane pixels, and at least 10 pixels.
    cutoff = max(10, int(v.shape[0]*v.shape[1]*.005))
    present = slice_voxels >= cutoff
    idx = np.flatnonzero(present)
    lung_span = float((idx[-1] - idx[0] + 1) * spacing[2]) if len(idx) else 0.0
    coverage = float(v.shape[2] * spacing[2])
    result = {
        'original_shape': 'x'.join(map(str, nii.shape)),
        'orientation': ''.join(nib.aff2axcodes(nii.affine)),
        'sampled_shape': 'x'.join(map(str, v.shape)),
        'si_extent_mm': round(coverage, 2),
        'air_proxy_slice_ratio': round(float(present.mean()), 4),
        'air_proxy_extent_mm': round(lung_span, 2),
        'air_proxy_voxel_fraction': round(float(internal_air.mean()), 5),
        'air_proxy_touches_end': bool(len(idx) and (idx[0] <= 1 or idx[-1] >= v.shape[2]-2)),
        'intensity_p01': round(float(np.percentile(v, 1)), 2),
        'intensity_p99': round(float(np.percentile(v, 99)), 2),
        'warning': 'Air proxy is NOT lung segmentation; non-HU scans or orientation errors can invalidate metrics.'
    }
    return result, (v, internal_air) if preview else None


def make_preview(v, mask, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = v.shape[2] // 2
    img = np.clip((v[:, :, k] + 1000) / 1400, 0, 1)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].imshow(img.T, origin='lower', cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Mid axial (display window)')
    axs[1].imshow(img.T, origin='lower', cmap='gray', vmin=0, vmax=1)
    axs[1].imshow(np.ma.masked_where(~mask[:, :, k].T, mask[:, :, k].T), origin='lower', cmap='autumn', alpha=.45)
    axs[1].set_title('Air proxy overlay (NOT lung mask)')
    for ax in axs: ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.queue.open(newline='', encoding='utf-8-sig') as f:
        queue = list(csv.DictReader(f))
    selected = []
    for pid in args.patient_ids:
        options = [r for r in queue if r.get('patient_id') == pid and r.get('split') == 'train' and r.get('geometry_flag') == 'HEADER_PASS']
        options.sort(key=lambda r: (r.get('is_v3_selected') != 'YES', -(float(r.get('si_coverage_mm') or 0))))
        selected.extend(options[:args.max_per_patient])
    rows, failures = [], []
    previews = args.output / 'previews'
    previews.mkdir(exist_ok=True)
    for i, r in enumerate(selected):
        path = Path(r['ct_path'])
        try:
            if not path.is_file():
                raise FileNotFoundError(str(path))
            stats, imgs = analyze(path, args.max_side, preview=i < args.preview_count)
            record = {k: r.get(k, '') for k in ['patient_id', 'split', 'ct_path', 'is_v3_selected', 'geometry_flag', 'si_spacing_mm']}
            record.update(stats)
            rows.append(record)
            if imgs is not None:
                make_preview(*imgs, previews / f'{i+1:03d}_{r["patient_id"]}_{path.name.replace(".nii.gz", "")}.png', r['patient_id'] + ' ' + path.name)
            print(f'[{i+1}/{len(selected)}] {r["patient_id"]} OK', flush=True)
        except Exception as e:
            failures.append({'patient_id': r.get('patient_id'), 'ct_path': str(path), 'error': repr(e)})
            print(f'[{i+1}/{len(selected)}] {r.get("patient_id")} FAILED: {e}', flush=True)
    def write_csv(path, data):
        with path.open('w', newline='', encoding='utf-8') as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
            else:
                f.write('patient_id,ct_path,error\n')
    write_csv(args.output / 'ct_stage5g_metrics.csv', rows)
    write_csv(args.output / 'ct_stage5g_failures.csv', failures)
    summary = {'mode': 'TRAIN_ONLY_AIR_PROXY_TRIAGE', 'requested': len(selected), 'processed': len(rows), 'failures': len(failures), 'patients': len(set(r['patient_id'] for r in rows)), 'previews': min(len(rows), args.preview_count), 'limitations': ['Heuristic air proxy, NOT validated lung segmentation.', 'Do not automatically exclude or rank scans by this metric.', 'HU assumption and anatomical orientation must be verified.', 'No validation/test scans read; no training or manifest modifications.']}
    (args.output / 'ct_stage5g_summary.json').write_text(json.dumps(summary, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
