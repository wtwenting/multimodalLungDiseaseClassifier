#!/usr/bin/env python3
"""Package five previously evaluated BIMCV test patients for GUI demonstration.

Run from ensemble/. Copies original CT and the exact X-ray files accepted by
frozen test preprocessing, plus optional reference features/predictions.
Never downloads, modifies originals, or trains/evaluates a model.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unique_index(path, expected_dim):
    df = pd.read_csv(path, dtype={'patient_id': str})
    if 'patient_id' not in df or df.patient_id.isna().any():
        raise ValueError(f'Invalid index: {path}')
    df.patient_id = df.patient_id.str.strip().str.upper()
    if df.patient_id.duplicated().any():
        raise ValueError(f'Duplicate patient IDs in {path}')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ct-index', type=Path, default=Path('output/ct_test_features/ct_b_test_index.csv'))
    ap.add_argument('--ct-features', type=Path, default=Path('output/ct_test_features/ct_b_test_features_1024d.npy'))
    ap.add_argument('--xray-index', type=Path, default=Path('output/xray_test_features_reviewed/xray_test_index.csv'))
    ap.add_argument('--xray-features', type=Path, default=Path('output/xray_test_features_reviewed/xray_test_features_256d.npy'))
    ap.add_argument('--v5-manifest', type=Path, default=Path('output/ct_v5_corrected_preprocessed/patient_manifest.csv'))
    ap.add_argument('--xray-qc', type=Path, default=Path('output/xray_test_features_reviewed/test_image_qc.csv'))
    ap.add_argument('--predictions', type=Path, default=Path('output/fusion_final_test/test_patient_predictions.csv'))
    ap.add_argument('--out', type=Path, default=Path('output/handover/GUI_Demo_5_Test_Patients.zip'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    ct_idx = unique_index(args.ct_index, 1024)
    xr_idx = unique_index(args.xray_index, 256)
    ct_arr = np.load(args.ct_features, allow_pickle=False)
    xr_arr = np.load(args.xray_features, allow_pickle=False)
    if ct_arr.shape != (len(ct_idx), 1024) or xr_arr.shape != (len(xr_idx), 256):
        raise ValueError('Feature array and index shape mismatch')
    if not np.isfinite(ct_arr).all() or not np.isfinite(xr_arr).all():
        raise ValueError('Non-finite feature values')
    ct_map = {pid: i for i, pid in enumerate(ct_idx.patient_id)}
    xr_map = {pid: i for i, pid in enumerate(xr_idx.patient_id)}
    if 'split' in ct_idx and not ct_idx['split'].eq('test').all():
        raise ValueError('CT index includes non-test rows')
    if 'split' in xr_idx and not xr_idx['split'].eq('test').all():
        raise ValueError('X-ray index includes non-test rows')

    pred = pd.read_csv(args.predictions, dtype={'patient_id': str})
    required = {'patient_id', 'label', 'probability_positive', 'predicted'}
    if not required.issubset(pred.columns):
        raise ValueError(f'Prediction columns missing: {required - set(pred.columns)}')
    pred.patient_id = pred.patient_id.str.strip().str.upper()
    if pred.patient_id.duplicated().any():
        raise ValueError('Duplicate predictions')
    pred_map = pred.set_index('patient_id')
    available = sorted(set(ct_map) & set(xr_map) & set(pred_map.index))
    pos = [p for p in available if int(pred_map.loc[p, 'label']) == 1]
    neg = [p for p in available if int(pred_map.loc[p, 'label']) == 0]
    if len(pos) < 3 or len(neg) < 2:
        raise ValueError('Need at least three positive and two negative test patients')
    rng = random.Random(args.seed)
    selected = rng.sample(pos, 3) + rng.sample(neg, 2)
    rng.shuffle(selected)

    v5 = pd.read_csv(args.v5_manifest, dtype=str).fillna('')
    if not {'patient', 'source_path'}.issubset(v5.columns):
        raise ValueError('V5 manifest must contain patient and source_path')
    v5['pid'] = v5.patient.str.strip().str.upper()
    if v5.pid.duplicated().any():
        raise ValueError('V5 manifest contains multiple rows per patient; resolve selected source first')
    ct_sources = v5.set_index('pid').source_path.to_dict()
    qc = pd.read_csv(args.xray_qc, dtype={'patient_id': str}).fillna('')
    if not {'patient_id', 'source_path', 'status'}.issubset(qc.columns):
        raise ValueError('X-ray QC missing patient_id/source_path/status')
    qc.patient_id = qc.patient_id.str.strip().str.upper()
    accepted = qc[qc.status.isin(['ACCEPT', 'ACCEPT_MANUAL'])]
    manifest_rows = []
    copies = []
    for n, pid in enumerate(selected, 1):
        demo_id = f'patient_{n:02d}'
        if pid not in ct_sources or not ct_sources[pid]:
            raise ValueError(f'Missing V5-selected original CT for {pid}')
        ct_source = Path(ct_sources[pid])
        if not ct_source.is_file():
            raise FileNotFoundError(ct_source)
        xr_rows = accepted[accepted.patient_id.eq(pid)]
        if xr_rows.empty:
            raise ValueError(f'No accepted X-rays for {pid}')
        ct_suffix = '.nii.gz' if ct_source.name.lower().endswith('.nii.gz') else ct_source.suffix.lower()
        if ct_suffix not in ('.nii.gz', '.nii'):
            raise ValueError(f'CT source is not NIfTI: {ct_source}')
        ct_dest = f'{demo_id}/ct/scan{ct_suffix}'
        copies.append((ct_source, ct_dest))
        xr_paths = sorted(set(xr_rows.source_path.astype(str)))
        for j, src in enumerate(xr_paths, 1):
            path = Path(src)
            if not path.is_file():
                raise FileNotFoundError(path)
            ext = path.suffix.lower()
            if ext not in ('.png', '.jpg', '.jpeg'):
                raise ValueError(f'Unexpected X-ray format: {path}')
            copies.append((path, f'{demo_id}/xray/image_{j:02d}{ext}'))
        manifest_rows.append({
            'demo_id': demo_id, 'patient_id': pid,
            'reference_label': int(pred_map.loc[pid, 'label']),
            'reference_probability_positive': float(pred_map.loc[pid, 'probability_positive']),
            'reference_prediction': int(pred_map.loc[pid, 'predicted']),
            'ct_file': ct_dest, 'xray_count': len(xr_paths),
            'ct_feature_row': ct_map[pid], 'xray_feature_row': xr_map[pid],
        })
    total_bytes = sum(src.stat().st_size for src, _ in copies)
    print(f'Selected {len(selected)} test patients; {len(copies)} image files; source bytes={total_bytes:,}')
    print('Patient IDs:', ', '.join(selected))
    if args.dry_run:
        print('DRY RUN COMPLETE: no ZIP created')
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fusion_demo_') as temp:
        root = Path(temp) / 'GUI_Demo_5_Test_Patients'
        root.mkdir()
        checksums = []
        for src, dest in copies:
            target = root / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            checksums.append({'file': dest, 'sha256': sha(target), 'bytes': target.stat().st_size})
        with (root / 'demo_manifest.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
            w.writeheader(); w.writerows(manifest_rows)
        np.save(root / 'ct_reference_features_1024d.npy', np.stack([ct_arr[ct_map[p]] for p in selected]), allow_pickle=False)
        np.save(root / 'xray_reference_features_256d.npy', np.stack([xr_arr[xr_map[p]] for p in selected]), allow_pickle=False)
        (root / 'checksums.json').write_text(json.dumps(checksums, indent=2))
        (root / 'README.md').write_text('''# CT + X-ray GUI demonstration (5 previously evaluated test patients)\n\nThis package is for GUI regression/integration demonstration, **not new independent evaluation**.\n\nEach `patient_XX/ct/scan.nii.gz` (or `.nii`) is the original CT selected by the V5 preprocessing manifest. Each `patient_XX/xray/` contains the original X-rays ACCEPTed in the reviewed test QC. All included X-rays must be used for mean aggregation to reproduce the reference feature vector.\n\n`demo_manifest.csv` maps demo IDs to source patient IDs and includes previously recorded labels, probabilities and predicted classes. **Do not use these reference outputs to tune the model or QC policy.**\n\nReference arrays have five rows in exactly the order of `demo_manifest.csv`: CT (5,1024), X-ray (5,256). To validate the fusion stage, concatenate CT first and X-ray second, then run the existing frozen StandardScaler + LogisticRegression pipeline with threshold 0.5.\n\nFor full end-to-end testing, preprocess the raw images with the frozen V5 CT and original X-ray B/C workflows and compare extracted features to the references. Some original X-rays required label-blind manual QC; the GUI may require the same review decisions to reproduce outputs. CT/X-ray images are patient-linked, not confirmed same-session.\n\nData source: BIMCV. Verify dataset license and redistribution permissions before sharing raw images. Research demonstration only; not for clinical diagnosis.\n''')
        if args.out.exists():
            raise FileExistsError(f'Refusing to overwrite existing ZIP: {args.out}')
        with zipfile.ZipFile(args.out, 'w', allowZip64=True) as z:
            for file in sorted(root.rglob('*')):
                if file.is_file():
                    z.write(file, file.relative_to(root.parent), compress_type=zipfile.ZIP_STORED if file.suffix.lower() in ('.gz', '.png', '.jpg', '.jpeg') else zipfile.ZIP_DEFLATED)
    print('ZIP created:', args.out.resolve())
    print('ZIP size:', f'{args.out.stat().st_size:,}', 'bytes')


if __name__ == '__main__':
    main()
