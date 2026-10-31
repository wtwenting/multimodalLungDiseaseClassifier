"""Read-only CT + X-ray fusion dataset audit. Python 3.9+.
Usage: python audit_fusion_dataset.py "D:\\path\\to\\dataset"
Dependencies: pip install numpy nibabel pillow pydicom
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import pydicom
except ImportError:
    pydicom = None

EXTS = ('.nii.gz', '.nii', '.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
LABELS = {'positive': 'Positive', 'covid': 'Positive', 'covid19': 'Positive',
          'negative': 'Negative', 'normal': 'Negative', 'healthy': 'Healthy',
          'noncovid': 'Negative', 'non-covid': 'Negative'}
PATTERN = re.compile(r'(?i)(?:sub[-_]?s?\d+|patient[-_]?\w+|pat[-_]?\d+)')


def label_from(parts):
    for part in parts:
        key = re.sub(r'[^a-z0-9-]', '', part.lower())
        if key in LABELS:
            return LABELS[key]
    return 'Unknown'


def patient_from(parts):
    for part in parts:
        m = PATTERN.fullmatch(part)
        if m:
            return re.sub(r'[^a-z0-9]', '', part.lower())
    return ''


def modality_from(parts, suffix, dicom_mod=''):
    if suffix in ('.nii', '.nii.gz'):
        return 'CT'
    if dicom_mod in ('CT', 'DX', 'CR'):
        return 'CT' if dicom_mod == 'CT' else 'Xray'
    s = '/'.join(parts).lower()
    if any(k in s for k in ('xray', 'x-ray', 'chest_x', 'cxr', 'radiograph')):
        return 'Xray'
    if any(k in s for k in ('/ct/', '/ct_', 'computed_tomography')):
        return 'CT'
    if suffix in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        return 'Xray (inferred)'
    return 'Unknown'


def quantiles(a):
    a = np.asarray(a)
    if a.size == 0:
        return {}
    if a.size > 100000:
        a = a.ravel()[::max(1, a.size // 100000)]
    else:
        a = a.ravel()
    a = a[np.isfinite(a)]
    if not a.size:
        return {}
    return {k: round(float(v), 2) for k, v in zip(('min', 'p01', 'median', 'p99', 'max'), np.percentile(a, [0, 1, 50, 99, 100]))}


def inspect(path, suffix, parts):
    row = dict(path=str(path), modality='', label=label_from(parts), patient=patient_from(parts),
               format=suffix, shape='', spacing='', orientation='', dtype='', intensity='',
               dicom_modality='', photometric='', error='')
    try:
        if suffix in ('.nii', '.nii.gz'):
            if nib is None:
                raise RuntimeError('nibabel not installed')
            im = nib.load(str(path))
            row.update(shape=str(tuple(im.shape)), spacing=str(tuple(round(float(x), 4) for x in im.header.get_zooms()[:3])),
                       orientation=''.join(nib.aff2axcodes(im.affine)), dtype=str(im.get_data_dtype()))
            # Only a small middle slice; get_fdata applies NIfTI scaling.
            if len(im.shape) >= 3:
                z = im.shape[2] // 2
                arr = np.asarray(im.dataobj[:, :, z])
            else:
                arr = np.asarray(im.dataobj)
            row['intensity'] = json.dumps(quantiles(arr))
        elif suffix in ('.dcm', '.dicom'):
            if pydicom is None:
                raise RuntimeError('pydicom not installed')
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            mod = str(getattr(ds, 'Modality', '')).upper()
            row.update(dicom_modality=mod, shape=str((getattr(ds, 'Rows', ''), getattr(ds, 'Columns', ''), getattr(ds, 'NumberOfFrames', 1))),
                       spacing=str(getattr(ds, 'PixelSpacing', '')), photometric=str(getattr(ds, 'PhotometricInterpretation', '')))
            # Avoid extracting DICOM PatientID (may be identifying); use folder naming only.
        else:
            if Image is None:
                raise RuntimeError('Pillow not installed')
            with Image.open(path) as im:
                row.update(shape=str((im.height, im.width, len(im.getbands()))), dtype=im.mode,
                           intensity=json.dumps(quantiles(np.asarray(im.resize((min(256, im.width), min(256, im.height)))))))
        row['modality'] = modality_from(parts, suffix, row['dicom_modality'])
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {str(exc)[:140]}'
        row['modality'] = modality_from(parts, suffix)
    return row


def main():
    parser = argparse.ArgumentParser(description='Audit CT/X-ray fusion dataset without modifying images')
    parser.add_argument('root', nargs='?', help='Directory containing CT and X-ray data')
    parser.add_argument('--out', help='Output directory (default: sibling folder fusion_audit_output)')
    args = parser.parse_args()
    root = Path(args.root.strip('"')) if args.root else Path(input('Dataset root directory: ').strip().strip('"'))
    if not root.is_dir():
        parser.error(f'Directory not found: {root}')
    root = root.resolve()
    out = Path(args.out).resolve() if args.out else root.parent / 'fusion_audit_output'
    if out == root or root in out.parents:
        parser.error('Output must be outside dataset root')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        suffix = '.nii.gz' if path.name.lower().endswith('.nii.gz') else path.suffix.lower()
        if suffix not in EXTS:
            continue
        row = inspect(path, suffix, path.relative_to(root).parts)
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f'Inspected {len(rows)} files...')
    columns = ['path', 'modality', 'label', 'patient', 'format', 'shape', 'spacing',
               'orientation', 'dtype', 'intensity', 'dicom_modality', 'photometric', 'error']
    with (out / 'fusion_inventory.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    modality = Counter(r['modality'] for r in rows)
    labels = {m: dict(Counter(r['label'] for r in rows if r['modality'] == m)) for m in modality}
    patients = defaultdict(set)
    for r in rows:
        if r['patient']:
            patients[r['modality']].add(r['patient'])
    ct = patients['CT']
    xr = patients['Xray'] | patients['Xray (inferred)']
    unknown_ids = sum(not r['patient'] for r in rows)
    ct_rows = [r for r in rows if r['modality'] == 'CT' and not r['error']]
    xr_rows = [r for r in rows if r['modality'].startswith('Xray') and not r['error']]
    ct_z = Counter()
    ct_orient = Counter()
    for r in ct_rows:
        ct_orient[r['orientation'] or 'Unknown'] += 1
        try:
            dims = eval(r['shape'], {'__builtins__': {}}, {})
            ct_z['1-2' if len(dims) < 3 or dims[2] < 3 else '3-15' if dims[2] < 16 else '16+'] += 1
        except Exception:
            ct_z['Unknown'] += 1
    warnings = []
    if unknown_ids:
        warnings.append(f'{unknown_ids} files lack a recognizable patient ID in folder/file names; patient matching may be incomplete.')
    if not ct or not xr:
        warnings.append('CT/X-ray patient overlap cannot be established reliably from the detected identifiers.')
    if 'Unknown' in modality:
        warnings.append('Some DICOM files have unknown modality.')
    if any(r['label'] == 'Unknown' for r in rows):
        warnings.append('Some files have unknown class labels; inspect naming and label mapping.')
    if any(r['label'] == 'Healthy' for r in rows):
        warnings.append('Healthy is not necessarily equivalent to COVID-negative; verify clinical label definitions.')
    if ct_z['1-2'] or ct_z['3-15']:
        warnings.append('Some CT files are too short for standard 2.5D study processing.')
    summary = dict(root=str(root), total_files=len(rows), modality_counts=dict(modality),
                   labels_by_modality=labels, patients_by_modality={k: len(v) for k, v in patients.items()},
                   matched_ct_xray_patient_ids=len(ct & xr), unmatched_patient_id_files=unknown_ids,
                   ct_slice_groups=dict(ct_z), ct_orientations=dict(ct_orient),
                   ct_shapes_top=Counter(r['shape'] for r in ct_rows).most_common(10),
                   ct_spacing_top=Counter(r['spacing'] for r in ct_rows).most_common(10),
                   xray_shapes_top=Counter(r['shape'] for r in xr_rows).most_common(10),
                   errors_count=sum(bool(r['error']) for r in rows),
                   errors_sample=[{'file': r['path'], 'error': r['error']} for r in rows if r['error']][:10],
                   warnings=warnings,
                   limitations=['Directory labels and patient IDs are inferred from filenames/folders; validate against metadata.',
                                'CT intensities sample only one middle slice and are not proof of correct HU calibration.',
                                'DICOM pixel data and full series reconstruction are not analyzed.',
                                'Metadata compatibility does not establish encoder accuracy or domain generalization.'])
    with (out / 'fusion_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('\n=== COPY FUSION AUDIT SUMMARY BACK TO CHATGPT ===')
    print(json.dumps({k: v for k, v in summary.items() if k != 'root'}, indent=2))
    print(f'\nSaved: {out / "fusion_summary.json"}\nSaved: {out / "fusion_inventory.csv"}')


if __name__ == '__main__':
    main()
