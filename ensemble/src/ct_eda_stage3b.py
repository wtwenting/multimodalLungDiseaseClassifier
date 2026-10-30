#!/usr/bin/env python3
"""CT-only Stage 3B: patient counts, technical image types and geometry audit.

No voxel arrays are loaded. Inventory-only mode uses existing metadata; optional
--check-headers reads original NIfTI headers (requires nibabel and local files).
No patient selection, train/validation/test changes, preprocessing or training.

Run from PsuprProj/ensemble:
 python src/ct_eda_stage3b.py --inventory output/ct3d_audit/ct_inventory.csv --output output/ct_eda_stage3b
 python src/ct_eda_stage3b.py --inventory output/ct3d_audit/ct_inventory.csv --output output/ct_eda_stage3b --check-headers
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def triplet(text, cast=float):
    try:
        v = tuple(cast(x) for x in str(text).split('x'))
        return v if len(v) == 3 else None
    except (TypeError, ValueError):
        return None


def parse_file(path):
    name = Path(path).name
    def part(key):
        m = re.search(r'(?:^|_)' + key + r'-([^_\.]+)', name)
        return m.group(1) if m else ''
    return part('ses'), part('acq'), part('run')


def geometry_from_header(path, nib, np, tol):
    result = {'header_check': 'NOT_CHECKED', 'affine_spacing_mm': '',
              'header_spacing_mm': '', 'affine_orientation': '',
              'header_spacing_match': '', 'qform_sform_match': '',
              'geometry_flag': 'NOT_CHECKED'}
    if not Path(path).is_file():
        result.update(header_check='FILE_MISSING', geometry_flag='FILE_MISSING')
        return result
    try:
        im = nib.load(str(path))
        result['header_check'] = 'OK'
        if len(im.shape) != 3:
            result['geometry_flag'] = 'NON_3D'
            return result
        hs = np.array(im.header.get_zooms()[:3], dtype=float)
        a = np.array(im.affine, dtype=float)
        result['header_spacing_mm'] = 'x'.join(f'{x:.5g}' for x in hs)
        if not np.isfinite(a).all() or not np.isfinite(hs).all() or (hs <= 0).any():
            result['geometry_flag'] = 'NONFINITE_OR_INVALID'
            return result
        asp = np.linalg.norm(a[:3, :3], axis=0)
        result['affine_spacing_mm'] = 'x'.join(f'{x:.5g}' for x in asp)
        result['affine_orientation'] = ''.join(nib.aff2axcodes(a))
        matched = bool(np.allclose(hs, asp, rtol=tol, atol=0.01))
        result['header_spacing_match'] = 'YES' if matched else 'NO'
        q, qc = im.get_qform(coded=True)
        s, sc = im.get_sform(coded=True)
        if qc and sc and q is not None and s is not None:
            qmatch = bool(np.allclose(q, s, rtol=tol, atol=0.1))
            result['qform_sform_match'] = 'YES' if qmatch else 'NO'
        else:
            qmatch = None
            result['qform_sform_match'] = 'ONE_OR_BOTH_UNSET'
        # Flag implausible per-axis scales, but do not claim image corruption.
        result['geometry_flag'] = ('SPACING_CONFLICT' if not matched else
                                   'QFORM_SFORM_CONFLICT' if qmatch is False else
                                   'IMPLAUSIBLE_SPACING' if (asp < 0.1).any() or (asp > 10).any() else
                                   'HEADER_PASS')
    except Exception as e:
        result['header_check'] = 'ERROR:' + type(e).__name__
        result['geometry_flag'] = 'HEADER_ERROR'
    return result


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-headers', action='store_true', help='Read NIfTI headers for ALL listed files; may take time')
    p.add_argument('--root-map', action='append', default=[], metavar='OLD=NEW')
    p.add_argument('--spacing-tolerance', type=float, default=0.01)
    args = p.parse_args()
    maps = []
    for v in args.root_map:
        if '=' not in v:
            p.error('--root-map must be OLD=NEW')
        maps.append(v.split('=', 1))
    with args.inventory.open(newline='', encoding='utf-8-sig') as f:
        inventory = list(csv.DictReader(f))
    required = {'patient_id', 'class_name', 'split', 'path', 'shape', 'spacing', 'orientation', 'status'}
    if not inventory or not required.issubset(inventory[0]):
        p.error('Missing inventory columns: ' + str(sorted(required - set(inventory[0]) if inventory else required)))
    nib = np = None
    if args.check_headers:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            p.error('--check-headers requires: pip install nibabel numpy')
    files = []
    for row in inventory:
        path = row['path']
        for old, new in maps:
            if path.startswith(old):
                path = new + path[len(old):]
                break
        ses, acq, run = parse_file(path)
        shape, spacing = triplet(row['shape'], int), triplet(row['spacing'])
        orient = row['orientation'].upper()
        si = next((i for i, x in enumerate(orient) if x in 'SI'), None) if len(orient) == 3 and sum(x in 'SI' for x in orient) == 1 else None
        valid = bool(shape and spacing and all(x > 0 for x in shape) and all(math.isfinite(x) and x > 0 for x in spacing))
        si_spacing = spacing[si] if valid and si is not None else None
        si_coverage = shape[si] * si_spacing if si_spacing is not None else None
        if not shape:
            dimension = 'UNKNOWN_OR_NON_3D'
        elif min(shape) == 1:
            dimension = 'SINGLE_SLICE_3D_SHAPE'
        else:
            dimension = '3D_SHAPE'
        if si_spacing is None:
            resolution = 'UNKNOWN'
        elif si_spacing <= 1.5:
            resolution = 'THIN_LE_1_5MM'
        elif si_spacing <= 3:
            resolution = 'MEDIUM_1_5_TO_3MM'
        elif si_spacing <= 5:
            resolution = 'THICK_3_TO_5MM'
        else:
            resolution = 'VERY_THICK_GT_5MM'
        extra = geometry_from_header(path, nib, np, args.spacing_tolerance) if args.check_headers else {
            'header_check': 'NOT_CHECKED', 'affine_spacing_mm': '', 'header_spacing_mm': '',
            'affine_orientation': '', 'header_spacing_match': '', 'qform_sform_match': '',
            'geometry_flag': 'NOT_CHECKED'}
        files.append({
            'patient_id': row['patient_id'], 'class_name': row['class_name'], 'split': row['split'],
            'session': ses, 'acquisition': acq or 'UNSPECIFIED', 'run': run or 'UNSPECIFIED',
            'path': path, 'original_status': row['status'], 'original_reason': row.get('reason', ''),
            'shape': row['shape'], 'spacing': row['spacing'], 'orientation': orient,
            'dimensionality': dimension, 'resolution_group': resolution,
            'si_spacing_mm': round(si_spacing, 4) if si_spacing is not None else '',
            'si_coverage_mm': round(si_coverage, 2) if si_coverage is not None else '', **extra})
    groups = defaultdict(list)
    for r in files:
        groups[r['patient_id']].append(r)
    patients = []
    for pid, rows in sorted(groups.items()):
        if len({r['class_name'] for r in rows}) != 1 or len({r['split'] for r in rows}) != 1:
            raise ValueError(f'Patient {pid} has conflicting labels or splits')
        c = Counter(r['original_status'] for r in rows)
        geometry = Counter(r['geometry_flag'] for r in rows)
        patients.append({
            'patient_id': pid, 'class_name': rows[0]['class_name'], 'split': rows[0]['split'],
            'ct_file_count': len(rows), 'session_count': len({r['session'] for r in rows}),
            'acquisition_count': len({(r['session'], r['acquisition']) for r in rows}),
            'run_count': len({(r['session'], r['run']) for r in rows}),
            'candidate_3d_count': c['CANDIDATE_3D'], 'review_count': c['REVIEW'],
            'exclude_candidate_count': c['EXCLUDE_CANDIDATE'],
            'thin_count': sum(r['resolution_group'] == 'THIN_LE_1_5MM' for r in rows),
            'medium_count': sum(r['resolution_group'] == 'MEDIUM_1_5_TO_3MM' for r in rows),
            'thick_count': sum(r['resolution_group'] == 'THICK_3_TO_5MM' for r in rows),
            'very_thick_count': sum(r['resolution_group'] == 'VERY_THICK_GT_5MM' for r in rows),
            'geometry_flagged_count': sum(r['geometry_flag'] not in ('HEADER_PASS', 'NOT_CHECKED') for r in rows),
            'has_candidate_3d': 'YES' if c['CANDIDATE_3D'] else 'NO',
            'header_check_mode': 'ENABLED' if args.check_headers else 'NOT_CHECKED'})
    train_files = [r for r in files if r['split'] == 'train']
    train_patients = [r for r in patients if r['split'] == 'train']
    def distribution(rows, key):
        return dict(sorted(Counter(r[key] for r in rows).items()))
    report = {
        'mode': 'HEADER_CHECK' if args.check_headers else 'INVENTORY_ONLY',
        'total_files': len(files), 'total_patients': len(patients),
        'training_files': len(train_files), 'training_patients': len(train_patients),
        'training_file_status': distribution(train_files, 'original_status'),
        'training_dimensionality': distribution(train_files, 'dimensionality'),
        'training_resolution_groups': distribution(train_files, 'resolution_group'),
        'training_orientations': distribution(train_files, 'orientation'),
        'training_geometry_flags': distribution(train_files, 'geometry_flag'),
        'training_patient_ct_count_distribution': distribution(train_patients, 'ct_file_count'),
        'training_patient_candidate_count_distribution': distribution(train_patients, 'candidate_3d_count'),
        'training_patient_class_counts': distribution(train_patients, 'class_name'),
        'training_class_file_breakdown': {
            label: {'patients': sum(r['class_name'] == label for r in train_patients),
                    'files': sum(r['class_name'] == label for r in train_files),
                    'candidate_files': sum(r['class_name'] == label and r['original_status'] == 'CANDIDATE_3D' for r in train_files)}
            for label in sorted({r['class_name'] for r in train_patients})},
        'cautions': [
            'Technical spacing groups are exploratory bins, not confirmed DICOM reconstruction types.',
            'Inventory-only mode cannot verify affine, qform/sform, true dimensionality or image quality.',
            'Header check validates metadata consistency, not actual lung completeness.',
            'Only training cohort summaries should inform preprocessing choices; validation/test stay held out.',
            'Do not infer CT view or acquisition protocol from run number alone.',
            'No CT selection or model training performed.'
        ]}
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'ct_patient_image_counts.csv', patients, list(patients[0]))
    write_csv(args.output / 'ct_image_type_distribution.csv', files, list(files[0]))
    write_csv(args.output / 'ct_geometry_validation.csv', files, list(files[0]))
    (args.output / 'ct_stage3b_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Outputs:', args.output.resolve())


if __name__ == '__main__':
    main()
