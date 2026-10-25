#!/usr/bin/env python3
"""CT-only geometry-aware, patient-level representative-volume selection.

Consumes Stage 1 inventory and Stage 3B header-validation CSV. Does not load
voxels, modify original images, train, or evaluate models. Selection policy is
fixed and label-blind. Train-only diagnostics; val/test selection is mechanical.
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_triplet(value, kind=float):
    try:
        vals = tuple(kind(x) for x in value.split('x'))
        return vals if len(vals) == 3 else None
    except (ValueError, AttributeError):
        return None


def map_path(path, mappings):
    for old, new in mappings:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def metadata(row):
    shape = parse_triplet(row.get('shape', ''), int)
    spacing = parse_triplet(row.get('affine_spacing_mm', ''))
    orientation = row.get('affine_orientation', '')
    if not spacing:
        spacing = parse_triplet(row.get('header_spacing_mm', ''))
    if not shape or not spacing or len(orientation) != 3:
        return None
    si = [i for i, axis in enumerate(orientation) if axis in 'SI']
    if len(si) != 1 or any(x <= 0 for x in shape) or any(not math.isfinite(x) or x <= 0 for x in spacing):
        return None
    axis = si[0]
    return {'si_coverage_mm': shape[axis] * spacing[axis],
            'si_spacing_mm': spacing[axis],
            'inplane_min_mm': min(shape[i] * spacing[i] for i in range(3) if i != axis)}


def choose(rows, minimum_coverage):
    eligible = [r for r in rows if r['eligible'] and r['metrics'] is not None]
    if not eligible:
        return None, 'NO_GEOMETRY_VALID_CANDIDATE'
    adequate = [r for r in eligible if r['metrics']['si_coverage_mm'] >= minimum_coverage]
    pool = adequate or eligible
    # Provisional: coverage gate, then maximum coverage, thinner SI spacing,
    # run-1 tie-breaker, path. Not an anatomical-quality assessment.
    selected = min(pool, key=lambda r: (-r['metrics']['si_coverage_mm'],
                                       r['metrics']['si_spacing_mm'],
                                       0 if re.search(r'(?:^|_)run-1_', Path(r['path']).name) else 1,
                                       r['path']))
    return selected, 'ADEQUATE_COVERAGE' if adequate else 'LOW_COVERAGE_REVIEW'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('output/ct_eda_v3'))
    parser.add_argument('--min-si-coverage', type=float, default=250.0)
    parser.add_argument('--root-map', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--verify-paths', action='store_true')
    args = parser.parse_args()
    if args.min_si_coverage <= 0:
        parser.error('--min-si-coverage must be positive')
    mappings = []
    for entry in args.root_map:
        if '=' not in entry:
            parser.error('--root-map must be OLD=NEW')
        mappings.append(tuple(entry.split('=', 1)))

    inventory = read_csv(args.inventory)
    geometry = read_csv(args.geometry)
    inv_required = {'patient_id', 'class_name', 'split', 'path', 'status'}
    geo_required = {'patient_id', 'class_name', 'split', 'path', 'original_status',
                    'geometry_flag', 'header_check', 'shape', 'affine_spacing_mm', 'affine_orientation'}
    if not inventory or not inv_required.issubset(inventory[0]):
        raise ValueError('Inventory missing required columns')
    if not geometry or not geo_required.issubset(geometry[0]):
        raise ValueError('Geometry CSV missing required columns')
    inv_map = {}
    for row in inventory:
        if row['path'] in inv_map:
            raise ValueError('Duplicate inventory path: ' + row['path'])
        inv_map[row['path']] = row
    geo_map = {}
    for row in geometry:
        if row['path'] in geo_map:
            raise ValueError('Duplicate geometry path: ' + row['path'])
        geo_map[row['path']] = row
    if set(inv_map) != set(geo_map):
        missing = set(inv_map) - set(geo_map)
        extra = set(geo_map) - set(inv_map)
        raise ValueError(f'Inventory/geometry mismatch: {len(missing)} missing, {len(extra)} extra')

    groups = defaultdict(list)
    for path, inv in inv_map.items():
        geo = geo_map[path]
        for field in ('patient_id', 'class_name', 'split'):
            if inv[field] != geo[field]:
                raise ValueError(f'{field} mismatch for {path}')
        if inv['status'] != geo['original_status']:
            raise ValueError(f'Status mismatch for {path}')
        metrics = metadata(geo)
        eligible = (inv['status'] == 'CANDIDATE_3D' and
                    geo['geometry_flag'] == 'HEADER_PASS' and
                    geo['header_check'] == 'OK' and metrics is not None)
        groups[inv['patient_id']].append({'path': path, 'inventory': inv,
            'geometry': geo, 'metrics': metrics, 'eligible': eligible})

    selected_rows, audit_rows = [], []
    for patient_id in sorted(groups):
        rows = groups[patient_id]
        splits = {r['inventory']['split'] for r in rows}
        classes = {r['inventory']['class_name'] for r in rows}
        if len(splits) != 1 or len(classes) != 1:
            raise ValueError('Inconsistent patient labels or splits: ' + patient_id)
        split, cls = next(iter(splits)), next(iter(classes))
        chosen, status = choose(rows, args.min_si_coverage)
        if chosen is None:
            status = ('NO_CANDIDATE_CT' if not any(r['inventory']['status'] == 'CANDIDATE_3D' for r in rows)
                      else status)
        resolved = map_path(chosen['path'], mappings) if chosen else ''
        if chosen and args.verify_paths and not Path(resolved).is_file():
            status = 'SELECTED_FILE_MISSING'
        selected_rows.append({'patient_id': patient_id, 'class_name': cls, 'split': split,
            'total_ct_files': len(rows),
            'original_candidate_count': sum(r['inventory']['status'] == 'CANDIDATE_3D' for r in rows),
            'geometry_valid_candidate_count': sum(r['eligible'] for r in rows),
            'selection_status': status, 'ct_path': resolved,
            'original_ct_path': chosen['path'] if chosen else '',
            'si_coverage_mm': round(chosen['metrics']['si_coverage_mm'], 3) if chosen else '',
            'si_spacing_mm': round(chosen['metrics']['si_spacing_mm'], 5) if chosen else '',
            'shape': chosen['geometry']['shape'] if chosen else '',
            'orientation': chosen['geometry']['affine_orientation'] if chosen else '',
            'geometry_flag': chosen['geometry']['geometry_flag'] if chosen else ''})
        for r in rows:
            inv, geo = r['inventory'], r['geometry']
            decision = ('SELECTED' if chosen is r else
                        'REJECT_ORIGINAL_STATUS' if inv['status'] != 'CANDIDATE_3D' else
                        'REJECT_GEOMETRY' if not r['eligible'] else 'NOT_SELECTED_ALTERNATIVE')
            audit_rows.append({'patient_id': patient_id, 'split': split, 'class_name': cls,
                'path': r['path'], 'original_status': inv['status'],
                'original_reason': inv.get('reason', ''), 'geometry_flag': geo['geometry_flag'],
                'header_check': geo['header_check'], 'si_coverage_mm': round(r['metrics']['si_coverage_mm'], 3) if r['metrics'] else '',
                'si_spacing_mm': round(r['metrics']['si_spacing_mm'], 5) if r['metrics'] else '',
                'eligible': r['eligible'], 'decision': decision,
                'patient_selection_status': status})

    train_audit = [r for r in audit_rows if r['split'] == 'train']
    train_selected = [r for r in selected_rows if r['split'] == 'train']
    summary = {'mode': 'GEOMETRY_AWARE_SELECTION_V3',
        'inventory_files': len(inventory), 'patients': len(groups),
        'training_files': len(train_audit), 'training_patients': len(train_selected),
        'training_original_candidate_files': sum(r['original_status'] == 'CANDIDATE_3D' for r in train_audit),
        'training_candidate_geometry_flags': dict(Counter(r['geometry_flag'] for r in train_audit if r['original_status'] == 'CANDIDATE_3D')),
        'training_geometry_valid_candidates': sum(r['eligible'] for r in train_audit),
        'training_selection_counts': dict(Counter(r['selection_status'] for r in train_selected)),
        'training_selected_by_class': dict(Counter(r['class_name'] for r in train_selected if r['ct_path'])),
        'all_patient_selection_counts': dict(Counter(r['selection_status'] for r in selected_rows)),
        'selected_patients_by_split': dict(Counter(r['split'] for r in selected_rows if r['ct_path'])),
        'changed_from_v2': 'Compare output manifest to prior V2 manifest; no V2 file supplied as input.',
        'selection_policy': {'require_original_status': 'CANDIDATE_3D',
            'require_geometry_flag': 'HEADER_PASS', 'require_header_check': 'OK',
            'minimum_si_coverage_mm': args.min_si_coverage,
            'ranking': 'coverage gate, maximum SI coverage, minimum SI spacing, run-1, path',
            'coverage_threshold_is_provisional': True},
        'cautions': ['HEADER_PASS is metadata screening, not anatomical or clinical quality validation.',
            'Do not choose thresholds from validation/test distributions.',
            'Only train-set aggregate diagnostic statistics are reported.',
            'Do not use this output for model training before visual QA and verifying split integrity.',
            'No CT arrays loaded; no model trained.']}
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'ct_selected_patient_manifest_v3.csv', selected_rows, list(selected_rows[0]))
    write_csv(args.output / 'ct_selection_audit_v3.csv', audit_rows, list(audit_rows[0]))
    (args.output / 'ct_selection_summary_v3.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
