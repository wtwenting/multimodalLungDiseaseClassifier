#!/usr/bin/env python3
"""CT-only BIMCV metadata EDA and representative volume selection.

Uses an existing CT inventory; never loads CT voxel arrays or trains a model.
Derives descriptive distributions from TRAIN patients only. Applies a fixed,
label-blind selection policy to all patients. Treats existing val/test splits
as read-only and never optimizes thresholds on them.

Example:
 python ct_eda_selection_v2.py --inventory ct_inventory.csv --output ct_eda_v2

Use --root-map OLD=NEW if inventory paths were created on another machine.
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_triplet(s, cast=float):
    try:
        a = tuple(cast(x) for x in str(s).split('x'))
        return a if len(a) == 3 else None
    except (ValueError, TypeError):
        return None


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def write_rows(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def quantiles(values):
    if not values:
        return {'count': 0}
    x = sorted(values)
    def pct(q):
        p = (len(x)-1)*q
        lo, hi = math.floor(p), math.ceil(p)
        return round(x[lo]*(hi-p)+x[hi]*(p-lo), 3) if lo != hi else round(x[lo], 3)
    return {'count': len(x), 'min': pct(0), 'p10': pct(.1), 'p25': pct(.25),
            'median': pct(.5), 'p75': pct(.75), 'p90': pct(.9), 'max': pct(1)}


def process(row, maps):
    r = dict(row)
    raw = r.get('path', '')
    for old, new in maps:
        if raw.startswith(old):
            raw = new + raw[len(old):]
            break
    r['resolved_path'] = raw
    shape = parse_triplet(r.get('shape'), int)
    spacing = parse_triplet(r.get('spacing'))
    orient = str(r.get('orientation', '')).upper()
    valid = (shape is not None and spacing is not None and
             all(v > 0 for v in shape) and
             all(math.isfinite(v) and v > 0 for v in spacing))
    si_axis = next((i for i, c in enumerate(orient) if c in 'SI'), None) if len(orient) == 3 else None
    if len(orient) == 3 and sum(c in 'SI' for c in orient) != 1:
        si_axis = None
    r['si_axis'] = si_axis
    r['si_coverage_mm'] = round(shape[si_axis]*spacing[si_axis], 3) if valid and si_axis is not None else None
    r['si_spacing_mm'] = spacing[si_axis] if valid and si_axis is not None else None
    r['inplane_min_mm'] = min(shape[i]*spacing[i] for i in range(3) if i != si_axis) if valid and si_axis is not None else None
    r['shape_parsed'] = shape
    r['spacing_parsed'] = spacing
    return r


def choose(candidates, min_si_coverage):
    """Prefer adequate SI coverage; then longer SI coverage; then thinner SI spacing.

    Selection is deterministic and does not use COVID label, split, or outcome.
    Threshold is a configurable *policy*, not a clinically validated cutoff.
    """
    eligible = [r for r in candidates if r['si_coverage_mm'] is not None and
                r['si_spacing_mm'] is not None]
    if not eligible:
        return None, 'NO_VALID_ORIENTATION_METADATA'
    adequate = [r for r in eligible if r['si_coverage_mm'] >= min_si_coverage]
    pool = adequate if adequate else eligible
    # Coverage first to avoid selecting thin-slice partial studies.
    selected = sorted(pool, key=lambda r: (-r['si_coverage_mm'], r['si_spacing_mm'],
                                          0 if re.search(r'(?:^|_)run-1_', Path(r['path']).name) else 1,
                                          r['path']))[0]
    return selected, 'ADEQUATE_COVERAGE' if adequate else 'LOW_COVERAGE_REVIEW'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('ct_eda_v2'))
    p.add_argument('--min-si-coverage', type=float, default=250.0,
                   help='Provisional coverage policy in mm; NOT a validated clinical threshold')
    p.add_argument('--root-map', action='append', default=[], metavar='OLD=NEW')
    p.add_argument('--verify-paths', action='store_true', help='Check selected files exist on disk')
    args = p.parse_args()
    if args.min_si_coverage <= 0:
        p.error('--min-si-coverage must be positive')
    maps = []
    for value in args.root_map:
        if '=' not in value:
            p.error('--root-map requires OLD=NEW')
        maps.append(tuple(value.split('=', 1)))
    data = [process(r, maps) for r in read_rows(args.inventory)]
    required = {'patient_id','class_name','split','path','status','shape','spacing','orientation'}
    if not data or not required.issubset(data[0]):
        p.error('Inventory empty or missing required columns: ' + str(sorted(required-set(data[0]) if data else required)))
    groups = defaultdict(list)
    for r in data:
        groups[r['patient_id']].append(r)
    for pid, group in groups.items():
        if len({r['class_name'] for r in group}) != 1 or len({r['split'] for r in group}) != 1:
            raise ValueError(f'Conflicting class or split for patient {pid}')
    train = [r for r in data if r['split'] == 'train' and r['status'] == 'CANDIDATE_3D']
    report = {
        'inventory_files': len(data), 'patients': len(groups),
        'status_counts': dict(Counter(r['status'] for r in data)),
        'candidate_files': sum(r['status']=='CANDIDATE_3D' for r in data),
        'training_candidate_files': len(train),
        'training_candidate_class_counts': dict(Counter(r['class_name'] for r in train)),
        'training_candidate_si_coverage_mm': quantiles([r['si_coverage_mm'] for r in train if r['si_coverage_mm'] is not None]),
        'training_candidate_si_spacing_mm': quantiles([r['si_spacing_mm'] for r in train if r['si_spacing_mm'] is not None]),
        'training_candidate_inplane_min_extent_mm': quantiles([r['inplane_min_mm'] for r in train if r['inplane_min_mm'] is not None]),
        'training_candidate_shape_axis_0': quantiles([r['shape_parsed'][0] for r in train if r['shape_parsed']]),
        'training_candidate_shape_axis_1': quantiles([r['shape_parsed'][1] for r in train if r['shape_parsed']]),
        'training_candidate_shape_axis_2': quantiles([r['shape_parsed'][2] for r in train if r['shape_parsed']]),
        'selection_policy': {'minimum_si_coverage_mm': args.min_si_coverage,
                             'note': 'Provisional; verify visually on training CTs before treating as final.',
                             'ranking': 'coverage threshold, maximum SI coverage, minimum SI spacing, run-1, path'},
        'warning': 'Metadata screening cannot confirm complete lung coverage or CT image quality.'
    }
    selected_rows, audit_rows = [], []
    for pid in sorted(groups):
        group = groups[pid]
        candidates = [r for r in group if r['status'] == 'CANDIDATE_3D']
        chosen, reason = choose(candidates, args.min_si_coverage)
        status = ('NO_CANDIDATE_CT' if not candidates else reason)
        if chosen and args.verify_paths and not Path(chosen['resolved_path']).is_file():
            status = 'SELECTED_FILE_MISSING'
        selected_rows.append({
            'patient_id': pid, 'class_name': group[0]['class_name'], 'split': group[0]['split'],
            'candidate_count': len(candidates), 'selection_status': status,
            'ct_path': chosen['resolved_path'] if chosen else '',
            'original_ct_path': chosen['path'] if chosen else '',
            'si_coverage_mm': chosen['si_coverage_mm'] if chosen else '',
            'si_spacing_mm': chosen['si_spacing_mm'] if chosen else '',
            'shape': chosen['shape'] if chosen else '',
            'spacing': chosen['spacing'] if chosen else ''})
        for r in group:
            audit_rows.append({
                'patient_id': pid, 'split': r['split'], 'class_name': r['class_name'],
                'path': r['path'], 'original_status': r['status'], 'original_reason': r['reason'],
                'si_coverage_mm': r['si_coverage_mm'], 'si_spacing_mm': r['si_spacing_mm'],
                'decision': 'SELECTED' if chosen is r else 'NOT_SELECTED',
                'patient_selection_status': status})
    report['selection_counts'] = dict(Counter(r['selection_status'] for r in selected_rows))
    report['selected_patient_counts_by_split'] = dict(Counter(r['split'] for r in selected_rows if r['ct_path']))
    report['low_coverage_training_selected'] = sum(r['split']=='train' and r['selection_status']=='LOW_COVERAGE_REVIEW' for r in selected_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output/'ct_selected_patient_manifest.csv', selected_rows, list(selected_rows[0]))
    write_rows(args.output/'ct_selection_audit.csv', audit_rows, list(audit_rows[0]))
    (args.output/'ct_train_eda_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Saved outputs to:', args.output.resolve())
    print('No CT image arrays loaded; no preprocessing or training performed.')


if __name__ == '__main__':
    main()
