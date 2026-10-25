#!/usr/bin/env python3
"""Adapt local BIMCV V3 CT selection files to teammate V5 input schema.
Read-only inputs. Does not run preprocessing or change patient splits.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label_of(name):
    mapping = {'covidPositive': 'Positive', 'covidNegative': 'Negative'}
    if name not in mapping:
        raise ValueError(f'Unexpected class_name: {name}')
    return mapping[name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--check-paths', action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    manifest = read_csv(args.manifest)
    audit = read_csv(args.audit)
    mrows = []
    selected = set()
    for row in manifest:
        patient = row['patient_id'].strip()
        label = label_of(row['class_name'])
        source = row['ct_path'].strip()
        status = 'accepted' if row['selection_status'] in ('ADEQUATE_COVERAGE', 'LOW_COVERAGE_REVIEW') and source else 'excluded'
        if status == 'accepted':
            if (label, patient) in selected:
                raise ValueError(f'Duplicate selected patient: {label}/{patient}')
            selected.add((label, patient))
        mrows.append({'label': label, 'patient': patient, 'source_path': source,
                      'status': status, 'split': row['split'],
                      'v3_selection_status': row['selection_status'],
                      'geometry_flag': row['geometry_flag'],
                      'si_coverage_mm': row['si_coverage_mm'],
                      'si_spacing_mm': row['si_spacing_mm']})
    srows = []
    for row in audit:
        label = label_of(row['class_name'])
        patient = row['patient_id'].strip()
        eligible = row['eligible'].strip().lower() == 'true'
        coverage = float(row['si_coverage_mm']) if row['si_coverage_mm'] else 0.0
        spacing = float(row['si_spacing_mm']) if row['si_spacing_mm'] else 1000.0
        # Mirrors V3's coverage gate -> coverage -> spacing ordering.
        # This is a selection preference, not anatomical quality evidence.
        quality = (1000000 if coverage >= 250 else 0) + min(coverage, 100000) - min(spacing, 1000) / 10000
        srows.append({'label': label, 'patient': patient, 'path': row['path'],
                      'status': 'chest_candidate' if eligible else 'excluded',
                      'quality': f'{quality:.6f}' if eligible else '',
                      'split': row['split'], 'v3_decision': row['decision'],
                      'geometry_flag': row['geometry_flag'],
                      'si_coverage_mm': row['si_coverage_mm'],
                      'si_spacing_mm': row['si_spacing_mm']})
    if args.check_paths:
        missing = [r['source_path'] for r in mrows if r['status'] == 'accepted' and not Path(r['source_path']).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} selected source paths missing; first: {missing[0]}')
    write_csv(out/'patient_manifest.csv', mrows, list(mrows[0]))
    write_csv(out/'series_audit.csv', srows, list(srows[0]))
    summary = {'source_manifest': str(args.manifest), 'source_audit': str(args.audit),
               'patients': len(mrows), 'accepted': sum(r['status']=='accepted' for r in mrows),
               'excluded': sum(r['status']=='excluded' for r in mrows),
               'accepted_by_split': dict(Counter(r['split'] for r in mrows if r['status']=='accepted')),
               'eligible_alternative_series': sum(r['status']=='chest_candidate' for r in srows),
               'series_total': len(srows),
               'warning': 'V5 PASS requires lungmask QC; accepted here means V3 selected, not V5 approved. Existing known anatomical concerns remain.'}
    (out/'adapter_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
