#!/usr/bin/env python3
"""Read-only BIMCV CT/X-ray patient and session inventory. Python 3.9+."""
import argparse
import csv
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

EXTS = {'.nii', '.nii.gz', '.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
PAT = re.compile(r'^sub-([A-Za-z0-9]+)$', re.I)
SES = re.compile(r'^ses-([A-Za-z0-9]+)$', re.I)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scan(root):
    groups = defaultdict(lambda: {'CT': [], 'Xray': []})
    skipped = []
    for label in ('covidPositive', 'covidNegative'):
        for folder, modality in (('ct', 'CT'), ('xray', 'Xray')):
            base = root / label / folder
            if not base.is_dir():
                skipped.append({'path': str(base), 'reason': 'MODALITY_FOLDER_MISSING'})
                continue
            for path in base.rglob('*'):
                if not path.is_file():
                    continue
                ext = '.nii.gz' if path.name.lower().endswith('.nii.gz') else path.suffix.lower()
                if ext not in EXTS:
                    continue
                rel = path.relative_to(base)
                pat = next((m.group(1).upper() for p in rel.parts if (m := PAT.fullmatch(p))), None)
                ses = next((m.group(1).upper() for p in rel.parts if (m := SES.fullmatch(p))), None)
                if not pat or not ses:
                    skipped.append({'path': str(path), 'reason': 'PATIENT_OR_SESSION_NOT_FOUND'})
                    continue
                groups[(label, 'S' + pat.lstrip('S'), 'E' + ses.lstrip('E'))][modality].append(str(path))
    return groups, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path, help='BIMCV data directory containing covidPositive and covidNegative')
    parser.add_argument('--output', required=True, type=Path, help='Output directory outside original dataset')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Dataset directory does not exist: {root}')
    if out == root or root in out.parents:
        parser.error('Output must be outside dataset root')
    out.mkdir(parents=True, exist_ok=True)
    groups, skipped = scan(root)
    sessions = []
    files = []
    patient_groups = defaultdict(list)
    for (label, patient, session), mods in sorted(groups.items()):
        ct, xr = sorted(mods['CT']), sorted(mods['Xray'])
        status = 'BOTH_SAME_SESSION' if ct and xr else 'CT_ONLY' if ct else 'XRAY_ONLY'
        sessions.append({'label': label, 'patient_id': patient, 'session_id': session,
                         'ct_count': len(ct), 'xray_count': len(xr), 'session_status': status})
        patient_groups[(label, patient)].append((session, len(ct), len(xr)))
        for modality, paths in (('CT', ct), ('Xray', xr)):
            for path in paths:
                files.append({'label': label, 'patient_id': patient, 'session_id': session,
                              'modality': modality, 'file_path': path, 'session_status': status})
    patients = []
    for (label, patient), entries in sorted(patient_groups.items()):
        ct_sessions = {s for s, c, x in entries if c}
        xr_sessions = {s for s, c, x in entries if x}
        common = ct_sessions & xr_sessions
        status = ('BOTH_WITH_MATCHING_SESSION' if common else
                  'BOTH_DIFFERENT_SESSIONS' if ct_sessions and xr_sessions else
                  'CT_ONLY_PATIENT' if ct_sessions else 'XRAY_ONLY_PATIENT')
        patients.append({'label': label, 'patient_id': patient, 'ct_sessions': len(ct_sessions),
                         'xray_sessions': len(xr_sessions), 'matched_sessions': len(common),
                         'ct_files': sum(c for _, c, _ in entries),
                         'xray_files': sum(x for _, _, x in entries), 'patient_status': status,
                         'matched_session_ids': ';'.join(sorted(common))})
    write_csv(out / 'ct_xray_session_groups.csv', sessions,
              ['label', 'patient_id', 'session_id', 'ct_count', 'xray_count', 'session_status'])
    write_csv(out / 'ct_xray_patient_summary.csv', patients,
              ['label', 'patient_id', 'ct_sessions', 'xray_sessions', 'matched_sessions',
               'ct_files', 'xray_files', 'patient_status', 'matched_session_ids'])
    write_csv(out / 'ct_xray_file_inventory.csv', files,
              ['label', 'patient_id', 'session_id', 'modality', 'file_path', 'session_status'])
    write_csv(out / 'ct_xray_skipped.csv', skipped, ['path', 'reason'])
    summary = {'data_root': str(root), 'unique_patients': len(patients),
               'unique_patient_sessions': len(sessions), 'recognized_image_files': len(files),
               'session_status_counts': dict(Counter(r['session_status'] for r in sessions)),
               'patient_status_counts': dict(Counter(r['patient_status'] for r in patients)),
               'by_label': {label: {'patients': sum(p['label'] == label for p in patients),
                                    'sessions': sum(s['label'] == label for s in sessions),
                                    'ct_files': sum(p['ct_files'] for p in patients if p['label'] == label),
                                    'xray_files': sum(p['xray_files'] for p in patients if p['label'] == label)}
                            for label in ('covidPositive', 'covidNegative')},
               'skipped_count': len(skipped),
               'cautions': ['Matching patient and session directory IDs do not prove identical acquisition timestamps.',
                            'All runs within a patient must remain in the same train/validation/test split.',
                            'This script counts files only; it does not inspect image contents or verify modality metadata.',
                            'Patient IDs are grouped within label folders; review any patients appearing under both labels.']}
    (out / 'ct_xray_session_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'data_root'}, indent=2))
    print(f'\nOutputs saved to: {out}')


if __name__ == '__main__':
    main()
