#!/usr/bin/env python3
"""Create read-only V5 encoder eligibility and quarantine manifests.

Run from ensemble/: python src/create_ct_v5_eligible_manifest.py \
  --v5 ct_v5_corrected_preprocessed/patient_manifest.csv \
  --v3 output/ct_eda_v3/ct_selected_patient_manifest_v3.csv \
  --out output/ct_v5_encoder_audit
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

QUARANTINE = {
    'S07090': 'V5_QC_REVIEW',
    'S10766': 'V5_QC_REVIEW',
    'S312604': 'V5_QC_REVIEW',
    'S03074': 'ANATOMICAL_ORIENTATION_REVIEW_DESPITE_V5_PASS',
}


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v5', required=True)
    ap.add_argument('--v3', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    v5 = read_csv(args.v5)
    v3 = read_csv(args.v3)
    splits = {}
    for r in v3:
        pid = r['patient_id'].strip().upper()
        if pid in splits:
            raise ValueError(f'Duplicate V3 patient: {pid}')
        splits[pid] = (r['split'].strip(), r['class_name'].strip())
    rows = []
    seen = set()
    for r in v5:
        pid = r['patient'].strip().upper()
        if pid in seen:
            raise ValueError(f'Duplicate V5 patient: {pid}')
        seen.add(pid)
        if pid not in splits:
            raise ValueError(f'Patient absent from V3 split: {pid}')
        split, class_name = splits[pid]
        expected = 'Positive' if class_name == 'covidPositive' else 'Negative' if class_name == 'covidNegative' else None
        if r['label'].strip() != expected:
            raise ValueError(f'Label mismatch: {pid} V3={class_name} V5={r["label"]}')
        reason = QUARANTINE.get(pid, '')
        if r['status'].strip().upper() != 'PASS':
            reason = reason or 'V5_QC_NOT_PASS'
        if not r['tensor_path'].strip():
            reason = reason or 'MISSING_TENSOR_PATH'
        entry = dict(r)
        entry.update(patient_id=pid, split=split, v3_class_name=class_name,
                     eligibility_status='QUARANTINE' if reason else 'ELIGIBLE',
                     quarantine_reason=reason)
        rows.append(entry)
    missing = sorted(set(QUARANTINE) - seen)
    if missing:
        raise ValueError(f'Quarantine patients missing from V5 manifest: {missing}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    eligible = [r for r in rows if r['eligibility_status'] == 'ELIGIBLE']
    quarantine = [r for r in rows if r['eligibility_status'] == 'QUARANTINE']
    write_csv(out/'ct_v5_encoder_eligible_manifest.csv', eligible, cols)
    write_csv(out/'ct_v5_quarantine_manifest.csv', quarantine, cols)
    summary = {'total_v5': len(rows), 'eligible': len(eligible), 'quarantined': len(quarantine),
               'eligible_by_split': dict(Counter(r['split'] for r in eligible)),
               'quarantined_by_split': dict(Counter(r['split'] for r in quarantine)),
               'quarantined_patients': {r['patient_id']: r['quarantine_reason'] for r in quarantine},
               'note': 'No files deleted. Eligibility is provisional pending broader anatomical QC. Preserve existing patient-level splits.'}
    (out/'ct_v5_encoder_eligibility_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
