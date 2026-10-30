#!/usr/bin/env python3
"""Stage 5D: manual CT anatomical QA and conservative V4 selection proposals.

No voxel arrays are loaded, no patient split changes, no automatic exclusions.
Workflow: run once to create review template, annotate PASS/REVIEW/FAIL,
run again. A replacement is proposed ONLY when an alternative CT is manually
marked PASS and is eligible in the V3 audit. Never auto-approve a replacement.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REVIEWS = ('PENDING', 'PASS', 'REVIEW', 'FAIL')

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, cols):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def key(row):
    return row['patient_id'].strip().removeprefix('sub-')

def truth(value):
    return str(value).strip().lower() in ('true', '1', 'yes')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--previews', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.manifest)
    audit = read_csv(args.audit)
    previews = read_csv(args.previews)
    by_patient = defaultdict(list)
    for row in audit:
        by_patient[key(row)].append(row)
    review_path = args.output / 'ct_manual_review.csv'
    if not review_path.exists():
        initial = []
        seen = set()
        for row in previews:
            if row.get('status', '').upper() != 'OK':
                continue
            pid, path = key(row), row['file_path'].strip()
            if (pid, path) in seen:
                continue
            seen.add((pid, path))
            initial.append(dict(patient_id=pid, ct_path=path,
                                manual_status='PENDING', reason='', reviewer='',
                                preview_path=row.get('preview_path', ''), notes=''))
        write_csv(review_path, initial,
                  ['patient_id','ct_path','manual_status','reason','reviewer','preview_path','notes'])
        print('Created manual review template:', review_path)
        print('Review PNGs and fill PASS / REVIEW / FAIL, then rerun.')
        print('You may add rows for alternative CT paths after visually reviewing them.')
        return

    reviews = read_csv(review_path)
    review_map = {}
    for r in reviews:
        pid, path = key(r), r['ct_path'].strip()
        status = r['manual_status'].strip().upper()
        if status not in REVIEWS:
            raise ValueError(f'Invalid manual_status for {pid}: {status}')
        if (pid, path) in review_map:
            raise ValueError(f'Duplicate manual review: {pid} {path}')
        if path not in {a['path'] for a in by_patient[pid]}:
            raise ValueError(f'Review path not found in V3 audit: {pid} {path}')
        review_map[(pid, path)] = r

    proposal = []
    stats = Counter()
    for row in manifest:
        pid = key(row)
        selected = row.get('ct_path', '').strip()
        split = row.get('split', '')
        if not selected:
            state, replacement, explanation = 'NO_CANDIDATE_CT', '', 'No V3 candidate'
        else:
            current = review_map.get((pid, selected), {})
            status = current.get('manual_status', 'PENDING').strip().upper()
            if status == 'PASS':
                state, replacement, explanation = 'RETAIN_REVIEWED_PASS', selected, 'Selected CT manually passed'
            elif status in ('PENDING', 'REVIEW'):
                state, replacement, explanation = 'NEEDS_REVIEW', '', 'Selected CT not manually approved'
            else:
                alternatives = []
                for a in by_patient[pid]:
                    if a['path'] == selected or not truth(a.get('eligible')):
                        continue
                    manual = review_map.get((pid, a['path']), {})
                    if manual.get('manual_status','').strip().upper() != 'PASS':
                        continue
                    coverage = float(a.get('si_coverage_mm') or 0)
                    spacing = float(a.get('si_spacing_mm') or 999)
                    alternatives.append((-coverage, spacing, a['path']))
                if alternatives:
                    alternatives.sort()
                    replacement = alternatives[0][2]
                    state = 'REPLACEMENT_PROPOSED'
                    explanation = 'Selected CT failed; eligible alternative manually passed; confirm before use'
                else:
                    replacement = ''
                    state = 'NO_APPROVED_ALTERNATIVE'
                    explanation = 'Selected CT failed; no manually approved eligible alternative'
        stats[state] += 1
        proposal.append(dict(patient_id=pid, split=split, class_name=row.get('class_name',''),
                             v3_ct_path=selected, proposed_ct_path=replacement,
                             proposal_status=state, reason=explanation))
    write_csv(args.output / 'ct_v4_proposals.csv', proposal, list(proposal[0]))
    summary = {'mode':'MANUAL_QA_PROPOSALS_ONLY', 'patients':len(manifest),
               'manual_review_counts':dict(Counter(r['manual_status'].upper() for r in reviews)),
               'proposal_counts':dict(stats), 'note':'No automatic replacements, exclusions, or model training. Review proposal CSV before updating manifest.'}
    (args.output / 'ct_stage5d_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Proposals:', args.output / 'ct_v4_proposals.csv')

if __name__ == '__main__':
    main()
