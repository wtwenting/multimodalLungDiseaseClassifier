#!/usr/bin/env python3
"""Stage 5F: CT-only targeted QA and conservative V4 proposals.

Reads V3 selected-patient manifest, V3 candidate audit, and Stage 5E screening.
Optionally reads a manual-review CSV (patient_id, ct_path, manual_status).
Does not load image voxels, train, change splits, or modify original files.
It NEVER automatically approves or substitutes a CT volume.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

KNOWN = {'S03048', 'S03074', 'S04583', 'S05295', 'S03834', 'S03259'}
REVIEW_STATES = {'PASS', 'REVIEW', 'FAIL', 'PENDING'}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def norm_id(s):
    return (s or '').strip().removeprefix('sub-')


def key(patient, path):
    return (norm_id(patient), (path or '').strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--screening', type=Path, required=True)
    p.add_argument('--manual', type=Path, help='Optional existing manually completed review CSV')
    p.add_argument('--output', type=Path, default=Path('output/ct_eda_stage5f'))
    p.add_argument('--max-alternatives', type=int, default=5)
    args = p.parse_args()
    if args.max_alternatives < 1:
        p.error('--max-alternatives must be >= 1')
    manifest, audit, screening = (read_csv(x) for x in (args.manifest, args.audit, args.screening))
    for name, rows, required in (
        ('manifest', manifest, {'patient_id', 'split', 'ct_path', 'selection_status'}),
        ('audit', audit, {'patient_id', 'split', 'path', 'eligible', 'geometry_flag'}),
        ('screening', screening, {'patient_id', 'split', 'ct_path', 'triage', 'flags'}),
    ):
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f'{name} missing required columns: {sorted(required - set(rows[0]) if rows else required)}')
    by_patient = defaultdict(list)
    for r in audit:
        if r['split'].strip() == 'train' and r['eligible'].strip().lower() in ('true', 'yes', '1') and r['geometry_flag'] == 'HEADER_PASS':
            by_patient[norm_id(r['patient_id'])].append(r)
    screens = {key(r['patient_id'], r['ct_path']): r for r in screening if r['split'].strip() == 'train'}
    reviews = {}
    if args.manual:
        manual_rows = read_csv(args.manual)
        if manual_rows and not {'patient_id', 'ct_path', 'manual_status'}.issubset(manual_rows[0]):
            raise ValueError('Manual CSV must contain patient_id, ct_path, manual_status')
        for r in manual_rows:
            k = key(r['patient_id'], r['ct_path'])
            status = r['manual_status'].strip().upper() or 'PENDING'
            if status not in REVIEW_STATES:
                raise ValueError(f'Invalid manual_status {status} for {k}')
            if k in reviews:
                raise ValueError(f'Duplicate manual review for {k}')
            reviews[k] = r
    selected = [r for r in manifest if r['split'].strip() == 'train' and r['ct_path'].strip()]
    candidates = []
    proposals = []
    for row in selected:
        pid, current = norm_id(row['patient_id']), row['ct_path'].strip()
        s = screens.get(key(pid, current), {})
        flag = s.get('flags', '')
        triage = s.get('triage', 'NOT_SCREENED')
        m = reviews.get(key(pid, current), {})
        status = m.get('manual_status', 'PENDING').strip().upper() or 'PENDING'
        priority = 'HIGH' if pid in KNOWN or status in ('FAIL', 'REVIEW') else ('MEDIUM' if triage == 'REVIEW' or row['selection_status'] == 'LOW_COVERAGE_REVIEW' else 'SAMPLE_QA')
        alternatives = [a for a in by_patient[pid] if a['path'].strip() != current]
        alternatives.sort(key=lambda a: (-float(a.get('si_coverage_mm') or 0), float(a.get('si_spacing_mm') or 999), a['path']))
        passed = [a for a in alternatives if reviews.get(key(pid, a['path']), {}).get('manual_status', '').strip().upper() == 'PASS']
        # Proposals are suggestions only; an explicit PASS is required to suggest a substitute.
        action = 'REVIEW_SELECTED' if priority == 'HIGH' else 'QA_SAMPLE_OR_REVIEW'
        proposed = ''
        if status == 'PASS':
            action = 'MANUAL_PASS_SELECTED'
        elif status == 'FAIL':
            if passed:
                action, proposed = 'PROPOSE_MANUAL_PASS_ALTERNATIVE', passed[0]['path']
            else:
                action = 'SELECTED_FAIL_REVIEW_ALTERNATIVES'
        elif status == 'REVIEW':
            action = 'REVIEW_SELECTED_AND_ALTERNATIVES'
        proposals.append(dict(patient_id=pid, split='train', current_ct_path=current,
                              selection_status=row['selection_status'], triage=triage,
                              flags=flag, priority=priority, manual_status=status,
                              eligible_alternative_count=len(alternatives),
                              manual_pass_alternative_count=len(passed), action=action,
                              proposed_ct_path=proposed,
                              note='Proposal only; never automatically update V3 manifest'))
        # Include selected volume and top alternatives in an editable manual QA queue.
        chosen = [a for a in by_patient[pid] if a['path'].strip() == current]
        for a in (chosen[:1] + alternatives[:args.max_alternatives]):
            path = a['path'].strip()
            old = reviews.get(key(pid, path), {})
            ss = screens.get(key(pid, path), {})
            candidates.append(dict(patient_id=pid, split='train', ct_path=path,
                                   is_v3_selected='YES' if path == current else 'NO',
                                   geometry_flag=a.get('geometry_flag', ''),
                                   si_coverage_mm=a.get('si_coverage_mm', ''),
                                   si_spacing_mm=a.get('si_spacing_mm', ''),
                                   screening_triage=ss.get('triage', 'NOT_SCREENED'),
                                   screening_flags=ss.get('flags', ''),
                                   review_priority=priority,
                                   manual_status=old.get('manual_status', 'PENDING') or 'PENDING',
                                   manual_notes=old.get('manual_notes', old.get('notes', ''))))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'ct_stage5f_review_queue.csv', candidates,
              ['patient_id', 'split', 'ct_path', 'is_v3_selected', 'geometry_flag',
               'si_coverage_mm', 'si_spacing_mm', 'screening_triage', 'screening_flags',
               'review_priority', 'manual_status', 'manual_notes'])
    write_csv(args.output / 'ct_stage5f_v4_proposals.csv', proposals,
              ['patient_id', 'split', 'current_ct_path', 'selection_status', 'triage', 'flags',
               'priority', 'manual_status', 'eligible_alternative_count',
               'manual_pass_alternative_count', 'action', 'proposed_ct_path', 'note'])
    summary = dict(mode='TRAIN_ONLY_STAGE5F_PROPOSALS', selected_training_patients=len(selected),
                   queue_ct_files=len(candidates), priorities=dict(Counter(x['priority'] for x in proposals)),
                   actions=dict(Counter(x['action'] for x in proposals)),
                   known_high_risk_training_patients=sorted(KNOWN & {norm_id(r['patient_id']) for r in selected}),
                   notes=['V3 manifest unchanged; no automatic substitution, exclusion or training.',
                          'Stage 5E flags are heuristic and are not clinical-quality labels.',
                          'Alternative CTs are not approved until manually reviewed.',
                          'Use training data to establish rules; apply frozen rules to validation/test.'])
    (args.output / 'ct_stage5f_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Output:', args.output.resolve())


if __name__ == '__main__':
    main()
