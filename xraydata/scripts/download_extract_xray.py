#!/usr/bin/env python3
"""Step 2: download ONE archive, extract only the selected patients' X-rays, verify, delete, repeat.

Adapted from the groupmate's download_extract_cleanup.py:
  - no hardcoded paths: everything lives under --root (default: this script's folder)
  - X-ray only: CT volumes are never written, even if the mapping says CT=Yes
  - archive queue comes from xray_only_archives.csv (most X-ray-only patients first),
    not a hand-typed list
  - downloads with Python itself (resumable), so it also works on Windows without curl
  - --target-patients stops once you have enough patients per class, so you don't
    have to pull hundreds of 5 GB archives
  - its own checkpoint file, so it can't clash with the groupmate's CT state

Layout written:
  <root>/data/covidPositive/xray/sub-S#####/ses-E#####/*.png
  <root>/data/covidNegative/xray/...
  <root>/xray_work/xray_extraction_state.json   (resume checkpoint)
  <root>/xray_work/xray_extraction_report.csv

Stdlib only. Python 3.9+. Safe to stop (Ctrl+C) and rerun: finished archives are skipped
and partial downloads resume.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:                                   # use the operating system's certificates for HTTPS (fixes
    import truststore                  # CERTIFICATE_VERIFY_FAILED on Windows); pip install truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
BASE = {
    'Negative': 'https://b2drop.bsc.es/nextcloud/public.php/dav/files/BIMCV-COVID19-cIter_1_2-Negative/',
    'Positive': 'https://b2drop.bsc.es/nextcloud/public.php/dav/files/BIMCV-COVID19-cIter_1_2_3/',
}
UA = {'User-Agent': 'BIMCV-research-xray/1.0'}
ARCHIVE_PATTERN = re.compile(r'^covid19_(?:posi|pos|neg)_subjects_part[a-z]+\.tar\.gz$', re.I)
PATIENT_PATTERN = re.compile(r'(?:^|/)sub-(S\d+)(?:/|_)', re.I)
SESSION_PATTERN = re.compile(r'(?:^|/)ses-(E\d+)(?:/|_)', re.I)
CHUNK = 1024 * 1024


# ------------------------------------------------------------------ mapping
def normalize_status(raw):
    v = str(raw).strip().lower()
    if v in ('positive', 'pos', 'posi'):
        return 'Positive'
    if v in ('negative', 'neg'):
        return 'Negative'
    raise ValueError(f'Unknown COVID status: {raw!r}')


def load_mapping(path):
    """{(status, archive): set(patient_ids)} for rows with X-ray=Yes."""
    groups = defaultdict(set)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        need = {'Patient ID', 'COVID status', 'Archive', 'X-ray'}
        if not need.issubset(reader.fieldnames or []):
            raise ValueError(f'Mapping missing columns: {sorted(need - set(reader.fieldnames or []))}')
        for row in reader:
            if row['X-ray'].strip().lower() not in ('yes', 'true', '1'):
                continue
            m = re.fullmatch(r'(?:sub-)?(S\d+)', row['Patient ID'].strip(), re.I)
            if not m:
                raise ValueError(f'Invalid patient ID: {row["Patient ID"]!r}')
            status = normalize_status(row['COVID status'])
            archive = Path(row['Archive'].strip()).name.lower()
            if not ARCHIVE_PATTERN.fullmatch(archive):
                raise ValueError(f'Unexpected archive filename: {archive!r}')
            if (status == 'Negative') != ('_neg_' in archive):
                raise ValueError(f'{m.group(1)}: status {status} does not match archive {archive}')
            groups[(status, archive)].add(m.group(1).upper())
    return groups


def load_plan(path, groups):
    """Archive order from the plan CSV (most patients first); fall back to mapping order."""
    order = []
    if path.is_file():
        with path.open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                key = (normalize_status(row['COVID status']), row['Archive'].strip().lower())
                if key in groups:
                    order.append(key)
    for key in sorted(groups, key=lambda k: (k[0], -len(groups[k]), k[1])):
        if key not in order:
            order.append(key)
    return order

#added to download the AP/PA views only
def load_frontal_plan(path):
    """groups + archive order from frontal_xray_only_plan.csv (Step 1 output)."""
    groups, order = {}, []
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            status = normalize_status(row['COVID status'])
            m = re.search(r'covid19_(?:posi|pos|neg)_subjects_part[a-z]+', row['Archive to download'].lower())
            if not m:
                raise ValueError(f'Unexpected archive filename: {row["Archive to download"]!r}')
            archive = m.group(0) + '.tar.gz'
            if not ARCHIVE_PATTERN.fullmatch(archive):
                raise ValueError(f'Unexpected archive filename: {archive!r}')
            key = (status, archive)
            groups[key] = {p.strip().upper() for p in row['Selected patient IDs'].split(';') if p.strip()}
            order.append(key)
    return groups, order
# ------------------------------------------------------------------ tar helpers
def is_xray(name):
    return bool(re.search(r'_vp-(?:ap|pa)_.*(?:cr|dx)\.png$', name.lower()))   # frontal AP/PA only


def safe_member_name(name):
    p = Path(name)
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name


def destination_for(root, status, pid, member_name):
    s = SESSION_PATTERN.search(member_name)
    session = ('ses-' + s.group(1).upper()) if s else 'unknown_session'
    cohort = 'covidPositive' if status == 'Positive' else 'covidNegative'
    return root / 'data' / cohort / 'xray' / ('sub-' + pid) / session / Path(member_name).name


def selected_members(tar, patients):
    """Yield (member, pid) for X-ray PNGs of selected patients."""
    for member in tar:
        if not member.isfile():
            continue
        if not safe_member_name(member.name):
            raise ValueError(f'Unsafe archive member: {member.name}')
        if not is_xray(member.name):
            continue
        m = PATIENT_PATTERN.search(member.name)
        if m and m.group(1).upper() in patients:
            yield member, m.group(1).upper()


def extract_xrays(path, root, status, patients):
    """Single streaming pass: write selected X-rays atomically, return per-patient counts."""
    extracted = existing = 0
    found = defaultdict(int)
    with tarfile.open(path, mode='r|gz') as tar:      # reading to the end also checks gzip CRC
        for member, pid in selected_members(tar, patients):
            found[pid] += 1
            dest = destination_for(root, status, pid, member.name)
            if dest.is_file() and dest.stat().st_size == member.size:
                existing += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise OSError(f'Cannot read {member.name}')
            fd, tmp = tempfile.mkstemp(prefix='.partial_', dir=dest.parent)
            try:
                with os.fdopen(fd, 'wb') as out:
                    remaining = member.size
                    while remaining:
                        chunk = stream.read(min(CHUNK, remaining))
                        if not chunk:
                            raise EOFError(f'Truncated image: {member.name}')
                        out.write(chunk)
                        remaining -= len(chunk)
                os.replace(tmp, dest)
                extracted += 1
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    return extracted, existing, found


def verify_xrays(path, root, status, patients):
    """Second pass: every selected X-ray in the archive must exist on disk at the right size."""
    count = 0
    with tarfile.open(path, mode='r|gz') as tar:
        for member, pid in selected_members(tar, patients):
            count += 1
            dest = destination_for(root, status, pid, member.name)
            if not dest.is_file() or dest.stat().st_size != member.size:
                return False, count, f'Missing or wrong-size file: {dest}'
    return (count > 0), count, '' if count else 'No selected X-rays found in archive'


# ------------------------------------------------------------------ download
def remote_size(url):
    with urlopen(Request(url, method='HEAD', headers=UA), timeout=90) as r:
        size = r.headers.get('Content-Length', '')
        if r.status != 200 or not size.isdigit() or int(size) < 1024:
            raise ValueError(f'Bad server response: HTTP {r.status}, length={size!r}')
        return int(size)


def download(url, target, attempts):
    """Resumable download to target (via target.partial). Returns target."""
    expected = remote_size(url)
    if target.is_file() and target.stat().st_size == expected:
        print(f'  Reusing existing {target.name}', flush=True)
        return target
    partial = target.with_name(target.name + '.partial')
    free = shutil.disk_usage(target.parent).free
    have = partial.stat().st_size if partial.exists() else 0
    if free < (expected - have) + 2 * 1024**3:
        raise RuntimeError(f'Not enough disk space: need ~{(expected - have) / 1e9:.1f} GB + 2 GB headroom, '
                           f'have {free / 1e9:.1f} GB')
    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have > expected:
            partial.unlink()
            have = 0
        print(f'  Download attempt {attempt}/{attempts}: {have / 1e9:.2f}/{expected / 1e9:.2f} GB', flush=True)
        headers = dict(UA)
        if have:
            headers['Range'] = f'bytes={have}-'
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as r:
                mode = 'ab' if (have and r.status == 206) else 'wb'
                done = have if mode == 'ab' else 0
                last = time.time()
                with partial.open(mode) as out:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 15:
                            print(f'    {done / expected:6.1%}  {done / 1e9:.2f} GB', flush=True)
                            last = time.time()
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f'  Interrupted: {exc}; retrying in 10 s', flush=True)
            time.sleep(10)
            continue
        if partial.stat().st_size == expected:
            os.replace(partial, target)
            return target
        print(f'  Incomplete ({partial.stat().st_size:,}/{expected:,} bytes); resuming', flush=True)
    raise RuntimeError(f'Download failed after {attempts} attempts; partial file kept for resume')


# ------------------------------------------------------------------ state / report
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def atomic_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def selection_hash(patients):
    return hashlib.sha256(json.dumps(sorted(patients)).encode()).hexdigest()


def save_report(path, rows):
    fields = ['COVID status', 'Archive', 'Status', 'Selected patients', 'Patients with X-rays',
              'Extracted files', 'Already existing files', 'Verified files', 'Detail']
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT, help="Project folder (default: this script's folder)")
    ap.add_argument('--downloads', type=Path, help='Where archives are temporarily stored (default: <root>/archives)')
    ap.add_argument('--mapping', type=Path, help='Default: <root>/xray_work/xray_only_mapping.csv')
    ap.add_argument('--plan', type=Path, help='Frontal plan CSV from Step 1; if given, replaces --mapping')         #added for downloading AP PA only
    ap.add_argument('--only', choices=['negative', 'positive', 'all'], default='all')
    ap.add_argument('--target-patients', type=int, default=0,
                    help='Stop a class once this many patients are extracted (0 = no limit)')
    ap.add_argument('--max-archives', type=int, default=0, help='Process at most this many archives this run (0 = no limit)')
    ap.add_argument('--attempts', type=int, default=5)
    ap.add_argument('--dry-run', action='store_true', help='Show the plan; download/extract/delete nothing')
    ap.add_argument('--keep-archives', action='store_true', help='Do not delete archives after verified extraction')
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    work = root / 'xray_work'
    work.mkdir(parents=True, exist_ok=True)
    downloads = (args.downloads or root / 'archives').expanduser().resolve()
    downloads.mkdir(parents=True, exist_ok=True)

    state_path = work / 'xray_extraction_state.json'
    report_path = work / 'xray_extraction_report.csv'

    if args.plan:
        groups, order = load_frontal_plan(args.plan.expanduser().resolve())
    else:
        mapping = (args.mapping or work / 'xray_only_mapping.csv').expanduser().resolve()
        if not mapping.is_file():
            ap.error(f'Mapping not found: {mapping}  (run build_xray_only_mapping.py first)')
        groups = load_mapping(mapping)
        order = load_plan(work / 'xray_only_archives.csv', groups)

    wanted = {'negative': ['Negative'], 'positive': ['Positive'], 'all': ['Negative', 'Positive']}[args.only]
    order = [k for k in order if k[0] in wanted]

    state = read_json(state_path)
    done_patients = defaultdict(set)
    for key, info in state.items():
        if info.get('complete'):
            done_patients[key.split('/')[0]].update(info.get('patients', []))

    rows, failures, processed = [], 0, 0
    for status, archive in order:
        patients = groups[(status, archive)]
        key = f'{status}/{archive}'
        digest = selection_hash(patients)
        rec = {'COVID status': status, 'Archive': archive, 'Status': '', 'Selected patients': len(patients),
               'Patients with X-rays': 0, 'Extracted files': 0, 'Already existing files': 0,
               'Verified files': 0, 'Detail': ''}
        prev = state.get(key, {})
        if prev.get('complete') and prev.get('selection_hash') == digest:
            rec.update(Status='SKIPPED_COMPLETE', **{'Patients with X-rays': len(prev.get('patients', []))})
            rows.append(rec)
            continue
        if args.target_patients and len(done_patients[status]) >= args.target_patients:
            rec.update(Status='SKIPPED_TARGET_REACHED')
            rows.append(rec)
            continue
        if args.max_archives and processed >= args.max_archives:
            rec.update(Status='NOT_RUN_MAX_ARCHIVES')
            rows.append(rec)
            continue
        print(f'\n[{status}] {archive}: {len(patients)} X-ray-only patients '
              f'(have {len(done_patients[status])} {status} so far)', flush=True)
        if args.dry_run:
            rec.update(Status='WOULD_PROCESS')
            rows.append(rec)
            processed += 1
            continue
        processed += 1
        target = downloads / archive
        try:
            download(BASE[status] + archive, target, args.attempts)
            print('  Extracting X-rays...', flush=True)
            extracted, existing, found = extract_xrays(target, root, status, patients)
            print('  Verifying...', flush=True)
            ok, count, reason = verify_xrays(target, root, status, patients)
            if not ok:
                raise RuntimeError(reason)
            missing = sorted(patients - set(found))
            state = read_json(state_path)
            state[key] = {'complete': True, 'selection_hash': digest, 'patients': sorted(found),
                          'missing_patients': missing, 'extracted_files': extracted,
                          'already_existing_files': existing, 'verified_files': count,
                          'finished_utc': datetime.now(timezone.utc).isoformat()}
            atomic_json(state_path, state)                       # checkpoint BEFORE deleting
            done_patients[status].update(found)
            rec.update(Status='COMPLETE', **{'Patients with X-rays': len(found), 'Extracted files': extracted,
                                             'Already existing files': existing, 'Verified files': count},
                       Detail=f'missing in archive: {";".join(missing)}' if missing else '')
            print(f'  OK: {extracted} extracted, {existing} already present, {count} verified, '
                  f'{len(found)} patients', flush=True)
            if not args.keep_archives:
                target.unlink()
                print(f'  Deleted {target.name}', flush=True)
        except KeyboardInterrupt:
            print('\nStopped by user. Rerun the same command to resume.', flush=True)
            rec.update(Status='INTERRUPTED')
            rows.append(rec)
            save_report(report_path, rows)
            return 130
        except Exception as exc:
            failures += 1
            rec.update(Status='FAILED', Detail=str(exc))
            print(f'  ERROR: {exc}', file=sys.stderr, flush=True)
        rows.append(rec)
        save_report(report_path, rows)

    save_report(report_path, rows)
    print(f'\nPatients extracted so far: ' +
          ', '.join(f'{s} {len(done_patients[s])}' for s in ('Positive', 'Negative')))
    print(f'Report: {report_path}\nFailures this run: {failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
