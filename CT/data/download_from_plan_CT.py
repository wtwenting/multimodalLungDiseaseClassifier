#!/usr/bin/env python3
"""Stage 2: read Stage 1 CSV; download -> extract missing frontal pairs -> verify -> delete."""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen
from bimcv_common import (ROOT, ARCHIVE_RE, PID_RE, destination, modality_of,
                          status_of, scan_listings, eligible_missing, write_csv)

BASE = {
    'Negative': 'https://b2drop.bsc.es/nextcloud/public.php/dav/files/BIMCV-COVID19-cIter_1_2-Negative/',
    'Positive': 'https://b2drop.bsc.es/nextcloud/public.php/dav/files/BIMCV-COVID19-cIter_1_2_3/',
}
LOCAL_RE = re.compile(r'^(covid19_(?:neg|posi|pos)_subjects_part[a-z]+)(?:_(?:webdav|fresh|retry|recovered))?\.tar\.gz$', re.I)
FIELDS = ['Timestamp UTC', 'COVID status', 'Archive', 'Status', 'Missing selected files', 'Extracted', 'Extracted X-rays', 'Extracted CTs', 'Already present', 'New completed pairs', 'Free before GiB', 'Free after GiB', 'Archive deleted', 'Detail']
GIB = 1024 ** 3

def free_gib(path):
    return round(shutil.disk_usage(path).free / GIB, 2)

def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def local_pair_counts(root, archives, patient_mods):
    # Count only patients with AP/PA and CT listed in metadata, and files on disk.
    from bimcv_common import patient_inventory
    local = patient_inventory(root, archives, patient_mods)
    return {status: sum(1 for key, mods in local.items() if key[0] == status and {'xray', 'ct'} <= mods)
            for status in ('Negative', 'Positive')}

def ensure_space(root, downloads, min_bytes, required_bytes, context):
    # If downloads and root share a volume, avoid counting the same space twice.
    root_free = shutil.disk_usage(root).free
    dl_free = shutil.disk_usage(downloads).free
    if root_free < min_bytes + required_bytes:
        raise RuntimeError(f'LOW_DISK {context}: dataset free={root_free/GIB:.2f} GiB; require reserve+working={ (min_bytes+required_bytes)/GIB:.2f} GiB')
    if dl_free < min_bytes + required_bytes:
        raise RuntimeError(f'LOW_DISK {context}: downloads free={dl_free/GIB:.2f} GiB; require reserve+working={ (min_bytes+required_bytes)/GIB:.2f} GiB')



def full_archive_check(path):
    """Read full compressed TAR to detect truncation/corruption."""
    try:
        with tarfile.open(path, 'r|gz') as tar:
            for member in tar:
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is None:
                        raise OSError(f'Unreadable member {member.name}')
                    while f.read(1024 * 1024):
                        pass
        return True, ''
    except (OSError, EOFError, tarfile.TarError, ValueError) as e:
        return False, str(e)


def find_local(downloads, archive):
    matches = []
    for path in downloads.iterdir():
        m = LOCAL_RE.fullmatch(path.name) if path.is_file() else None
        if m and m.group(1).lower() + '.tar.gz' == archive.lower():
            matches.append(path)
    return sorted(matches, key=lambda p: (p.name != archive, -p.stat().st_size))


def get_archive(downloads, status, archive, attempts, root, min_free):
    for local in find_local(downloads, archive):
        print('Checking local archive:', local, flush=True)
        valid, reason = full_archive_check(local)
        if valid:
            return local
        print('Invalid local archive, preserved:', reason, flush=True)
    url = BASE[status] + archive
    with urlopen(Request(url, method='HEAD'), timeout=90) as response:
        size = response.headers.get('Content-Length')
        if response.status != 200 or not size or not size.isdigit() or int(size) < 1024:
            raise RuntimeError(f'Unexpected remote size/status for {archive}: {size}')
        expected = int(size)
    ensure_space(root, downloads, min_free, expected, 'before archive download')
    target = downloads / archive
    if target.exists():
        target = downloads / archive.replace('.tar.gz', '_recovered.tar.gz')
    partial = Path(str(target) + '.partial')
    for attempt in range(1, attempts + 1):
        print(f'Downloading {archive}, attempt {attempt}/{attempts}, bytes={expected}', flush=True)
        command = ['curl', '--fail', '--location', '--continue-at', '-', '--retry', '3',
                   '--retry-delay', '10', '--connect-timeout', '40', '--speed-time', '120',
                   '--speed-limit', '1024', '--output', str(partial), url]
        print(f'  Free before download: {free_gib(root):.2f} GiB', flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode or not partial.exists() or partial.stat().st_size != expected:
            print('Download incomplete, preserving partial for retry', flush=True)
            continue
        valid, reason = full_archive_check(partial)
        if valid:
            os.replace(partial, target)
            return target
        print('Archive failed full integrity check:', reason, flush=True)
        partial.rename(partial.with_name(partial.name + '.corrupt-' + str(int(time.time()))))
    raise RuntimeError('Download attempts exhausted; partial files preserved')


def safe_member(name):
    p = Path(name)
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name


def extract_missing(archive_path, root, status, required, min_free, downloads):
    """Extract exact missing members; require all planned outputs match tar sizes."""
    wanted = {member: (pid, mod) for pid, mod, member in required}
    extracted = already = 0
    by_modality = defaultdict(int)
    seen = set()
    with tarfile.open(archive_path, 'r|gz') as tar:
        for item in tar:
            if not item.isfile() or item.name not in wanted:
                continue
            if not safe_member(item.name):
                raise RuntimeError(f'Unsafe TAR member: {item.name}')
            pid, mod = wanted[item.name]
            if PID_RE.search(item.name) is None or modality_of(item.name) != mod:
                raise RuntimeError(f'Metadata/archive mismatch: {item.name}')
            dest = destination(root, status, mod, pid, item.name)
            seen.add(item.name)
            if dest.is_file() and dest.stat().st_size == item.size:
                already += 1
                continue
            # Check available space before every new member, including reserve.
            ensure_space(root, downloads, min_free, item.size, f'before extracting {item.name}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(item)
            if source is None:
                raise RuntimeError(f'Cannot read {item.name}')
            fd, temp = tempfile.mkstemp(prefix='.partial_', dir=dest.parent)
            try:
                with os.fdopen(fd, 'wb') as out:
                    remaining = item.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise EOFError(f'Truncated {item.name}')
                        # Recheck free space while streaming, so large CTs stop safely.
                        if shutil.disk_usage(root).free < min_free + len(chunk):
                            raise RuntimeError('LOW_DISK during extraction; partial output removed, archive retained')
                        out.write(chunk)
                        remaining -= len(chunk)
                os.replace(temp, dest)
                extracted += 1
                by_modality[mod] += 1
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
    missing = set(wanted) - seen
    if missing:
        raise RuntimeError(f'{len(missing)} metadata members absent from archive; first: {next(iter(missing))}')
    # Verify against actual TAR member sizes; do not delete on any failure.
    verified = set()
    with tarfile.open(archive_path, 'r|gz') as tar:
        for item in tar:
            if not item.isfile() or item.name not in wanted:
                continue
            pid, mod = wanted[item.name]
            dest = destination(root, status, mod, pid, item.name)
            if not dest.is_file() or dest.stat().st_size != item.size:
                raise RuntimeError(f'Extraction verification failed: {dest}')
            verified.add(item.name)
    if verified != set(wanted):
        raise RuntimeError('Incomplete verification')
    return extracted, already, dict(by_modality)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--plan', type=Path)
    p.add_argument('--downloads', type=Path, default=Path.home() / 'Downloads')
    p.add_argument('--apply', action='store_true', help='Actually download, extract and delete')
    p.add_argument('--only', choices=['all', 'negative', 'positive'], default='all')
    p.add_argument('--max-archives', type=int, default=0, help='Limit processed archive count (0 = all)')
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--keep-archives', action='store_true')
    p.add_argument('--min-free-gb', type=float, default=20.0)
    p.add_argument('--working-headroom-gb', type=float, default=16.0, help='Extra free space required before each archive for download and extraction')
    args = p.parse_args()
    root = args.root.expanduser()
    plan = args.plan or root / 'dataAnalysis/output/frontal_download/archives_to_download_filteredXrayView_CT.csv'
    if not plan.is_file():
        p.error(f'Plan missing: {plan}. Run plan_frontal_archives.py first.')
    with plan.open(newline='', encoding='utf-8-sig') as f:
        plan_rows = list(csv.DictReader(f))
    archives, patient_mods, counts = scan_listings(root)
    if not counts['members']:
        p.error('No usable metadata listing members found')
    report = root / 'dataAnalysis/output/frontal_download/run_report.csv'
    args.downloads.mkdir(parents=True, exist_ok=True)
    results = []
    processed = failures = 0
    min_free = int(args.min_free_gb * GIB)
    headroom = int(args.working_headroom_gb * GIB)
    if min_free < 0 or headroom < 0:
        p.error('Disk limits must be nonnegative')
    before_counts = local_pair_counts(root, archives, patient_mods)
    print(f'START {now()} | initial free={free_gib(root):.2f} GiB | '
          f'reserve={args.min_free_gb:.1f} GiB | headroom={args.working_headroom_gb:.1f} GiB', flush=True)
    print(f'Initial metadata-eligible local pairs: {before_counts}', flush=True)
    stop_reason = 'PLAN_COMPLETE'
    for row in plan_rows:
        status = row.get('COVID status', '').strip().title()
        archive = row.get('Archive to download', '').strip()
        if status not in BASE or not ARCHIVE_RE.fullmatch(archive) or status_of(archive) != status:
            raise ValueError(f'Invalid plan row: {row}')
        if args.only != 'all' and status.lower() != args.only:
            continue
        if args.max_archives and processed >= args.max_archives:
            stop_reason = 'MAX_ARCHIVES'
            break
        key = (status, archive)
        if key not in archives:
            raise ValueError(f'Archive absent from current metadata: {archive}')
        ids = {pid.strip().upper() for pid in row.get('Selected patient IDs', '').split(';') if pid.strip()}
        if not ids:
            raise ValueError('Plan missing Selected patient IDs; rerun Stage 1 planner')
        selected = {(status, pid) for pid in ids}
        required = eligible_missing(root, {key: archives[key]}, patient_mods, selected).get(key, [])
        before = free_gib(root)
        counts_pre = local_pair_counts(root, archives, patient_mods)
        entry = {'Timestamp UTC': now(), 'COVID status': status, 'Archive': archive,
                 'Status': '', 'Missing selected files': len(required), 'Extracted': 0,
                 'Extracted X-rays': 0, 'Extracted CTs': 0, 'Already present': 0,
                 'New completed pairs': 0, 'Free before GiB': before,
                 'Free after GiB': before, 'Archive deleted': 'No', 'Detail': ''}
        print(f'[{now()}] [{status}] {archive}: missing={len(required)} '
              f'free_before={before:.2f} GiB', flush=True)
        if not required:
            entry.update(Status='SKIPPED_ALREADY_PRESENT', Detail='All planned members already present')
            print('  SKIP: selected members already present', flush=True)
        elif not args.apply:
            entry.update(Status='WOULD_PROCESS', Detail='Dry run; no files changed')
        else:
            if shutil.disk_usage(root).free < min_free + headroom:
                entry.update(Status='STOP_LOW_DISK', Detail='Insufficient reserve + headroom before archive')
                stop_reason = 'LOW_DISK'
            else:
                path = None
                try:
                    path = get_archive(args.downloads, status, archive, args.attempts, root, min_free)
                    print(f'  Download/reuse: SUCCESS ({path.name}); free={free_gib(root):.2f} GiB', flush=True)
                    # Conservative estimate from selected member sizes in TAR, before extraction.
                    wanted = {member for _, _, member in required}
                    planned_bytes = 0
                    with tarfile.open(path, 'r|gz') as tar:
                        for item in tar:
                            if item.isfile() and item.name in wanted:
                                pid, mod = next((pid, mod) for pid, mod, name in required if name == item.name)
                                dest = destination(root, status, mod, pid, item.name)
                                if not (dest.is_file() and dest.stat().st_size == item.size):
                                    planned_bytes += item.size
                    print(f'  Planned extraction: {planned_bytes/GIB:.2f} GiB', flush=True)
                    ensure_space(root, args.downloads, min_free, planned_bytes, 'before extraction')
                    extracted, already, by_mod = extract_missing(path, root, status, required, min_free, args.downloads)
                    entry.update(Status='COMPLETE', Extracted=extracted,
                                 **{'Already present': already, 'Extracted X-rays': by_mod.get('xray', 0),
                                    'Extracted CTs': by_mod.get('ct', 0)})
                    print(f'  Extraction VERIFIED: xray={by_mod.get("xray", 0)} '
                          f'ct={by_mod.get("ct", 0)} already={already}', flush=True)
                    if not args.keep_archives:
                        path.unlink()
                        entry['Archive deleted'] = 'Yes'
                        entry['Detail'] = f'Verified and deleted {path.name}'
                        print(f'  Archive deletion: SUCCESS ({path.name})', flush=True)
                    else:
                        entry['Detail'] = 'Archive retained (--keep-archives)'
                        print('  Archive deletion: SKIPPED (--keep-archives)', flush=True)
                except Exception as exc:
                    message = str(exc)
                    entry.update(Status='STOP_LOW_DISK' if 'LOW_DISK' in message else 'FAILED', Detail=message)
                    if 'LOW_DISK' in message:
                        stop_reason = 'LOW_DISK'
                    else:
                        failures += 1
                    print(f'  ERROR: {message}; archive retained if present', file=sys.stderr, flush=True)
        counts_post = local_pair_counts(root, archives, patient_mods) if args.apply else counts_pre
        new_pairs = max(0, counts_post[status] - counts_pre[status])
        entry['New completed pairs'] = new_pairs
        entry['Free after GiB'] = free_gib(root)
        results.append(entry)
        write_csv(report, FIELDS, results)
        processed += 1
        print(f'  Pairs newly completed ({status}): {new_pairs}; '
              f'current counts={counts_post}; free_after={entry["Free after GiB"]:.2f} GiB; '
              f'status={entry["Status"]}', flush=True)
        if entry['Status'] == 'STOP_LOW_DISK' or (args.apply and shutil.disk_usage(root).free < min_free + headroom):
            stop_reason = 'LOW_DISK'
            break
    final_counts = local_pair_counts(root, archives, patient_mods)
    summary = {'timestamp_utc': now(), 'stop_reason': stop_reason, 'apply': args.apply,
               'initial_pairs': before_counts, 'final_pairs': final_counts,
               'new_pairs': {s: final_counts[s] - before_counts[s] for s in before_counts},
               'processed_archives': processed, 'failed_archives': failures,
               'initial_free_gib': None, 'final_free_gib': free_gib(root),
               'minimum_free_gib': args.min_free_gb,
               'note': 'Pair counts are nonempty AP/PA PNG + CT NIfTI, not validated 3D CT volumes.'}
    summary_path = report.with_name('overnight_summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'FINAL SUMMARY: {json.dumps(summary, sort_keys=True)}', flush=True)
    print(f'Report: {report}; summary: {summary_path}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
