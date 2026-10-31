#!/usr/bin/env python3
"""Download BIMCV-COVID-19 archive parts with resume and integrity checks.

This utility downloads source TAR.GZ archives only. It does not extract files,
select patients, or modify a dataset directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URLS = {
    "negative": "https://b2drop.bsc.es/nextcloud/public.php/dav/files/BIMCV-COVID19-cIter_1_2-Negative/",
    "positive": "https://b2drop.bsc.es/nextcloud/public.php/dav/files/BIMCV-COVID19-cIter_1_2_3/",
}
PARTS = {
    "negative": "ao ax az bf bg bn cc ck ct".split(),
    "positive": "kb lf ma os ot pd qe qj qq qr qz rg rl sn sy vq".split(),
}
GIB = 1024**3


def archive_name(kind: str, part: str) -> str:
    prefix = "covid19_neg" if kind == "negative" else "covid19_posi"
    return f"{prefix}_subjects_part{part}.tar.gz"


def archive_url(kind: str, filename: str) -> str:
    return BASE_URLS[kind] + filename


def remote_size(url: str) -> int:
    request = Request(url, method="HEAD", headers={"User-Agent": "BIMCV-research-downloader/1.0"})
    with urlopen(request, timeout=90) as response:
        value = response.headers.get("Content-Length")
        if response.status != 200 or not value or not value.isdigit():
            raise RuntimeError(f"unexpected HEAD response: status={response.status}, length={value}")
        return int(value)


def gzip_valid(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as stream:
            while stream.read(4 * 1024 * 1024):
                pass
        return True
    except (OSError, EOFError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / GIB


def verify_marker(marker: Path, expected_bytes: int) -> bool:
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("bytes") == expected_bytes and data.get("gzip_ok") is True


def write_marker(marker: Path, target: Path, expected_bytes: int, include_sha256: bool) -> None:
    data = {"bytes": expected_bytes, "gzip_ok": True}
    if include_sha256:
        data["sha256"] = sha256(target)
    marker.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def download_one(
    kind: str,
    part: str,
    destination: Path,
    attempts: int,
    min_free_gb: float,
    verify_checksum: bool,
) -> bool:
    filename = archive_name(kind, part)
    url = archive_url(kind, filename)
    target = destination / filename
    partial = destination / (filename + ".partial")
    marker = destination / (filename + ".verified.json")
    expected = remote_size(url)
    required_free = expected / GIB + min_free_gb

    print(f"[{kind}] {filename} ({expected / GIB:.2f} GiB)", flush=True)
    if target.is_file() and target.stat().st_size == expected and verify_marker(marker, expected):
        print("  SKIP: archive already verified", flush=True)
        return True
    if target.is_file() and target.stat().st_size == expected and gzip_valid(target):
        write_marker(marker, target, expected, verify_checksum)
        print("  SKIP: existing archive passed gzip verification", flush=True)
        return True
    if target.exists() and not partial.exists():
        target.replace(partial)
    if free_gib(destination) < required_free:
        raise RuntimeError(
            f"insufficient free space: {free_gib(destination):.2f} GiB available, "
            f"at least {required_free:.2f} GiB required"
        )

    for attempt in range(1, attempts + 1):
        print(f"  download attempt {attempt}/{attempts}; resume file={partial.name}", flush=True)
        command = [
            "curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
            "--retry-delay", "10", "--connect-timeout", "40", "--speed-time", "120",
            "--speed-limit", "1024", "--output", str(partial), url,
        ]
        result = subprocess.run(command, check=False)
        actual = partial.stat().st_size if partial.exists() else 0
        if result.returncode != 0 or actual != expected:
            print(f"  incomplete download: {actual:,}/{expected:,} bytes", flush=True)
            time.sleep(3)
            continue
        print("  verifying gzip stream", flush=True)
        if not gzip_valid(partial):
            corrupt = partial.with_name(partial.name + f".corrupt-{int(time.time())}")
            partial.replace(corrupt)
            print(f"  invalid gzip; preserved as {corrupt.name}", flush=True)
            continue
        partial.replace(target)
        write_marker(marker, target, expected, verify_checksum)
        print(f"  OK: {target}", flush=True)
        return True
    print(f"  FAILED: {filename}; partial data was preserved", flush=True)
    return False


def plan_archives(plan: Path | None, only: str) -> list[tuple[str, str]]:
    if plan is None:
        return [
            (kind, part)
            for kind in ("negative", "positive")
            if only in ("all", kind)
            for part in PARTS[kind]
        ]
    import csv

    selected: list[tuple[str, str]] = []
    with plan.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            kind = row.get("COVID status", row.get("status", "")).strip().lower()
            kind = {"negative": "negative", "positive": "positive"}.get(kind, kind)
            filename = row.get("Archive to download", row.get("Archive", "")).strip()
            matched = next(
                ((candidate_kind, part) for candidate_kind in PARTS for part in PARTS[candidate_kind]
                 if archive_name(candidate_kind, part).lower() == filename.lower()),
                None,
            )
            if matched is None or matched[0] != kind:
                raise ValueError(f"invalid plan row: status={kind!r}, archive={filename!r}")
            if only in ("all", kind) and matched not in selected:
                selected.append(matched)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path.home() / "Downloads" / "BIMCV-COVID-19-archives")
    parser.add_argument("--plan", type=Path, help="Existing planner CSV; downloads its Archive column only")
    parser.add_argument("--only", choices=("all", "negative", "positive"), default="all")
    parser.add_argument("--max-archives", type=int, default=0, help="Stop after this many archives; 0 means no limit")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--sha256", action="store_true", help="Also record archive SHA-256 in verification markers")
    args = parser.parse_args()
    if args.attempts < 1 or args.max_archives < 0 or args.min_free_gb < 0:
        parser.error("attempts must be positive; max-archives and min-free-gb must be nonnegative")
    if shutil.which("curl") is None:
        parser.error("curl is required and was not found on PATH")

    args.dir = args.dir.expanduser().resolve()
    args.dir.mkdir(parents=True, exist_ok=True)
    jobs = plan_archives(args.plan.expanduser() if args.plan else None, args.only)
    if args.max_archives:
        jobs = jobs[:args.max_archives]
    if not jobs:
        parser.error("no archives selected")

    failures = []
    for kind, part in jobs:
        try:
            if not download_one(kind, part, args.dir, args.attempts, args.min_free_gb, args.sha256):
                failures.append(f"{kind}:part{part}")
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
            print(f"  ERROR: {exc}", flush=True)
            failures.append(f"{kind}:part{part}")
    print(f"Finished: {len(jobs) - len(failures)}/{len(jobs)} archives verified.")
    if failures:
        print("Failed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
