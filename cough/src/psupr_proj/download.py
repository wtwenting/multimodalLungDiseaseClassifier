"""Download and extract the public COUGHVID dataset from Zenodo."""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ZENODO_RECORD_ID = 7024894
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ARCHIVE_NAME = "public_dataset_v3.zip"
CHUNK_SIZE = 1024 * 1024


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, output_dir: Path) -> None:
    """Extract a ZIP archive without allowing paths outside output_dir."""
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (output_dir / member.filename).resolve()
            if destination != output_root and output_root not in destination.parents:
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        archive.extractall(output_dir)


def download_dataset(output_dir: Path, extract: bool = True) -> dict[str, str | int]:
    """Download the Zenodo dataset, resume partial downloads, and extract it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API_URL) as response:
        record = json.load(response)

    files = record.get("files", [])
    dataset_file = next(
        (item for item in files if item.get("key") == ARCHIVE_NAME),
        None,
    )
    if dataset_file is None:
        raise RuntimeError(f"Could not find {ARCHIVE_NAME} in Zenodo record")

    archive_path = output_dir / ARCHIVE_NAME
    partial_path = output_dir / f"{ARCHIVE_NAME}.part"
    expected_size = int(dataset_file["size"])
    expected_md5 = dataset_file["checksum"].removeprefix("md5:")
    download_url = dataset_file["links"]["self"]

    if archive_path.exists() and (
        archive_path.stat().st_size != expected_size
        or _md5(archive_path) != expected_md5
    ):
        archive_path.unlink()

    if not archive_path.exists():
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        request = urllib.request.Request(download_url)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with urllib.request.urlopen(request) as response:
            if offset and response.status != 206:
                offset = 0
                partial_path.unlink(missing_ok=True)
            mode = "ab" if offset else "wb"
            with partial_path.open(mode) as archive:
                shutil.copyfileobj(response, archive, length=CHUNK_SIZE)

        if partial_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Download incomplete: {partial_path.stat().st_size} of {expected_size} bytes"
            )
        if _md5(partial_path) != expected_md5:
            partial_path.unlink()
            raise RuntimeError("Downloaded archive failed the Zenodo MD5 checksum")
        partial_path.replace(archive_path)

    if extract:
        _safe_extract(archive_path, output_dir)

    return {
        "archive": str(archive_path),
        "size": expected_size,
        "extracted_to": str(output_dir),
    }