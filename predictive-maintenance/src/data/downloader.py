"""Download and safely extract the AI4I dataset from UCI."""
from __future__ import annotations

import hashlib
import logging
import shutil
import zipfile
from pathlib import Path

import requests

from config.config import (
    DATASET_CSV_NAME, DATASET_URL, DATASET_ZIP_NAME, RAW_CSV_PATH, RAW_DIR,
)

logger = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    pass


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if dest_dir not in target.parents and target != dest_dir:
                raise DatasetDownloadError(f"Unsafe path in archive: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return dest_dir


def download_dataset(url: str = DATASET_URL, zip_name: str = DATASET_ZIP_NAME,
                     csv_name: str = DATASET_CSV_NAME, raw_dir: Path = RAW_DIR,
                     csv_path: Path = RAW_CSV_PATH, force: bool = False,
                     timeout: int = 60) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and not force:
        logger.info("Using cached dataset at %s", csv_path)
        return csv_path
    zip_path = raw_dir / zip_name
    if force or not zip_path.exists():
        logger.info("Downloading %s -> %s", url, zip_path)
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = zip_path.with_suffix(zip_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if chunk:
                            f.write(chunk)
                tmp.replace(zip_path)
        except requests.RequestException as e:
            raise DatasetDownloadError(
                f"Failed to download from {url}: {e}\n"
                f"Manual procedure:\n"
                f"  1. Open https://archive.ics.uci.edu/dataset/601/"
                f"ai4i+2020+predictive+maintenance+dataset in a browser\n"
                f"  2. Download the zip, extract, and place '{csv_name}' in\n"
                f"     {raw_dir}\n"
            ) from e
    if not zipfile.is_zipfile(zip_path):
        raise DatasetDownloadError(f"File at {zip_path} is not a valid zip archive. Delete it and retry.")
    extract_dir = raw_dir / "_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract_zip(zip_path, extract_dir)
    matches = list(extract_dir.rglob(csv_name))
    if not matches:
        candidates = list(extract_dir.rglob("*.csv"))
        if len(candidates) == 1:
            matches = candidates
            logger.warning("Expected %r, using only CSV found: %r", csv_name, matches[0].name)
        else:
            raise DatasetDownloadError(
                f"Expected '{csv_name}' in archive. Found CSVs: {[c.name for c in candidates]}"
            )
    shutil.copy2(matches[0], csv_path)
    logger.info("Dataset CSV ready at %s", csv_path)
    return csv_path


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()