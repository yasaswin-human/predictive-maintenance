"""Download the official AI4I dataset."""
from __future__ import annotations

import argparse, sys
from config.config import configure_logging
from src.data.downloader import DatasetDownloadError, download_dataset, sha256_of_file

logger = configure_logging(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = download_dataset(force=args.force)
    except DatasetDownloadError as e:
        logger.error("%s", e); return 1
    logger.info("OK -> %s (sha256=%s...)", path, sha256_of_file(path)[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())