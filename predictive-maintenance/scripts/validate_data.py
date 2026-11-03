"""Validate the AI4I CSV schema."""
from __future__ import annotations

import argparse, json, sys
from config.config import RAW_CSV_PATH, configure_logging
from src.data.validator import DatasetValidationError, validate_dataset

logger = configure_logging(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(RAW_CSV_PATH))
    args = parser.parse_args(argv)
    try:
        report = validate_dataset(args.path)
    except DatasetValidationError as e:
        logger.error("Validation failed: %s", e); return 1
    logger.info("Validated: %d rows x %d cols", report.rows, report.columns)
    logger.info("Duplicates: %d", report.duplicate_rows)
    logger.info("Type values: %s", report.type_values)
    logger.info("Target counts: %s", report.target_counts)
    print(json.dumps(report.__dict__, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())