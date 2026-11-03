"""Validate + run EDA + save a processed copy of the dataset."""
from __future__ import annotations

import sys
import pandas as pd
from config.config import PROCESSED_CSV_PATH, RAW_CSV_PATH, configure_logging
from src.data.validator import validate_dataset
from src.preprocessing.eda import run_eda

logger = configure_logging(__name__)


def main() -> int:
    report = validate_dataset(RAW_CSV_PATH)
    logger.info("Schema OK: %d rows", report.rows)
    df = pd.read_csv(RAW_CSV_PATH).drop_duplicates().reset_index(drop=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    logger.info("Processed dataset saved -> %s (%d rows)", PROCESSED_CSV_PATH, len(df))
    run_eda(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())