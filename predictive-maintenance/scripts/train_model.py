"""Train all candidate models, select the best, and save artifacts."""
from __future__ import annotations

import sys
import pandas as pd
from config.config import PROCESSED_CSV_PATH, RAW_CSV_PATH, configure_logging
from src.models.trainer import train_all

logger = configure_logging(__name__)


def main() -> int:
    path = PROCESSED_CSV_PATH if PROCESSED_CSV_PATH.exists() else RAW_CSV_PATH
    logger.info("Training from %s", path)
    train_all(pd.read_csv(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())