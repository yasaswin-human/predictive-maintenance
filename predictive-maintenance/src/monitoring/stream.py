"""Simulated sensor stream - replays genuine sensor measurements from the dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from config.config import FEATURE_COLUMNS, PROCESSED_CSV_PATH, RAW_CSV_PATH

logger = logging.getLogger(__name__)


class SensorStream:
    def __init__(self, csv_path: Path | None = None, shuffle: bool = False, seed: int = 42):
        path = csv_path or (PROCESSED_CSV_PATH if PROCESSED_CSV_PATH.exists() else RAW_CSV_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(f"Sensor CSV not found: {path}")
        df = pd.read_csv(path)
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Sensor CSV missing columns: {missing}")
        self.df = df[FEATURE_COLUMNS].copy()
        if shuffle:
            self.df = self.df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        self._index = 0

    def __len__(self) -> int:
        return len(self.df)

    def next_reading(self) -> dict | None:
        if self._index >= len(self.df):
            return None
        row = self.df.iloc[self._index].to_dict()
        self._index += 1
        return row

    def reset(self) -> None:
        self._index = 0

    @property
    def position(self) -> int:
        return self._index

    def history(self, n: int = 50) -> pd.DataFrame:
        start = max(0, self._index - n)
        return self.df.iloc[start:self._index].copy()


def replay_generator(csv_path: Path | None = None) -> Iterator[dict]:
    stream = SensorStream(csv_path=csv_path)
    while (row := stream.next_reading()) is not None:
        yield row