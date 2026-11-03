"""Schema and sanity validation for the AI4I dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from config.config import (
    CATEGORICAL_FEATURES, FAILURE_MODE_COLS, ID_COLS, NUMERICAL_FEATURES, TARGET_COL,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = (
    ID_COLS + CATEGORICAL_FEATURES + NUMERICAL_FEATURES + FAILURE_MODE_COLS + [TARGET_COL]
)


class DatasetValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    path: str
    rows: int
    columns: int
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    missing_values_per_column: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    type_values: List[str] = field(default_factory=list)
    target_counts: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataset(path, strict: bool = True) -> ValidationReport:
    path = Path(path)
    if not path.exists():
        raise DatasetValidationError(f"Dataset file not found: {path}")
    if path.stat().st_size == 0:
        raise DatasetValidationError(f"Dataset file is empty: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DatasetValidationError(f"Failed to read CSV at {path}: {e}") from e
    report = ValidationReport(path=str(path), rows=len(df), columns=df.shape[1])
    report.missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.extra_columns = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    report.missing_values_per_column = df.isna().sum().to_dict()
    report.duplicate_rows = int(df.duplicated().sum())
    cat = CATEGORICAL_FEATURES[0]
    if cat in df.columns:
        report.type_values = sorted(df[cat].dropna().astype(str).unique().tolist())
    if TARGET_COL in df.columns:
        report.target_counts = df[TARGET_COL].value_counts().to_dict()
    if report.missing_columns:
        report.errors.append(f"Missing required columns: {report.missing_columns}")
    if report.rows == 0:
        report.errors.append("Dataset has zero rows")
    if TARGET_COL in df.columns:
        uniq = set(df[TARGET_COL].dropna().unique().tolist())
        if not uniq.issubset({0, 1}):
            report.errors.append(f"Target '{TARGET_COL}' must be binary 0/1, found {sorted(uniq)}")
    if strict and not report.is_valid:
        raise DatasetValidationError("; ".join(report.errors))
    return report