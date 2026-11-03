"""Prediction engine — loads saved model, validates inputs, returns structured results."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.config import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERICAL_FEATURES,
    REFERENCE_RANGE_HIGH_Q, REFERENCE_RANGE_LOW_Q,
)
from src.models.evaluator import load_model, load_threshold

logger = logging.getLogger(__name__)


class PredictionError(ValueError):
    pass


@dataclass
class PredictionResult:
    timestamp: str
    input: dict
    failure_probability: float
    failure_predicted: bool
    risk_level: str
    threshold: float
    operating_range_notes: list[str] = field(default_factory=list)
    recommendation: str = ""


_RISK_RECOMMENDATIONS = {
    "LOW": ("Continue normal operation. No immediate maintenance action is required "
            "based on this advisory prediction."),
    "MEDIUM": ("Monitor the machine more closely over the next shifts. Review recent "
               "operating trends and verify sensor readings. Schedule a routine inspection "
               "with qualified maintenance personnel if the trend continues."),
    "HIGH": ("Elevated failure risk detected. Verify sensor readings, review recent operating "
             "history, and schedule an assessment by qualified maintenance personnel. Record the "
             "inspection outcome. This is an advisory signal, not a certified industrial safety alarm."),
}


def _reference_ranges_from(df):
    if df is None or df.empty:
        return {}
    out = {}
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            out[col] = (float(df[col].quantile(REFERENCE_RANGE_LOW_Q)),
                        float(df[col].quantile(REFERENCE_RANGE_HIGH_Q)))
    return out


class PredictionEngine:
    def __init__(self, model: Any, threshold: float, reference_df=None):
        self.model = model
        self.threshold = threshold
        self.reference_ranges = _reference_ranges_from(reference_df)

    @classmethod
    def from_artifacts(cls, reference_csv: Path | None = None) -> "PredictionEngine":
        model = load_model(); threshold = load_threshold()
        ref_df = pd.read_csv(reference_csv) if reference_csv and Path(reference_csv).exists() else None
        return cls(model=model, threshold=threshold, reference_df=ref_df)

    @staticmethod
    def validate_row(row: dict) -> dict:
        if not isinstance(row, dict):
            raise PredictionError("Input must be a dictionary of features.")
        missing = [c for c in FEATURE_COLUMNS if c not in row]
        if missing:
            raise PredictionError(f"Missing required fields: {missing}")
        cleaned = {}
        for col in CATEGORICAL_FEATURES:
            v = row[col]
            if not isinstance(v, str) or v.strip() == "":
                raise PredictionError(f"Field '{col}' must be a non-empty string.")
            cleaned[col] = v.strip()
        for col in NUMERICAL_FEATURES:
            v = row[col]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise PredictionError(f"Field '{col}' must be numeric (got {v!r}).")
            if not np.isfinite(fv):
                raise PredictionError(f"Field '{col}' must be finite (got {v!r}).")
            cleaned[col] = fv
        return cleaned

    def _risk_level(self, prob: float) -> str:
        if prob >= self.threshold:
            return "HIGH"
        if prob >= self.threshold * 0.5:
            return "MEDIUM"
        return "LOW"

    def _range_notes(self, row: dict) -> list[str]:
        notes = []
        for col, (lo, hi) in self.reference_ranges.items():
            v = row.get(col)
            if v is None:
                continue
            if v < lo:
                notes.append(f"{col}={v:.3f} is below the 1st percentile ({lo:.3f}) of reference "
                             f"operating data - statistical reference only, not a manufacturer safety limit.")
            elif v > hi:
                notes.append(f"{col}={v:.3f} is above the 99th percentile ({hi:.3f}) of reference "
                             f"operating data - statistical reference only, not a manufacturer safety limit.")
        return notes

    def predict_one(self, row: dict) -> PredictionResult:
        cleaned = self.validate_row(row)
        X = pd.DataFrame([cleaned])[FEATURE_COLUMNS]
        prob = float(self.model.predict_proba(X)[0, 1])
        predicted = prob >= self.threshold
        risk = self._risk_level(prob)
        return PredictionResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            input=cleaned, failure_probability=prob, failure_predicted=bool(predicted),
            risk_level=risk, threshold=self.threshold,
            operating_range_notes=self._range_notes(cleaned),
            recommendation=_RISK_RECOMMENDATIONS[risk],
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise PredictionError("Batch input is empty.")
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise PredictionError(f"Batch CSV missing columns: {missing}")
        X = df[FEATURE_COLUMNS].copy()
        for col in NUMERICAL_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[NUMERICAL_FEATURES].isna().any().any():
            raise PredictionError("Batch CSV contains non-numeric values in numeric columns.")
        probs = self.model.predict_proba(X)[:, 1]
        out = df.copy()
        out["failure_probability"] = probs
        out["failure_predicted"] = probs >= self.threshold
        out["risk_level"] = [self._risk_level(float(p)) for p in probs]
        return out

    def to_dict(self, result: PredictionResult) -> dict:
        return asdict(result)