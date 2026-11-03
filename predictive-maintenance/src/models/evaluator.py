"""Load a saved model and compute evaluation artifacts for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from config.config import EVAL_METRICS_PATH, FINAL_MODEL_PATH, TARGET_COL, THRESHOLD_PATH
from src.preprocessing.pipeline import select_features


def load_model() -> Any:
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {FINAL_MODEL_PATH}. Run scripts/train_model.py first.")
    return joblib.load(FINAL_MODEL_PATH)


def load_threshold() -> float:
    if THRESHOLD_PATH.exists():
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    return 0.5


def load_test_metrics() -> dict:
    if not EVAL_METRICS_PATH.exists():
        return {}
    return json.loads(EVAL_METRICS_PATH.read_text())


def compute_curves(df: pd.DataFrame) -> dict:
    model = load_model(); threshold = load_threshold()
    X = select_features(df); y = df[TARGET_COL].astype(int).values
    prob = model.predict_proba(X)[:, 1]
    if len(np.unique(y)) < 2:
        return {"roc": None, "pr": None, "threshold": threshold}
    fpr, tpr, _ = roc_curve(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    y_pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
        "confusion_matrix": cm, "threshold": threshold,
    }