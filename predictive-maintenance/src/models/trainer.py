"""Train, tune threshold, compare, and save the final model."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config.config import (
    CATEGORICAL_FEATURES, COMPARISON_PATH, DEFAULT_THRESHOLD, EVAL_METRICS_PATH,
    FEATURE_COLUMNS, FINAL_MODEL_PATH, MODEL_METADATA_PATH, NUMERICAL_FEATURES,
    RANDOM_SEED, TARGET_COL, THRESHOLD_PATH,
)
from src.preprocessing.pipeline import build_preprocessor, select_features
from src.preprocessing.splits import stratified_three_way_split

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    roc_auc: float
    avg_precision: float
    tn: int
    fp: int
    fn: int
    tp: int


def _make_candidates() -> dict[str, Any]:
    return {
        "DummyClassifier": DummyClassifier(strategy="prior", random_state=RANDOM_SEED),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=300, min_samples_split=5,
                                                         class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=300, min_samples_split=5,
                                                     class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
    }


def _safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _metrics_at(y_true, y_prob, threshold: float) -> ModelResult:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ModelResult(
        name="", threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        f2=float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        roc_auc=_safe_auc(y_true, y_prob), avg_precision=_safe_ap(y_true, y_prob),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def tune_threshold_f2(y_true, y_prob) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f2 = (5 * prec * rec) / (4 * prec + rec + 1e-12)
    f2 = np.nan_to_num(f2, nan=-1.0)
    best_idx = int(np.argmax(f2[:-1])) if len(thr) else 0
    return float(thr[best_idx]) if len(thr) else DEFAULT_THRESHOLD


def train_all(df: pd.DataFrame, models_dir: Path | None = None):
    train_df, val_df, test_df = stratified_three_way_split(df)
    logger.info("Split sizes train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    X_train = select_features(train_df); y_train = train_df[TARGET_COL].astype(int).values
    X_val = select_features(val_df); y_val = val_df[TARGET_COL].astype(int).values
    X_test = select_features(test_df); y_test = test_df[TARGET_COL].astype(int).values
    comparison_rows = []; fitted = {}; thresholds = {}
    for name, estimator in _make_candidates().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        logger.info("Training %s ...", name)
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        thr = tune_threshold_f2(y_val, val_prob)
        m = _metrics_at(y_val, val_prob, thr); m.name = name
        comparison_rows.append(asdict(m)); fitted[name] = pipe; thresholds[name] = thr
        logger.info("  %s val: F2=%.3f F1=%.3f P=%.3f R=%.3f thr=%.3f",
                    name, m.f2, m.f1, m.precision, m.recall, thr)
    comparison_df = pd.DataFrame(comparison_rows).sort_values("f2", ascending=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    best_name = comparison_df.iloc[0]["name"]
    best_thr = thresholds[best_name]
    logger.info("Best model on validation F2: %s (thr=%.3f)", best_name, best_thr)
    refit_pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", _make_candidates()[best_name])])
    X_fit = pd.concat([X_train, X_val], ignore_index=True)
    y_fit = np.concatenate([y_train, y_val])
    logger.info("Refitting %s on train+val (%d rows)", best_name, len(X_fit))
    refit_pipe.fit(X_fit, y_fit)
    test_prob = refit_pipe.predict_proba(X_test)[:, 1]
    test_metrics = _metrics_at(y_test, test_prob, best_thr)
    test_metrics.name = f"{best_name} (refit on train+val)"
    test_dict = asdict(test_metrics)
    EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_PATH.write_text(json.dumps(test_dict, indent=2))
    THRESHOLD_PATH.write_text(json.dumps({"threshold": best_thr, "model": best_name}, indent=2))
    joblib.dump(refit_pipe, FINAL_MODEL_PATH)
    feature_names_out = refit_pipe.named_steps["preprocessor"].get_feature_names_out().tolist()
    metadata = {
        "model_name": best_name, "threshold": best_thr,
        "trained_on_rows": int(len(X_fit)),
        "raw_feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "transformed_feature_names": feature_names_out,
        "random_seed": RANDOM_SEED,
        "sklearn_pipeline_steps": ["preprocessor", "model"],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model -> %s", FINAL_MODEL_PATH)
    logger.info("Test metrics: %s", json.dumps(test_dict, indent=2))
    return refit_pipe, best_thr, comparison_df, test_dict