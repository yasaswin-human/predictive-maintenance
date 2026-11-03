"""Central configuration for the AI4I Predictive Maintenance project."""
from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
ALERTS_DIR = DATA_DIR / "alerts"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
METRICS_DIR = MODELS_DIR / "metrics"

LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, ALERTS_DIR,
           ARTIFACTS_DIR, METRICS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_URL = os.getenv(
    "AI4I_DATASET_URL",
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
)
DATASET_ZIP_NAME = "ai4i2020.zip"
DATASET_CSV_NAME = "ai4i2020.csv"
RAW_CSV_PATH = RAW_DIR / DATASET_CSV_NAME
PROCESSED_CSV_PATH = PROCESSED_DIR / "processed.csv"

TARGET_COL = "Machine failure"
ID_COLS = ["UDI", "Product ID"]
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LEAKAGE_COLS = ID_COLS + FAILURE_MODE_COLS + [TARGET_COL]

CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
assert abs((TRAIN_SIZE + VAL_SIZE + TEST_SIZE) - 1.0) < 1e-9

FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
EVAL_METRICS_PATH = METRICS_DIR / "test_metrics.json"

ALERTS_DB_PATH = ALERTS_DIR / "alerts.sqlite"
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))
DEFAULT_THRESHOLD = 0.5

REFERENCE_RANGE_LOW_Q = 0.01
REFERENCE_RANGE_HIGH_Q = 0.99

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging(name: str | None = None) -> logging.Logger:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logging.getLogger(name) if name else root