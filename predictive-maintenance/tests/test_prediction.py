import pandas as pd
import pytest
from config.config import FINAL_MODEL_PATH, THRESHOLD_PATH
from src.prediction.engine import PredictionEngine, PredictionError

requires_model = pytest.mark.skipif(
    not FINAL_MODEL_PATH.exists() or not THRESHOLD_PATH.exists(),
    reason="Trained model artifacts missing - run scripts/train_model.py",
)


@pytest.fixture(scope="module")
def engine() -> PredictionEngine:
    return PredictionEngine.from_artifacts()


def test_validate_rejects_missing_fields():
    with pytest.raises(PredictionError):
        PredictionEngine.validate_row({"Type": "M"})


def test_validate_rejects_bad_type():
    row = {"Type": 123, "Air temperature [K]": 300, "Process temperature [K]": 310,
           "Rotational speed [rpm]": 1500, "Torque [Nm]": 40, "Tool wear [min]": 0}
    with pytest.raises(PredictionError):
        PredictionEngine.validate_row(row)


def test_validate_rejects_non_numeric():
    row = {"Type": "M", "Air temperature [K]": "hot", "Process temperature [K]": 310,
           "Rotational speed [rpm]": 1500, "Torque [Nm]": 40, "Tool wear [min]": 0}
    with pytest.raises(PredictionError):
        PredictionEngine.validate_row(row)


@requires_model
def test_predict_one(engine, sample_features):
    r = engine.predict_one(sample_features)
    assert 0.0 <= r.failure_probability <= 1.0
    assert r.risk_level in {"LOW", "MEDIUM", "HIGH"} and r.timestamp


@requires_model
def test_predict_batch(engine, sample_df):
    out = engine.predict_batch(sample_df)
    assert "failure_probability" in out.columns and len(out) == len(sample_df)


@requires_model
def test_predict_batch_rejects_empty(engine):
    with pytest.raises(PredictionError):
        engine.predict_batch(pd.DataFrame())