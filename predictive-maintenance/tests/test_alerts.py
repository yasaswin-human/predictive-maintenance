from src.alerts.engine import AlertEngine
from src.prediction.engine import PredictionResult


def make_result(prob: float, thr: float = 0.5) -> PredictionResult:
    return PredictionResult(
        timestamp="2024-01-01T00:00:00+00:00", input={"Type": "M"},
        failure_probability=prob, failure_predicted=prob >= thr,
        risk_level="HIGH" if prob >= thr else "LOW",
        threshold=thr, recommendation="test",
    )


def test_no_alert_when_healthy():
    assert AlertEngine().evaluate(make_result(0.1)) is None


def test_alert_when_above_threshold():
    a = AlertEngine().evaluate(make_result(0.9))
    assert a is not None and a.risk_level == "HIGH"


def test_duplicate_suppression():
    eng = AlertEngine(cooldown_seconds=60)
    assert eng.evaluate(make_result(0.9)) is not None
    assert eng.evaluate(make_result(0.95)) is None