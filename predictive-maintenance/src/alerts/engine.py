"""Alert engine - turns prediction results into actionable alerts."""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from config.config import ALERT_COOLDOWN_SECONDS
from src.prediction.engine import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    alert_id: str
    timestamp: str
    equipment_id: str
    risk_level: str
    failure_probability: float
    sensor_readings: dict
    reason: str
    recommendation: str
    status: str = "OPEN"

    def to_dict(self) -> dict:
        return asdict(self)


class AlertEngine:
    def __init__(self, cooldown_seconds: int = ALERT_COOLDOWN_SECONDS):
        self.cooldown = cooldown_seconds
        self._last: dict[tuple[str, str], datetime] = {}

    def _should_suppress(self, equipment_id: str, risk_level: str) -> bool:
        key = (equipment_id, risk_level)
        last = self._last.get(key)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last).total_seconds() < self.cooldown

    def evaluate(self, result: PredictionResult, equipment_id: str = "DEMO-MACHINE-01") -> Alert | None:
        if not result.failure_predicted:
            return None
        if self._should_suppress(equipment_id, result.risk_level):
            logger.info("Suppressed duplicate alert for %s (%s)", equipment_id, result.risk_level)
            return None
        self._last[(equipment_id, result.risk_level)] = datetime.now(timezone.utc)
        return Alert(
            alert_id=str(uuid.uuid4()),
            timestamp=result.timestamp,
            equipment_id=equipment_id,
            risk_level=result.risk_level,
            failure_probability=result.failure_probability,
            sensor_readings=result.input,
            reason=(f"Predicted failure probability {result.failure_probability:.3f} "
                    f"crossed decision threshold {result.threshold:.3f}."),
            recommendation=result.recommendation, status="OPEN",
        )