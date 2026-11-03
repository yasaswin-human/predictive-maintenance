"""SQLite persistence layer for alerts."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from config.config import ALERTS_DB_PATH
from src.alerts.engine import Alert

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    equipment_id TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    failure_probability REAL NOT NULL,
    sensor_readings TEXT NOT NULL,
    reason TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts(risk_level);
"""


@contextmanager
def _conn(db_path: Path = ALERTS_DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA); yield conn; conn.commit()
    finally:
        conn.close()


def save_alert(alert: Alert, db_path: Path = ALERTS_DB_PATH) -> None:
    with _conn(db_path) as c:
        c.execute(
            """INSERT OR REPLACE INTO alerts
               (alert_id, timestamp, equipment_id, risk_level,
                failure_probability, sensor_readings, reason, recommendation, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (alert.alert_id, alert.timestamp, alert.equipment_id, alert.risk_level,
             float(alert.failure_probability), json.dumps(alert.sensor_readings),
             alert.reason, alert.recommendation, alert.status),
        )
    logger.info("Saved alert %s (%s)", alert.alert_id, alert.risk_level)


def fetch_alerts(db_path: Path = ALERTS_DB_PATH, risk_level: str | None = None,
                 since: str | None = None, limit: int = 500) -> pd.DataFrame:
    query = "SELECT * FROM alerts"; clauses, params = [], []
    if risk_level and risk_level != "ALL":
        clauses.append("risk_level = ?"); params.append(risk_level)
    if since:
        clauses.append("timestamp >= ?"); params.append(since)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp DESC LIMIT ?"; params.append(limit)
    with _conn(db_path) as c:
        return pd.read_sql_query(query, c, params=params)


def acknowledge_alert(alert_id: str, db_path: Path = ALERTS_DB_PATH) -> bool:
    with _conn(db_path) as c:
        cur = c.execute("UPDATE alerts SET status='ACKNOWLEDGED' WHERE alert_id=?", (alert_id,))
        return cur.rowcount > 0


def export_alerts_csv(out_path: Path, db_path: Path = ALERTS_DB_PATH) -> Path:
    df = fetch_alerts(db_path=db_path, limit=100_000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False); return out_path