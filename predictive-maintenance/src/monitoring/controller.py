"""Monitoring controller with start/pause/stop/step API for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.monitoring.stream import SensorStream
from src.prediction.engine import PredictionEngine, PredictionResult


@dataclass
class MonitorState:
    running: bool = False
    paused: bool = False
    position: int = 0
    total: int = 0
    readings: list[PredictionResult] = field(default_factory=list)


class MonitorController:
    def __init__(self, engine: PredictionEngine, stream: SensorStream):
        self.engine = engine
        self.stream = stream
        self.state = MonitorState(total=len(stream))

    def start(self) -> None:
        self.state.running = True; self.state.paused = False

    def pause(self) -> None:
        self.state.paused = True

    def stop(self) -> None:
        self.state.running = False; self.state.paused = False

    def reset(self) -> None:
        self.stream.reset(); self.state = MonitorState(total=len(self.stream))

    def step(self) -> PredictionResult | None:
        row = self.stream.next_reading()
        if row is None:
            self.state.running = False
            return None
        result = self.engine.predict_one(row)
        self.state.readings.append(result)
        self.state.position = self.stream.position
        return result

    def recent(self, n: int = 50) -> list[PredictionResult]:
        return self.state.readings[-n:]