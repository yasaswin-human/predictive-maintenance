import pandas as pd
import pytest
from config.config import FEATURE_COLUMNS
from src.monitoring.stream import SensorStream


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({
        "Type": ["M", "L", "H"],
        "Air temperature [K]": [298, 299, 300],
        "Process temperature [K]": [308, 309, 310],
        "Rotational speed [rpm]": [1500, 1600, 1700],
        "Torque [Nm]": [40, 41, 42],
        "Tool wear [min]": [0, 10, 20],
    }).to_csv(p, index=False)
    return p


def test_stream_iterates(csv):
    s = SensorStream(csv_path=csv)
    assert len(s) == 3 and s.next_reading() is not None
    s.next_reading(); s.next_reading()
    assert s.next_reading() is None


def test_stream_reset(csv):
    s = SensorStream(csv_path=csv); s.next_reading(); s.next_reading(); s.reset()
    assert s.position == 0


def test_stream_missing_column(tmp_path):
    p = tmp_path / "bad.csv"; pd.DataFrame({"Type": ["M"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        SensorStream(csv_path=p)