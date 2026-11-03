"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_features() -> dict:
    return {
        "Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 0,
    }


@pytest.fixture
def sample_df(sample_features) -> pd.DataFrame:
    return pd.DataFrame([sample_features] * 5)