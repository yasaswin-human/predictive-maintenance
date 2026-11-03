from pathlib import Path
import pandas as pd
import pytest
from src.data.validator import DatasetValidationError, validate_dataset


def test_validate_dataset_missing_file(tmp_path: Path):
    with pytest.raises(DatasetValidationError):
        validate_dataset(tmp_path / "nope.csv")


def test_validate_dataset_empty_file(tmp_path: Path):
    p = tmp_path / "empty.csv"; p.write_text("")
    with pytest.raises(DatasetValidationError):
        validate_dataset(p)


def test_validate_dataset_missing_columns(tmp_path: Path):
    p = tmp_path / "bad.csv"; pd.DataFrame({"Type": ["M"]}).to_csv(p, index=False)
    with pytest.raises(DatasetValidationError):
        validate_dataset(p)


def test_validate_dataset_happy(tmp_path: Path):
    df = pd.DataFrame({
        "UDI": [1, 2], "Product ID": ["A", "B"], "Type": ["M", "L"],
        "Air temperature [K]": [298.0, 300.0], "Process temperature [K]": [308.0, 310.0],
        "Rotational speed [rpm]": [1500, 1400], "Torque [Nm]": [40.0, 30.0],
        "Tool wear [min]": [0, 100], "Machine failure": [0, 1],
        "TWF": [0, 0], "HDF": [0, 1], "PWF": [0, 0], "OSF": [0, 0], "RNF": [0, 0],
    })
    p = tmp_path / "ok.csv"; df.to_csv(p, index=False)
    report = validate_dataset(p)
    assert report.is_valid and report.rows == 2