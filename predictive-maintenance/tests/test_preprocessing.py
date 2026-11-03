import pandas as pd
from config.config import LEAKAGE_COLS
from src.preprocessing.pipeline import build_preprocessor, select_features


def test_select_features_excludes_leakage(sample_df: pd.DataFrame):
    sample_df = sample_df.copy()
    for col in LEAKAGE_COLS:
        sample_df[col] = 0
    X = select_features(sample_df)
    for col in LEAKAGE_COLS:
        assert col not in X.columns


def test_preprocessor_fit_transform(sample_df: pd.DataFrame):
    pre = build_preprocessor(); X = select_features(sample_df)
    out = pre.fit_transform(X)
    assert out.shape[0] == len(X) and out.shape[1] > 0


def test_preprocessor_handles_unseen_category(sample_df: pd.DataFrame):
    pre = build_preprocessor(); X = select_features(sample_df); pre.fit(X)
    new = X.copy(); new["Type"] = "Z"
    assert pre.transform(new).shape[0] == len(new)