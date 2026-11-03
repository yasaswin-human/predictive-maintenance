"""Reproducible stratified train / validation / test split."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import RANDOM_SEED, TARGET_COL, TEST_SIZE, TRAIN_SIZE, VAL_SIZE


def stratified_three_way_split(df: pd.DataFrame, target: str = TARGET_COL,
                               seed: int = RANDOM_SEED):
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_SIZE + TEST_SIZE), stratify=df[target],
        random_state=seed, shuffle=True,
    )
    val_share_of_temp = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_share_of_temp),
        stratify=temp_df[target], random_state=seed, shuffle=True,
    )
    n = len(df)
    assert abs(len(train_df) / n - TRAIN_SIZE) < 0.02
    assert abs(len(val_df) / n - VAL_SIZE) < 0.02
    assert abs(len(test_df) / n - TEST_SIZE) < 0.02
    return train_df, val_df, test_df