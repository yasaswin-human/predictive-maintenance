"""Exploratory Data Analysis helpers producing files under models/metrics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import CATEGORICAL_FEATURES, METRICS_DIR, NUMERICAL_FEATURES, TARGET_COL

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure -> %s", path)


def run_eda(df: pd.DataFrame, out_dir: Path = METRICS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "missing_per_col": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "target_counts": {str(k): int(v) for k, v in df[TARGET_COL].value_counts().items()},
        "target_ratio": float(df[TARGET_COL].mean()),
        "describe_numeric": df[NUMERICAL_FEATURES].describe().to_dict(),
    }
    if CATEGORICAL_FEATURES[0] in df.columns:
        stats["type_counts"] = {str(k): int(v) for k, v in df[CATEGORICAL_FEATURES[0]].value_counts().items()}
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(6, 4))
    df[TARGET_COL].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4CAF50", "#E53935"])
    ax.set_title("Machine failure distribution"); ax.set_xlabel("Machine failure (0=healthy, 1=failure)"); ax.set_ylabel("Count")
    _save(fig, out_dir / "eda_target_distribution.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flatten(), NUMERICAL_FEATURES):
        df[col].hist(ax=ax, bins=40, color="#1E88E5", edgecolor="black"); ax.set_title(col)
    for ax in axes.flatten()[len(NUMERICAL_FEATURES):]:
        ax.axis("off")
    _save(fig, out_dir / "eda_numeric_histograms.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[NUMERICAL_FEATURES + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax); ax.set_title("Correlation heatmap")
    _save(fig, out_dir / "eda_correlation_heatmap.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flatten(), NUMERICAL_FEATURES):
        sns.boxplot(x=TARGET_COL, y=col, data=df, ax=ax, palette=["#4CAF50", "#E53935"])
        ax.set_title(f"{col} vs failure")
    for ax in axes.flatten()[len(NUMERICAL_FEATURES):]:
        ax.axis("off")
    _save(fig, out_dir / "eda_boxplots_vs_target.png")

    cat = CATEGORICAL_FEATURES[0]
    if cat in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=cat, data=df, ax=ax, palette="viridis"); ax.set_title("Machine Type distribution")
        _save(fig, out_dir / "eda_type_distribution.png")

    logger.info("EDA complete. Artifacts in %s", out_dir)
    return stats