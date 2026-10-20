"""Scoring: how far predictions are from the target (see DECISIONS.md, "Locked target").

The official score is dropback-weighted MAE on 2024 val; unweighted MAE is logged
alongside it to show whether a change helps across all games or mostly on
high-dropback ones.
"""

import polars as pl

from src.features.target import TARGET_COL, WEIGHT_COL


def mae(df: pl.DataFrame, pred_col: str = "prediction") -> dict[str, float]:
    """Return {"weighted": ..., "unweighted": ...} mean absolute error of pred_col vs the target."""
    # Step 1: how wrong each game's prediction was, ignoring direction.
    error = (df[TARGET_COL] - df[pred_col]).abs()
    weight = df[WEIGHT_COL].cast(pl.Float64)
    return {
        # Step 2: a 45-dropback game counts 45x as much as a 1-dropback game would.
        "weighted": float((error * weight).sum() / weight.sum()),
        # Step 3: every game counts the same.
        "unweighted": float(error.mean()),
    }
