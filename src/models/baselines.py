"""Baselines: the simplest honest predictions. Every real model has to beat them."""

import polars as pl

from src.features.target import TARGET_COL, WEIGHT_COL


def league_average(train: pl.DataFrame) -> float:
    """Dropback-weighted mean target over the train rows. Pass train only, never val/test."""
    return float((train[TARGET_COL] * train[WEIGHT_COL]).sum() / train[WEIGHT_COL].sum())


def predict_constant(df: pl.DataFrame, value: float) -> pl.DataFrame:
    """Give every row the same prediction (Phase 3, Concept 1)."""
    return df.with_columns(pl.lit(value).alias("prediction"))
