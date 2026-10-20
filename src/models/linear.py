"""Linear models: learn one weight per feature from train (Phase 4)."""

import polars as pl
from sklearn.linear_model import LinearRegression

from src.features.target import TARGET_COL, WEIGHT_COL, check_no_leakage


def fit_linear(train: pl.DataFrame, feature_cols: list[str]) -> LinearRegression:
    """Fit prediction = w0 + w1*f1 + w2*f2 + ... on train rows (Phase 4, Concept 1).

    Rows are weighted by dropbacks, matching the official weighted MAE: a 45-dropback
    game pulls on the weights 45x as hard as a 1-dropback game would.
    """
    check_no_leakage(feature_cols)
    model = LinearRegression()
    # Step 1: the features become a plain number grid (rows = QB-games, columns = features).
    X = train.select(feature_cols).to_numpy()
    # Step 2: find the weights that make squared error, weighted by dropbacks, as small as possible.
    model.fit(X, train[TARGET_COL].to_numpy(), sample_weight=train[WEIGHT_COL].to_numpy())
    return model


def predict_linear(model, df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Add a "prediction" column from a fitted model."""
    return df.with_columns(pl.Series("prediction", model.predict(df.select(feature_cols).to_numpy())))
