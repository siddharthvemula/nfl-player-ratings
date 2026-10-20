"""The final model (frozen in DECISIONS.md, "Final model spec") and per-QB explanations of its ratings.

Final model = carry-over + a linear adjustment on six features (Phase 4, C6), fitted with the same
steps as one walk-forward fold: everything fitted from data uses only seasons before the one predicted.
"""

import itertools

import polars as pl

from src.features.history import BASE_FEATURES, SACK_FEATURES, base_features, sack_features
from src.features.opponent import DEFENSE_FEATURES, defense_features
from src.models.walk_forward import BASE_COL, Experiment, run_fold

FINAL_FEATURES = BASE_FEATURES + DEFENSE_FEATURES + SACK_FEATURES
DEF_K_GRID = (50, 100, 200, 300, 500, 800, 1200, 2000)
SACK_K_GRID = (50, 100, 200, 300, 500, 800, 1200)


def build_final(games: pl.DataFrame, targets: pl.DataFrame, league_avg: float, k: tuple[int, int]) -> pl.DataFrame:
    """The six features; k = (defense K, sack K)."""
    return sack_features(games, defense_features(games, base_features(games, targets, league_avg), league_avg, k[0]), k[1])


FINAL = Experiment("final: C6 linear", FINAL_FEATURES, build_final, tuple(itertools.product(DEF_K_GRID, SACK_K_GRID)))


def fit_and_predict(games: pl.DataFrame, targets: pl.DataFrame, season: int) -> tuple[pl.DataFrame, dict]:
    """Fit on every season before `season`, predict `season`. Returns (rows with features + prediction, what was fitted).

    Exactly one walk-forward fold, so the final model is the same recipe that was scored in Phases 4-5.
    """
    return run_fold(games, targets, season, FINAL)


def explain(rows: pl.DataFrame, fitted: dict, feature_cols: list[str] = FINAL_FEATURES) -> pl.DataFrame:
    """Split each prediction into carry-over + the average adjustment + one piece per feature.

    For a linear adjustment, each feature's SHAP value is exact and simple:
        contribution_j = weight_j * (his value_j - training mean_j)
    "How much higher or lower is his rating because this feature differs from a typical training row?"
    The pieces add up exactly:
        prediction = carry_over + avg_adjustment + sum of contributions
        avg_adjustment = intercept + sum(weight_j * mean_j)   (the adjustment for a perfectly average row)
    rows needs carry_over and the feature columns (fit_and_predict() output); fitted is its second return value.
    Adds avg_adjustment and one "<feature>_contrib" column per feature.
    """
    weights = dict(zip(feature_cols, fitted["model"].coef_))
    means = fitted["feature_means"]
    avg_adjustment = float(fitted["model"].intercept_) + sum(weights[c] * means[c] for c in feature_cols)
    return rows.with_columns(
        pl.lit(avg_adjustment).alias("avg_adjustment"),
        *[(weights[c] * (pl.col(c) - means[c])).alias(f"{c}_contrib") for c in feature_cols],
    )


def check_explanation_adds_up(explained: pl.DataFrame, feature_cols: list[str] = FINAL_FEATURES, tol: float = 1e-9) -> None:
    """Raise if carry_over + avg_adjustment + contributions doesn't equal the prediction."""
    total = pl.col(BASE_COL) + pl.col("avg_adjustment") + pl.sum_horizontal(f"{c}_contrib" for c in feature_cols)
    gap = explained.select((total - pl.col("prediction")).abs().max()).item()
    if gap > tol:
        raise ValueError(f"explanation is off by {gap}")
