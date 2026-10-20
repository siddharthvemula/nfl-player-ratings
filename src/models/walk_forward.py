"""Walk-forward cross-validation with residual models (evaluation protocol v2; see DECISIONS.md).

For each fold season S in 2019-2024:
  1. Everything fitted from data comes from seasons before S only: the league average,
     carry-over's K, any feature K, and the model weights.
  2. Prediction = carry_over + adjustment, where the adjustment is a linear model fit on
     (target - carry_over). Carry-over's weight is fixed at 1.
  3. Score season S.
All folds are pooled for the official score. 2025 (the Phase 5 test) is dropped first.
"""

from dataclasses import dataclass, field
from typing import Callable

import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.features.target import FIRST_TARGET_SEASON, TARGET_COL, WEIGHT_COL, check_no_leakage
from src.models.baselines import carry_over, league_average
from src.models.evaluate import bootstrap_compare, mae

FOLD_SEASONS = list(range(2019, 2025))
LAST_SCORED_SEASON = 2024  # 2025 is never touched until the final test
CARRY_K_GRID = [100, 150, 200, 250, 300, 400, 500, 700, 1000]
KEY = ["passer_id", "season", "week"]
BASE_COL = "carry_over"


@dataclass(frozen=True)
class Experiment:
    """One row of the scoreboard: which features to add on top of carry-over, and how to build them.

    build(games, targets, league_avg, k) must return targets plus feature_cols, using weeks < N only.
    k_grid: candidate K values, picked inside each fold on its earlier seasons (None = no K).
    league_average_only: the naive baseline; every row gets the fold's league average (no features).
    make_model(param) -> an unfitted model for the adjustment (None = plain LinearRegression).
    param_grid: candidate settings for make_model (e.g. Ridge's alpha), picked inside each fold by
      training on seasons < S-1 and scoring season S-1 (in-sample fit would always pick the most flexible one).
    """

    name: str
    feature_cols: list[str] = field(default_factory=list)
    build: Callable | None = None
    k_grid: tuple = (None,)
    league_average_only: bool = False
    make_model: Callable | None = None
    param_grid: tuple = (None,)


def fit_residual(train: pl.DataFrame, feature_cols: list[str], model=None):
    """Fit the adjustment: (target - carry_over) = w0 + w1*f1 + ..., weighted by dropbacks.

    model: any unfitted sklearn-style regressor (default LinearRegression). For a Pipeline
    (e.g. scaler + Ridge), the dropback weights go to its last step.
    """
    check_no_leakage(feature_cols)
    model = LinearRegression() if model is None else model
    residual = (train[TARGET_COL] - train[BASE_COL]).to_numpy()
    weights = train[WEIGHT_COL].to_numpy()
    if isinstance(model, Pipeline):
        model.fit(train.select(feature_cols).to_numpy(), residual, **{f"{model.steps[-1][0]}__sample_weight": weights})
    else:
        model.fit(train.select(feature_cols).to_numpy(), residual, sample_weight=weights)
    return model


def pick_param(feats: pl.DataFrame, season: int, exp: Experiment):
    """The make_model setting that scores best on season - 1 when trained on the seasons before it."""
    inner_train = feats.filter(pl.col("season") < season - 1)
    inner_val = feats.filter(pl.col("season") == season - 1)
    scores = {}
    for param in exp.param_grid:
        model = fit_residual(inner_train, exp.feature_cols, exp.make_model(param))
        scores[param] = mae(predict_residual(model, inner_val, exp.feature_cols))["weighted"]
    return min(scores, key=scores.get)


def predict_residual(model, df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """prediction = carry_over (weight fixed at 1) + the learned adjustment."""
    adjustment = model.predict(df.select(feature_cols).to_numpy())
    return df.with_columns((pl.col(BASE_COL) + pl.Series(adjustment)).alias("prediction"))


def pick_carry_k(games: pl.DataFrame, earlier: pl.DataFrame, league_avg: float) -> int:
    """Carry-over K with the lowest weighted MAE on the earlier seasons."""
    scores = {k: mae(carry_over(games, earlier, league_avg, k))["weighted"] for k in CARRY_K_GRID}
    return min(scores, key=scores.get)


def run_fold(games: pl.DataFrame, targets: pl.DataFrame, season: int, exp: Experiment) -> tuple[pl.DataFrame, dict]:
    """Train on seasons < season, predict season. Returns (scored rows, what was picked)."""
    # Step 1: only seasons up to the one being scored exist from here on.
    rows = targets.filter(pl.col("season") <= season)
    earlier = rows.filter(pl.col("season") < season)

    # Step 2: numbers fitted from data, earlier seasons only.
    league_avg = league_average(earlier)
    carry_k = pick_carry_k(games, earlier, league_avg)
    rows = carry_over(games, rows, league_avg, carry_k).rename({"prediction": BASE_COL}).select(*targets.columns, BASE_COL)
    picked = {"league_avg": league_avg, "carry_k": carry_k}

    # Naive baseline: one number for everyone, the league average of the earlier seasons.
    if exp.league_average_only:
        return rows.filter(pl.col("season") == season).with_columns(pl.lit(league_avg).alias("prediction")), picked

    # No features: the experiment *is* carry-over.
    if not exp.feature_cols:
        return rows.filter(pl.col("season") == season).with_columns(pl.col(BASE_COL).alias("prediction")), picked

    # Step 3: pick the feature's K by in-sample fit on the earlier seasons, then fit the adjustment there.
    best = None
    for k in exp.k_grid:
        feats = exp.build(games, rows, league_avg, k)
        train = feats.filter(pl.col("season") < season)
        model = fit_residual(train, exp.feature_cols)
        score = mae(predict_residual(model, train, exp.feature_cols))["weighted"]
        if best is None or score < best[0]:
            best = (score, k, model, feats)
    _, k, model, feats = best
    picked |= {"feature_k": k, "intercept": float(model.intercept_), "weights": dict(zip(exp.feature_cols, model.coef_.round(4)))}

    # Step 3b (optional): swap in a different model on the same features, its setting picked on season - 1.
    if exp.make_model is not None:
        param = pick_param(feats, season, exp)
        model = fit_residual(feats.filter(pl.col("season") < season), exp.feature_cols, exp.make_model(param))
        picked |= {"param": param, "model": model}

    # Step 4: predict the held-out season.
    return predict_residual(model, feats.filter(pl.col("season") == season), exp.feature_cols), picked


def walk_forward(games: pl.DataFrame, targets: pl.DataFrame, exp: Experiment) -> tuple[pl.DataFrame, list[dict]]:
    """Run every fold and pool the scored rows. 2025 is dropped before anything else.

    History-only seasons (2015) stay in games for features but never appear as target rows,
    so they can't enter any fold's training rows or its league average.
    """
    games = games.filter(pl.col("season") <= LAST_SCORED_SEASON)
    targets = targets.filter(pl.col("season").is_between(FIRST_TARGET_SEASON, LAST_SCORED_SEASON))
    pooled, picks = [], []
    for season in FOLD_SEASONS:
        scored, picked = run_fold(games, targets, season, exp)
        pooled.append(scored.select(*KEY, TARGET_COL, WEIGHT_COL, BASE_COL, "prediction"))
        picks.append({"season": season, **picked})
    return pl.concat(pooled), picks


def summarize(pooled: pl.DataFrame, best_col: str = BASE_COL) -> dict:
    """Pooled MAE, per-season gains vs the current best, bootstrap, and the keep/drop call."""
    per_season = {}
    for season in FOLD_SEASONS:
        s = pooled.filter(pl.col("season") == season)
        per_season[season] = mae(s, best_col)["weighted"] - mae(s)["weighted"]  # positive = new is better
    wins = sum(g > 0 for g in per_season.values())
    boot = bootstrap_compare(pooled, best_col, "prediction")
    return {
        "mae": mae(pooled)["weighted"],
        "mae_unweighted": mae(pooled)["unweighted"],
        "best_mae": mae(pooled, best_col)["weighted"],
        "per_season_gain": per_season,
        "seasons_won": wins,
        "win_rate": boot["win_rate"],
        "gain_low": boot["low"],
        "gain_high": boot["high"],
        "keep": boot["win_rate"] >= 0.80 and wins > len(FOLD_SEASONS) / 2,
    }


def compare(new: pl.DataFrame, best: pl.DataFrame) -> dict:
    """summarize() the pooled predictions of `new` against the pooled predictions of the current `best`."""
    joined = new.join(best.select(*KEY, pl.col("prediction").alias("current_best")), on=KEY)
    return summarize(joined, best_col="current_best")
