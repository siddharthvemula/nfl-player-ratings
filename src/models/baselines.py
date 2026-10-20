"""Baselines: the simplest honest predictions. Every real model has to beat them."""

import polars as pl

from src.features.target import TARGET_COL, WEIGHT_COL


def league_average(train: pl.DataFrame) -> float:
    """Dropback-weighted mean target over the train rows. Pass train only, never val/test."""
    return float((train[TARGET_COL] * train[WEIGHT_COL]).sum() / train[WEIGHT_COL].sum())


def predict_constant(df: pl.DataFrame, value: float) -> pl.DataFrame:
    """Give every row the same prediction (Phase 3, Concept 1)."""
    return df.with_columns(pl.lit(value).alias("prediction"))


def season_to_date(games: pl.DataFrame, targets: pl.DataFrame, fallback: float) -> pl.DataFrame:
    """Predict each target row with the QB's EPA/dropback over his earlier games that season.

    games is the full qb_games() history (1-2 dropback games count too); targets is
    build_targets() output. A QB with no earlier game that season gets `fallback`
    (pass the train league average). Phase 3, Concept 2.
    """
    history = (
        # Step 1: each QB's games in time order within a season.
        games.sort("passer_id", "season", "week")
        # Step 2: running totals *before* this game: cumulative sum minus this game.
        # Subtracting the current game is what keeps week N's prediction to weeks < N.
        .with_columns(
            (pl.col("epa_sum").cum_sum().over("passer_id", "season") - pl.col("epa_sum")).alias("prior_epa"),
            (pl.col("dropbacks").cum_sum().over("passer_id", "season") - pl.col("dropbacks")).alias("prior_dropbacks"),
        )
        .select("game_id", "passer_id", "prior_epa", "prior_dropbacks")
    )
    return (
        targets.join(history, on=["game_id", "passer_id"], how="left")
        # Step 3: total EPA / total dropbacks so far, so bigger games count more.
        # No earlier games -> the fallback.
        .with_columns(
            pl.when(pl.col("prior_dropbacks") > 0)
            .then(pl.col("prior_epa") / pl.col("prior_dropbacks"))
            .otherwise(pl.lit(fallback))
            .alias("prediction")
        )
    )


def shrunk_to_date(games: pl.DataFrame, targets: pl.DataFrame, league_avg: float, k: float) -> pl.DataFrame:
    """Season-to-date average blended toward the league average (Phase 3, Concept 3).

    Every QB starts each season with k imaginary dropbacks at league_avg, then his
    real ones are added: (his EPA so far + k * league_avg) / (his dropbacks so far + k).
    Few real dropbacks -> close to league_avg; many -> close to his own average.
    k = 0 is the raw season-to-date average.
    """
    # Same running totals as season_to_date (weeks < N only, same season).
    preds = season_to_date(games, targets, fallback=league_avg).with_columns(
        pl.col("prior_epa").fill_null(0.0),
        pl.col("prior_dropbacks").fill_null(0),
    )
    if k == 0:
        return preds
    return preds.with_columns(
        ((pl.col("prior_epa") + k * league_avg) / (pl.col("prior_dropbacks") + k)).alias("prediction")
    )
