"""Model features built from each QB's past games (Phase 4).

Every feature for a target row (QB, season, week N) uses only that QB's games
from weeks < N of the same season, or from earlier seasons. Nothing from the
game being predicted (see check_no_leakage in target.py).
"""

import polars as pl

from src.features.target import check_no_leakage

# Concept 1: the same ingredients carry-over uses, handed to the model raw.
BASE_FEATURES = ["std_avg", "std_dropbacks", "last_avg", "last_dropbacks"]


def base_features(games: pl.DataFrame, targets: pl.DataFrame, league_avg: float) -> pl.DataFrame:
    """Add this-season-so-far and last-season averages (and their dropback counts) to targets.

    std_avg   = EPA/dropback over weeks < N this season (league_avg if he hasn't played yet)
    last_avg  = EPA/dropback over all of last season    (league_avg if he didn't play)
    Pass league_avg computed on train only.
    """
    check_no_leakage(BASE_FEATURES)

    # Step 1: running totals of each QB's games *before* this one in the same season.
    # Sorting by week and shifting by one makes the running sum stop just short of week N.
    running = (
        games.sort("season", "week")
        .with_columns(
            pl.col("epa_sum").cum_sum().shift(1).over("passer_id", "season").fill_null(0.0).alias("std_epa"),
            pl.col("dropbacks").cum_sum().shift(1).over("passer_id", "season").fill_null(0).alias("std_dropbacks"),
        )
        .select("passer_id", "season", "week", "std_epa", "std_dropbacks")
    )

    # Step 2: each QB's full-season totals, relabeled as "last season" for the following year.
    last = (
        games.group_by("passer_id", "season")
        .agg(pl.col("epa_sum").sum().alias("last_epa"), pl.col("dropbacks").sum().alias("last_dropbacks"))
        .with_columns(pl.col("season") + 1)
    )

    # Step 3: attach both to the target rows and turn totals into per-dropback averages.
    # A model can't read a blank, so "no data" becomes the league average.
    return (
        targets.join(running, on=["passer_id", "season", "week"], how="left")
        .join(last, on=["passer_id", "season"], how="left")
        .with_columns(
            pl.col("std_epa").fill_null(0.0),
            pl.col("std_dropbacks").fill_null(0),
            pl.col("last_epa").fill_null(0.0),
            pl.col("last_dropbacks").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("std_dropbacks") > 0)
            .then(pl.col("std_epa") / pl.col("std_dropbacks"))
            .otherwise(pl.lit(league_avg))
            .alias("std_avg"),
            pl.when(pl.col("last_dropbacks") > 0)
            .then(pl.col("last_epa") / pl.col("last_dropbacks"))
            .otherwise(pl.lit(league_avg))
            .alias("last_avg"),
        )
        .drop("std_epa", "last_epa")
    )
