"""Features about the opponent in the game being predicted (Phase 4, Concept 4).

Who the opponent is comes from the schedule, so it's known before kickoff. How good
the opponent is must come only from its games in weeks < N of the same season.
"""

import polars as pl

from src.features.target import check_no_leakage

DEFENSE_FEATURES = ["def_shrunk"]
DEFENSE_K = 300  # imaginary league-average dropbacks; default only, pick_defense_k() picks it on train


def defense_features(games: pl.DataFrame, targets: pl.DataFrame, league_avg: float, k: float = DEFENSE_K) -> pl.DataFrame:
    """Add the opponent defense's EPA/dropback allowed so far this season, shrunk toward league_avg.

    def_shrunk = (EPA it allowed in weeks < N + k * league_avg) / (dropbacks it faced in weeks < N + k)
    Negative = stingy defense, positive = leaky one. Week 1 (nothing played yet) = league_avg.
    Pass league_avg computed on train only.
    """
    check_no_leakage(DEFENSE_FEATURES)

    # Step 1: one row per defense per game, adding up every QB it faced that week.
    per_game = games.group_by("defteam", "season", "week").agg(
        pl.col("epa_sum").sum().alias("def_epa"), pl.col("dropbacks").sum().alias("def_dropbacks")
    )

    # Step 2: running totals that stop just before week N (shift by one game, like base_features).
    # The defense's week N game holds the very game we're predicting, so it must not count.
    running = (
        per_game.sort("season", "week")
        .with_columns(
            pl.col("def_epa").cum_sum().shift(1).over("defteam", "season").fill_null(0.0).alias("def_epa_before"),
            pl.col("def_dropbacks").cum_sum().shift(1).over("defteam", "season").fill_null(0).alias("def_dropbacks_before"),
        )
        .select("defteam", "season", "week", "def_epa_before", "def_dropbacks_before")
    )

    # Step 3: attach to each target row by its opponent, then shrink by how much the defense has played.
    return (
        targets.join(running, on=["defteam", "season", "week"], how="left")
        .with_columns(pl.col("def_epa_before").fill_null(0.0), pl.col("def_dropbacks_before").fill_null(0))
        .with_columns(
            ((pl.col("def_epa_before") + k * league_avg) / (pl.col("def_dropbacks_before") + k)).alias("def_shrunk")
        )
        .drop("def_epa_before")
    )


# Concept 7: where the game is played. Like the opponent, it's on the schedule before kickoff.
HOME_FEATURES = ["home"]


def home_features(games: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Add home = +1 (home), -1 (away), 0 (neutral site) for the game being predicted."""
    check_no_leakage(HOME_FEATURES)
    return targets.join(games.select("passer_id", "season", "week", "home"), on=["passer_id", "season", "week"], how="left")
