"""Collapse play-by-play into one row per passer per game.

This is the table the target is built from: each row says how many times a
passer dropped back in a game and how much EPA he produced per dropback.
"""

import polars as pl

# Columns that identify one passer in one game. season/week/season_type/defteam
# are constant within a game, so grouping by them too just carries them along.
GAME_KEYS = ["season", "week", "season_type", "game_id", "posteam", "defteam"]


def qb_games(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (game, team, passer) with dropbacks and EPA per dropback."""
    return (
        # Step 1: keep only dropbacks (pass attempts, sacks, scrambles).
        pbp.filter(pl.col("qb_dropback") == 1)
        # Step 2: sort the "receipts" into piles, one pile per passer per game.
        # passer_id (not passer_player_id) because only passer_id is filled in
        # on scrambles; see Concept 4 in notebooks/01_explore.ipynb.
        .group_by([*GAME_KEYS, "passer_id"])
        # Step 3: write one summary line per pile.
        .agg(
            pl.col("passer").first(),
            pl.len().alias("dropbacks"),
            pl.col("epa").sum().alias("epa_sum"),
            # mean() skips null EPA, so this is epa_sum / (dropbacks with an EPA).
            pl.col("epa").mean().alias("epa_per_db"),
        )
        .sort(["season", "week", "game_id", "posteam", "dropbacks"], descending=[False, False, False, False, True])
    )
