"""Collapse play-by-play into one row per passer per game.

This is the table the target is built from: each row says how many times a
passer dropped back in a game and how much EPA he produced per dropback.
"""

import polars as pl

# Columns that identify one passer in one game. season/week/season_type/defteam
# are constant within a game, so grouping by them too just carries them along.
GAME_KEYS = ["season", "week", "season_type", "game_id", "posteam", "defteam"]


def qb_games(pbp: pl.DataFrame, epa_col: str = "qb_epa") -> pl.DataFrame:
    """Return one row per (game, team, passer) with dropbacks and EPA per dropback.

    epa_col is the per-play EPA column to average. The default, qb_epa, doesn't
    charge the QB for a receiver's fumble after the catch (Concept 5). The
    notebook passes "epa" to reproduce the Phase 2 exploration.
    """
    # Step 1: keep only dropbacks (pass attempts, sacks, scrambles).
    dropbacks = pbp.filter(pl.col("qb_dropback") == 1)

    # Some dropbacks have no passer_id (nflverse hasn't tagged the passer yet). Grouped, they'd become
    # a fake "None" QB who then breaks sorting downstream, so they never become QB rows. Say how many per season.
    no_passer = dropbacks.filter(pl.col("passer_id").is_null())
    if no_passer.height:
        per_season = no_passer.group_by("season").len().sort("season")
        counts = ", ".join(f"{s}: {n}" for s, n in per_season.iter_rows())
        print(f"qb_games: dropped {no_passer.height} dropbacks with null passer_id ({counts})")
    # They still happened, so the defense that faced them keeps the credit (DECISIONS.md). Totals per
    # team-game ride along on one QB row of that game, in def_extra_* columns that only opponent.py and league_sack_rate read.
    extra = no_passer.group_by(GAME_KEYS).agg(
        pl.col(epa_col).sum().alias("def_extra_epa"),
        pl.len().cast(pl.UInt32).alias("def_extra_dropbacks"),
        pl.col("sack").fill_null(0).sum().alias("def_extra_sacks"),
    )

    qb_rows = (
        dropbacks.filter(pl.col("passer_id").is_not_null())
        # Step 2: sort the "receipts" into piles, one pile per passer per game.
        # passer_id (not passer_player_id) because only passer_id is filled in
        # on scrambles; see Concept 4 in notebooks/01_explore.ipynb.
        .group_by([*GAME_KEYS, "passer_id"])
        # Step 3: write one summary line per pile.
        .agg(
            pl.col("passer").first(),
            pl.len().alias("dropbacks"),
            pl.col(epa_col).sum().alias("epa_sum"),
            # mean() skips null EPA, so this is epa_sum / (dropbacks with an EPA).
            pl.col(epa_col).mean().alias("epa_per_db"),
            # CPOE (Phase 4, Concept 5): totals over throws that have one. Sacks, scrambles and
            # some passes are blank; sum() skips them and cpoe_att counts only the filled ones,
            # so blanks are never treated as "exactly average accuracy" (0).
            pl.col("cpoe").sum().alias("cpoe_sum"),
            pl.col("cpoe").is_not_null().sum().alias("cpoe_att"),
            # Sacks (Phase 4, Concept 6): sacks / dropbacks is the QB's sack rate.
            pl.col("sack").fill_null(0).sum().alias("sacks"),
            # Home field (Phase 4, Concept 7): +1 home, -1 away, 0 neutral site (London, Super Bowl).
            # Known from the schedule before kickoff, so it's safe as a feature.
            pl.when(pl.col("location") == "Neutral").then(0)
            .when(pl.col("posteam") == pl.col("home_team")).then(1)
            .otherwise(-1).first().alias("home"),
        )
        .sort(["season", "week", "game_id", "posteam", "dropbacks"], descending=[False, False, False, False, True])
    )
    # Hand the extra totals to the first QB row of each team-game (0 on the others, so nothing double counts).
    return (
        qb_rows.join(extra, on=GAME_KEYS, how="left")
        .with_columns(
            pl.when(pl.int_range(pl.len()).over(GAME_KEYS) == 0).then(pl.col("def_extra_epa")).otherwise(None).fill_null(0.0).alias("def_extra_epa"),
            pl.when(pl.int_range(pl.len()).over(GAME_KEYS) == 0).then(pl.col("def_extra_dropbacks")).otherwise(None).fill_null(0).cast(pl.UInt32).alias("def_extra_dropbacks"),
            pl.when(pl.int_range(pl.len()).over(GAME_KEYS) == 0).then(pl.col("def_extra_sacks")).otherwise(None).fill_null(0.0).alias("def_extra_sacks"),
        )
    )
