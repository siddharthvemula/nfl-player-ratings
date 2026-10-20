"""The locked prediction target (Phase 2; see DECISIONS.md, "Locked target").

Each row is one QB-game we try to predict: the QB's qb_epa per dropback in that
game. Features for a row may only use that QB's *earlier* games, and they are
built from the full qb_games() table, including 1-2 dropback games that are
too small to be targets themselves.

Usage:
    games = qb_games(pbp)          # history: every QB-game, any size
    targets = build_targets(games) # what we predict and score
"""

import polars as pl

from src.features.qb_games import GAME_KEYS

MIN_TARGET_DROPBACKS = 3  # Concept 2: smaller games are mostly trick plays and cameos.

TARGET_COL = "target"  # qb_epa per dropback in the game being predicted
WEIGHT_COL = "weight"  # dropbacks in the game being predicted

# Columns describing the game being predicted. They are only known after it is
# played, so using any of them as a feature would leak the answer. The weight is
# fine for training and scoring, never as a model input.
LEAKY_COLUMNS = {TARGET_COL, WEIGHT_COL, "dropbacks", "epa_sum", "epa_per_db"}

# Time-based splits by the target game's season. Later seasons are "live":
# the site rates them, but they're never used to score the model.
SPLITS = {"train": range(2016, 2024), "val": range(2024, 2025), "test": range(2025, 2026)}


def split_for_season(season: int) -> str:
    """Return "train", "val", "test", or "live" for a target game's season."""
    for name, seasons in SPLITS.items():
        if season in seasons:
            return name
    if season > max(SPLITS["test"]):
        return "live"
    raise ValueError(f"season {season} is before our data starts")


def build_targets(games: pl.DataFrame, min_dropbacks: int = MIN_TARGET_DROPBACKS) -> pl.DataFrame:
    """Turn the qb_games() table into one target row per QB-game with min_dropbacks+."""
    return (
        # Step 1: only games big enough to be a fair test of the QB.
        games.filter(pl.col("dropbacks") >= min_dropbacks)
        # Step 2: name the answer and its weight. Same-game stats are dropped so
        # they can't accidentally ride along into a feature table.
        .select(
            *GAME_KEYS,
            "passer_id",
            "passer",
            pl.col("epa_per_db").alias(TARGET_COL),
            pl.col("dropbacks").alias(WEIGHT_COL),
        )
        # Step 3: tag each row with its split.
        .with_columns(
            pl.col("season").map_elements(split_for_season, return_dtype=pl.String).alias("split")
        )
        # Step 4: time order. Playoff weeks continue the week numbers (Concept 6).
        .sort(["season", "week", "game_id", "passer_id"])
    )


def check_no_leakage(feature_cols: list[str]) -> None:
    """Raise if any feature is a same-game column (the answer, its weight, or its stats)."""
    leaky = sorted(set(feature_cols) & LEAKY_COLUMNS)
    if leaky:
        raise ValueError(f"Leakage: {leaky} describe the game being predicted and can't be features")
