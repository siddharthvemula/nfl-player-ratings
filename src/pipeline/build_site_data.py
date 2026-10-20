"""Weekly site data: precompute every rating the website shows, as JSON in site_data/ (Phase 6).

    .venv/bin/python -m src.pipeline.build_site_data            # use cached play-by-play
    .venv/bin/python -m src.pipeline.build_site_data --refresh  # re-download the current season first

Every rating comes from the frozen final model (DECISIONS.md, "Final model spec"), fitted once per season on
earlier seasons only (fit_and_predict = one walk-forward fold), so each rating was made before its game.
The API only reads these files; no model runs per request.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy
import polars as pl

from src.data.load import FIRST_SEASON, PROJECT_ROOT, load_pbp
from src.features.qb_games import GAME_KEYS, qb_games
from src.features.target import TARGET_COL, WEIGHT_COL, build_targets, split_for_season
from src.models.final import FINAL_FEATURES, check_explanation_adds_up, explain, fit_and_predict
from src.pipeline.experiments import parse_experiments

SITE_DIR = PROJECT_ROOT / "site_data"
FIRST_RATED_SEASON = 2019  # the first walk-forward fold; earlier seasons have too little history to train on
RECENT_WEEKS = 2  # rankings list every QB with a dropback in the last 2 played weeks (no depth-chart data)

# Confidence = dropbacks of evidence (last season + this season so far). Cutoffs approved by Sid; each tier's
# real error is measured on walk-forward 2019-2024 (2025 is spent as a test set) and shown on the site.
CONFIDENCE_TIERS = (("Low", 0, 250), ("Med", 250, 600), ("High", 600, math.inf))
TIER_SEASONS = range(2019, 2025)

# The one-time 2025 test, frozen (DECISIONS.md, "2025 test result"). Never recomputed.
TEST_2025 = {
    "games": 624,
    "weighted_mae": {"league_average": 0.2491, "carry_over": 0.2431, "final": 0.2382},
    "unweighted_mae": {"league_average": 0.2858, "carry_over": 0.2779, "final": 0.2717},
    "win_rate_vs_carry_over": 0.993,
    "win_rate_vs_league_average": 0.998,
    "gain_vs_carry_over": {"gain": 0.0049, "low": 0.0009, "high": 0.0093},
    "low_history": {"games": 186, "carry_over": 0.2740, "final": 0.2559},
    "veterans": {"games": 438, "carry_over": 0.2322, "final": 0.2319},
}


def confidence(evidence: pl.Expr) -> pl.Expr:
    """Low / Med / High from dropbacks of evidence."""
    expr = pl.lit(None, dtype=pl.String)
    for name, lo, hi in reversed(CONFIDENCE_TIERS):
        expr = pl.when((evidence >= lo) & (evidence < hi)).then(pl.lit(name)).otherwise(expr)
    return expr


def next_game_placeholders(games: pl.DataFrame, schedule: pl.DataFrame, live_season: int) -> pl.DataFrame:
    """One zero-dropback games row per QB eligible for the rankings, at his team's next scheduled game.

    Feature code finds "everything before week N" by looking up a row at week N; an unplayed game has none.
    These rows add 0 to every total (dropbacks, EPA, sacks, CPOE), so no existing number changes, but they give
    each QB (and the opponent defense) a week-N row to stop at. No next game (season over) -> week = latest + 1,
    no opponent (the defense feature then falls back to the league average, i.e. neutral).
    """
    # Step 1: the last RECENT_WEEKS played weeks, in (season, week) order, and who dropped back in them.
    played = games.select("season", "week").unique().sort("season", "week")
    recent = played.tail(RECENT_WEEKS)
    latest = recent.row(-1, named=True)
    qbs = (
        games.join(recent, on=["season", "week"])
        .sort("season", "week")
        .group_by("passer_id")
        .agg(pl.col("passer").last(), pl.col("posteam").last())
    )

    # Step 2: each team's next scheduled game after the latest played week (in the live season).
    after = (latest["week"] if latest["season"] == live_season else 0)
    upcoming = schedule.filter(pl.col("season") == live_season, pl.col("week") > after, pl.col("result").is_null())
    sides = pl.concat([
        upcoming.select("week", "game_type", "game_id", pl.col("home_team").alias("posteam"), pl.col("away_team").alias("defteam"), pl.lit(1).alias("home")),
        upcoming.select("week", "game_type", "game_id", pl.col("away_team").alias("posteam"), pl.col("home_team").alias("defteam"), pl.lit(-1).alias("home")),
    ])
    next_game = sides.sort("week").group_by("posteam").first()

    # Step 3: the placeholder rows, in qb_games() shape.
    fallback_week = (latest["week"] + 1) if latest["season"] == live_season else 1
    return (
        qbs.join(next_game, on="posteam", how="left")
        .with_columns(
            pl.lit(live_season).cast(pl.Int32).alias("season"),
            pl.col("week").fill_null(fallback_week).cast(pl.Int32),
            pl.when(pl.col("game_type").is_null() | (pl.col("game_type") == "REG")).then(pl.lit("REG")).otherwise(pl.lit("POST")).alias("season_type"),
            pl.coalesce("game_id", pl.lit(f"{live_season}_next") + "_" + pl.col("posteam")).alias("game_id"),
            pl.col("home").fill_null(0).cast(pl.Int32),
        )
        .with_columns(
            pl.lit(0, dtype=pl.UInt32).alias("dropbacks"),
            pl.lit(0.0).alias("epa_sum"),
            pl.lit(None, dtype=pl.Float64).alias("epa_per_db"),
            pl.lit(0.0).alias("cpoe_sum"),
            pl.lit(0, dtype=pl.UInt32).alias("cpoe_att"),
            pl.lit(0.0).alias("sacks"),
        )
        .select(games.columns)
        .cast({c: games.schema[c] for c in games.columns})
    )


def placeholder_targets(placeholders: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Target rows for the placeholders: same shape as build_targets(), with no answer and no weight yet."""
    return placeholders.select(
        *GAME_KEYS,
        "passer_id",
        "passer",
        pl.lit(None, dtype=targets.schema[TARGET_COL]).alias(TARGET_COL),
        pl.lit(None, dtype=targets.schema[WEIGHT_COL]).alias(WEIGHT_COL),
        pl.col("season").map_elements(split_for_season, return_dtype=pl.String).alias("split"),
    )


def rate_season(games: pl.DataFrame, targets: pl.DataFrame, season: int) -> tuple[pl.DataFrame, dict]:
    """Every rating for one season (played games plus any placeholders), with the SHAP pieces and extras."""
    rows, fitted = fit_and_predict(games, targets, season)
    weight_def = float(dict(zip(FINAL_FEATURES, fitted["model"].coef_))["def_shrunk"])
    rows = explain(rows, fitted)
    check_explanation_adds_up(rows)
    rows = rows.with_columns(
        # Opponent-neutral rating (Sid's call): the same prediction with the defense set to league average.
        (pl.col("prediction") + weight_def * (fitted["league_avg"] - pl.col("def_shrunk"))).alias("neutral"),
        (pl.col("last_dropbacks") + pl.col("std_dropbacks")).alias("evidence"),
    ).with_columns(confidence(pl.col("evidence")).alias("confidence"))
    model = {
        "season": season,
        "trained_on": f"{FIRST_SEASON + 1}-{season - 1}",
        "league_avg": fitted["league_avg"],
        "carry_k": fitted["carry_k"],
        "defense_k": fitted["feature_k"][0],
        "sack_k": fitted["feature_k"][1],
        "intercept": float(fitted["model"].intercept_),
        "weights": {c: float(w) for c, w in zip(FINAL_FEATURES, fitted["model"].coef_)},
        "feature_means": fitted["feature_means"],
    }
    return rows, model


def to_records(rows: pl.DataFrame) -> list[dict]:
    """Flat JSON records, ranked by the opponent-neutral rating.

    Played games are ranked within their (season, week). Upcoming games are ranked all together: a QB whose
    team is on a bye has his next game a week later, but he belongs in this week's rankings.
    """
    rows = rows.with_columns(pl.col(TARGET_COL).is_null().alias("upcoming"))
    rows = rows.with_columns(
        pl.when(pl.col("upcoming")).then(pl.lit(0)).otherwise(pl.col("week")).alias("rank_group")
    ).with_columns(
        pl.col("neutral").rank("ordinal", descending=True).over("season", "rank_group").cast(pl.Int32).alias("rank")
    )
    out = rows.select(
        "season", "week", "season_type", "game_id", "passer_id", "passer",
        pl.col("posteam").alias("team"), pl.col("defteam").alias("opponent"), "upcoming", "rank",
        pl.col("neutral").round(4).alias("rating"),
        pl.col("prediction").round(4).alias("matchup_rating"),
        pl.col(TARGET_COL).round(4).alias("actual"),
        pl.col(WEIGHT_COL).alias("dropbacks"),
        "evidence", "confidence",
        pl.col("carry_over").round(4),
        pl.col("avg_adjustment").round(4),
        pl.struct([pl.col(f"{c}_contrib").round(4).alias(c) for c in FINAL_FEATURES]).alias("contributions"),
        pl.struct([pl.col(c).cast(pl.Float64).round(4) for c in FINAL_FEATURES]).alias("features"),
    ).sort("season", "week", "rank")
    return out.to_dicts()


def tier_errors(rows: pl.DataFrame) -> list[dict]:
    """Each confidence tier's real (out-of-sample) dropback-weighted MAE on walk-forward 2019-2024."""
    scored = rows.filter(pl.col("season").is_in(list(TIER_SEASONS)), pl.col(TARGET_COL).is_not_null())
    out = []
    for name, lo, hi in CONFIDENCE_TIERS:
        x = scored.filter(pl.col("confidence") == name)
        w = x[WEIGHT_COL].cast(pl.Float64)
        out.append({
            "tier": name, "min_evidence": lo, "max_evidence": None if math.isinf(hi) else hi, "games": x.height,
            "weighted_mae": round(float(((x[TARGET_COL] - x["prediction"]).abs() * w).sum() / w.sum()), 4),
        })
    return out


def build(refresh: bool = False, out_dir: Path = SITE_DIR) -> dict:
    # Step 1: data. The live season may not exist yet in nflverse (preseason), so it's loaded separately.
    live_season = nflreadpy.get_current_season()
    pbp = load_pbp(list(range(FIRST_SEASON, live_season)))
    try:
        pbp = pl.concat([pbp, load_pbp([live_season], refresh_current=refresh)], how="diagonal_relaxed")
    except Exception as e:  # no data for the live season yet
        print(f"No {live_season} play-by-play yet ({e}); rating from earlier seasons only")
    games = qb_games(pbp)
    targets = build_targets(games)
    schedule = nflreadpy.load_schedules([live_season])

    # Step 2: placeholders for each eligible QB's next game, in the live season.
    placeholders = next_game_placeholders(games, schedule, live_season)
    games_all = pl.concat([games, placeholders])
    targets_all = pl.concat([targets, placeholder_targets(placeholders, targets)])

    # Step 3: rate every season from 2019 on, each with a model trained only on earlier seasons.
    rated, models = [], []
    for season in range(FIRST_RATED_SEASON, live_season + 1):
        if targets_all.filter(pl.col("season") == season).is_empty():
            continue
        rows, model = rate_season(games_all, targets_all, season)
        rated.append(rows)
        models.append(model)
    rated = pl.concat(rated, how="diagonal_relaxed")

    # Step 4: write the JSON files the API serves.
    records = to_records(rated)
    # The current rankings = the upcoming games, labeled by the earliest upcoming week.
    upcoming = rated.filter(pl.col(TARGET_COL).is_null())
    current = {"season": live_season, "week": int(upcoming["week"].min()), "upcoming": True} if upcoming.height else {
        "season": records[-1]["season"], "week": records[-1]["week"], "upcoming": False}
    played_weeks = rated.filter(pl.col(TARGET_COL).is_not_null()).select("season", "week").unique().sort("season", "week")
    meta = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "live_season": live_season,
        "current": current,
        "played_weeks": played_weeks.to_dicts(),
        "features": FINAL_FEATURES,
        "models": models,
        "confidence_tiers": tier_errors(rated),
        "test_2025": TEST_2025,
    }
    experiments = parse_experiments((PROJECT_ROOT / "experiments.md").read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in [("meta.json", meta), ("ratings.json", records), ("experiments.json", experiments)]:
        (out_dir / name).write_text(json.dumps(payload, separators=(",", ":"), default=str))
    print(f"Wrote {len(records):,} ratings ({placeholders.height} upcoming, {current['season']} week {current['week']}) to {out_dir}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download the current season first")
    build(refresh=parser.parse_args().refresh)
