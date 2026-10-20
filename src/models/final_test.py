"""The one-time 2025 test (frozen in DECISIONS.md, "Final model spec"). Run once; never tune after seeing it.

    .venv/bin/python -m src.models.final_test

Trains on 2016-2024 (2015 = history only) and scores every 2025 QB-game, regular season and playoffs:
the final model vs league average vs carry-over, weighted + unweighted MAE, bootstrap by QB.
"""

import polars as pl

from src.data.load import load_pbp
from src.features.history import base_features
from src.features.qb_games import qb_games
from src.features.target import FIRST_TARGET_SEASON, build_targets
from src.models.evaluate import bootstrap_compare, mae
from src.models.final import fit_and_predict
from src.models.walk_forward import KEY, Experiment, run_fold

TEST_SEASON = 2025


def run_final_test() -> dict:
    # Step 1: data through 2025 only (2026 is live, never scored).
    games = qb_games(load_pbp(list(range(2015, TEST_SEASON + 1))))
    targets = build_targets(games).filter(pl.col("season").is_between(FIRST_TARGET_SEASON, TEST_SEASON))

    # Step 2: three models, each fitted on 2016-2024 only, exactly like one walk-forward fold.
    final, fitted = fit_and_predict(games, targets, TEST_SEASON)
    league, _ = run_fold(games, targets, TEST_SEASON, Experiment("league average", league_average_only=True))
    carry, _ = run_fold(games, targets, TEST_SEASON, Experiment("carry-over"))
    scored = final.select(*KEY, "target", "weight", pl.col("prediction").alias("final")).join(
        league.select(*KEY, pl.col("prediction").alias("league_avg")), on=KEY
    ).join(carry.select(*KEY, pl.col("prediction").alias("carry_over")), on=KEY)

    # Step 3: scores and bootstraps (QBs resampled, 5000 fake seasons, seed 0).
    result = {
        "games": scored.height,
        "fitted": {k: v for k, v in fitted.items() if k not in ("model", "feature_means")},
        "mae": {m: mae(scored, m) for m in ("league_avg", "carry_over", "final")},
        "final_vs_carry_over": bootstrap_compare(scored, "carry_over", "final"),
        "final_vs_league_avg": bootstrap_compare(scored, "league_avg", "final"),
        "carry_over_vs_league_avg": bootstrap_compare(scored, "league_avg", "carry_over"),
    }

    # Step 4 (pre-registered secondary): low-history QBs (last season <= 212 dropbacks) vs the rest.
    hist = base_features(games, targets, fitted["league_avg"]).select(*KEY, "last_dropbacks")
    scored = scored.join(hist, on=KEY)
    for label, cond in [("low_history", pl.col("last_dropbacks") <= 212), ("veterans", pl.col("last_dropbacks") > 212)]:
        x = scored.filter(cond)
        result[label] = {"games": x.height, "carry_over": mae(x, "carry_over")["weighted"], "final": mae(x, "final")["weighted"],
                         "win_rate": bootstrap_compare(x, "carry_over", "final")["win_rate"]}
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(run_final_test(), indent=2, default=str))
