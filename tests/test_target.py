import polars as pl
import pytest

from src.features.qb_games import qb_games
from src.features.target import (
    LEAKY_COLUMNS,
    TARGET_COL,
    WEIGHT_COL,
    build_targets,
    check_no_leakage,
    split_for_season,
)


def _play(passer_id, qb_epa, season=2024, week=1, season_type="REG", epa=None):
    return {
        "season": season, "week": week, "season_type": season_type,
        "game_id": f"{season}_{week:02d}_BAL_KC", "posteam": "KC", "defteam": "BAL",
        "passer_id": passer_id, "passer": passer_id, "qb_dropback": 1,
        "qb_epa": qb_epa, "epa": qb_epa if epa is None else epa, "cpoe": 0.0,
        "sack": 0.0,
    }


def test_targets_keep_only_games_with_3_plus_dropbacks_but_history_keeps_all():
    pbp = pl.DataFrame([
        *[_play("mahomes", 0.3) for _ in range(3)],
        *[_play("backup", -1.0) for _ in range(2)],  # 2 dropbacks: history only
    ])
    games = qb_games(pbp)
    targets = build_targets(games)

    assert set(games["passer_id"]) == {"mahomes", "backup"}
    assert targets["passer_id"].to_list() == ["mahomes"]
    row = targets.row(0, named=True)
    assert row[WEIGHT_COL] == 3
    assert abs(row[TARGET_COL] - 0.3) < 1e-9


def test_target_uses_qb_epa_not_epa():
    # A receiver fumble: epa is a disaster, qb_epa keeps credit for the catch.
    pbp = pl.DataFrame([_play("watson", 0.5, epa=-10.8) for _ in range(3)])
    assert abs(build_targets(qb_games(pbp))[TARGET_COL][0] - 0.5) < 1e-9


def test_playoff_games_are_targets_in_week_order():
    pbp = pl.DataFrame([
        *[_play("mahomes", 0.6, week=19, season_type="POST") for _ in range(3)],
        *[_play("mahomes", 0.1, week=17) for _ in range(3)],
    ])
    targets = build_targets(qb_games(pbp))
    assert targets["week"].to_list() == [17, 19]
    assert targets["season_type"].to_list() == ["REG", "POST"]


def test_split_by_target_season():
    assert split_for_season(2016) == "train"
    assert split_for_season(2023) == "train"
    assert split_for_season(2024) == "val"
    assert split_for_season(2025) == "test"
    assert split_for_season(2026) == "live"
    with pytest.raises(ValueError):
        split_for_season(2015)


def test_same_game_columns_are_never_features():
    # The target game's dropbacks set the weight, so it must never be a model input.
    for col in [WEIGHT_COL, TARGET_COL, "dropbacks", "epa_per_db"]:
        with pytest.raises(ValueError, match="Leakage"):
            check_no_leakage(["prev_epa_per_db", col])
    check_no_leakage(["prev_epa_per_db", "season_to_date_epa"])  # past-only features are fine


def test_target_table_carries_no_same_game_stats_besides_target_and_weight():
    pbp = pl.DataFrame([_play("mahomes", 0.3) for _ in range(3)])
    cols = set(build_targets(qb_games(pbp)).columns)
    assert cols & LEAKY_COLUMNS == {TARGET_COL, WEIGHT_COL}
