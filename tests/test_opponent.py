import polars as pl

from tests.test_evaluate import _game


def _vs(passer_id, week, defteam, dropbacks, epa_sum):
    return {**_game(passer_id, week, dropbacks, epa_sum), "defteam": defteam}


def test_defense_uses_only_earlier_weeks_and_shrinks():
    from src.features.opponent import defense_features
    from src.features.target import build_targets

    games = pl.DataFrame([
        _vs("a", 1, "DEN", 40, -8.0),  # DEN allows -0.2 in week 1
        _vs("b", 1, "DEN", 10, 1.0),   # two QBs vs DEN the same week both count
        _vs("c", 2, "DEN", 30, 30.0),  # the game being predicted: must not count for itself
        _vs("d", 2, "CAR", 30, 0.0),
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    f = defense_features(games, build_targets(games), league_avg=0.05, k=100)
    get = lambda qb: f.filter(pl.col("passer_id") == qb)["def_shrunk"][0]
    assert abs(get("a") - 0.05) < 1e-9                              # week 1: nothing yet, league average
    assert abs(get("c") - (-8.0 + 1.0 + 100 * 0.05) / (50 + 100)) < 1e-9  # week 1 only, shrunk
    assert abs(get("d") - 0.05) < 1e-9                              # CAR had no earlier games


def test_home_flag_attaches_to_the_predicted_game():
    from src.features.opponent import home_features
    from src.features.target import build_targets

    games = pl.DataFrame([{**_game("a", 1, 30, 3.0), "home": 1}, {**_game("b", 1, 30, 3.0), "home": -1}])
    games = games.with_columns(pl.col("dropbacks").cast(pl.UInt32))
    f = home_features(games, build_targets(games))
    assert f.sort("passer_id")["home"].to_list() == [1, -1]
