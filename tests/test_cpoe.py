import polars as pl

from tests.test_evaluate import _game


def _g(passer_id, week, cpoe_sum, att, season=2024):
    return {**_game(passer_id, week, att + 2, 0.0, season=season), "cpoe_sum": cpoe_sum, "cpoe_att": att}


def test_cpoe_carries_over_and_uses_only_earlier_weeks():
    from src.features.history import cpoe_features
    from src.features.target import build_targets

    games = pl.DataFrame([
        _g("a", 1, 300.0, 30, season=2023),  # last season: +10 over 30 throws
        _g("a", 1, 100.0, 20),               # week 1 this season: +5 over 20
        _g("a", 2, -900.0, 30),              # the game being predicted: must not count
        _g("rookie", 1, 50.0, 10),
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    k = 100
    f = cpoe_features(games, build_targets(games), k=k)
    get = lambda qb, wk: f.filter(pl.col("passer_id") == qb, pl.col("season") == 2024, pl.col("week") == wk)["cpoe_carry"][0]
    start = 300.0 / (30 + k)
    assert abs(get("a", 1) - start) < 1e-9                              # week 1: last season only
    assert abs(get("a", 2) - (100.0 + k * start) / (20 + k)) < 1e-9      # week 1 piles on; week 2 excluded
    assert get("rookie", 1) == 0.0                                       # no history: average accuracy
