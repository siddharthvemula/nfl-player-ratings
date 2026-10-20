import polars as pl

from tests.test_evaluate import _game


def _g(passer_id, week, dropbacks, sacks, season=2024):
    return {**_game(passer_id, week, dropbacks, 0.0, season=season), "sacks": sacks}


def test_sack_rate_carries_over_and_uses_only_earlier_weeks():
    from src.features.history import sack_features
    from src.features.target import build_targets

    games = pl.DataFrame([
        _g("vet", 1, 100, 10, season=2023),  # last season: 10% sacked; also sets the league rate
        _g("vet", 1, 20, 4),                 # week 1 this season: 20%
        _g("vet", 2, 30, 30),                # the game being predicted: must not count
        _g("rookie", 1, 10, 1),
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    k = 50
    f = sack_features(games, build_targets(games), k=k)
    get = lambda qb, wk: f.filter(pl.col("passer_id") == qb, pl.col("season") == 2024, pl.col("week") == wk)["sack_carry"][0]
    league = 10 / 100  # 2023 only: seasons before the newest target season
    start = (10 + k * league) / (100 + k)
    assert abs(get("vet", 1) - start) < 1e-9                     # week 1: last season only
    assert abs(get("vet", 2) - (4 + k * start) / (20 + k)) < 1e-9  # week 1 piles on; week 2 excluded
    assert abs(get("rookie", 1) - league) < 1e-9                 # no history: league rate



def test_same_game_sacks_are_not_a_feature():
    from src.features.target import check_no_leakage

    try:
        check_no_leakage(["sacks"])
        raise AssertionError("same-game sacks must be refused")
    except ValueError:
        pass
