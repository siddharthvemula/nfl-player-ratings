import polars as pl

from tests.test_evaluate import _game


def test_base_features_use_only_the_past():
    from src.features.history import base_features
    from src.features.target import build_targets

    games = pl.DataFrame([
        _game("qb", 1, 500, 100.0, season=2023),  # +0.2 last season
        _game("qb", 1, 10, 1.0),                  # +0.1
        _game("qb", 2, 20, 10.0),                 # the game itself must not count
        _game("rookie", 1, 30, 3.0),
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    f = base_features(games, build_targets(games), league_avg=0.05)
    get = lambda qb, s, w, col: f.filter(pl.col("passer_id") == qb, pl.col("season") == s, pl.col("week") == w)[col][0]
    assert abs(get("qb", 2024, 1, "std_avg") - 0.05) < 1e-9   # nothing this season yet: league average
    assert get("qb", 2024, 1, "std_dropbacks") == 0
    assert abs(get("qb", 2024, 2, "std_avg") - 0.1) < 1e-9    # week 1 only
    assert abs(get("qb", 2024, 2, "last_avg") - 0.2) < 1e-9
    assert get("qb", 2024, 2, "last_dropbacks") == 500
    assert abs(get("rookie", 2024, 1, "last_avg") - 0.05) < 1e-9  # no last season: league average


def test_fit_linear_recovers_known_weights():
    from src.models.linear import fit_linear, predict_linear

    x = [0.0, 1.0, 2.0, 3.0]
    df = pl.DataFrame({"f": x, "target": [0.1 + 0.5 * v for v in x], "weight": [10, 20, 30, 40]})
    model = fit_linear(df, ["f"])
    assert abs(model.intercept_ - 0.1) < 1e-9 and abs(model.coef_[0] - 0.5) < 1e-9
    assert abs(predict_linear(model, df, ["f"])["prediction"][3] - 1.6) < 1e-9


def test_fit_linear_refuses_leaky_features():
    import pytest
    from src.models.linear import fit_linear

    df = pl.DataFrame({"target": [0.1, 0.2], "weight": [10, 20]})
    with pytest.raises(ValueError, match="Leakage"):
        fit_linear(df, ["target"])


def test_shrunk_features_blend_toward_league_average():
    from src.features.history import shrunk_features
    from src.features.target import build_targets

    games = pl.DataFrame([
        _game("qb", 1, 500, 100.0, season=2023),  # +0.2 last season
        _game("qb", 1, 50, -10.0),                # -0.2
        _game("qb", 2, 20, 10.0),
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    f = shrunk_features(games, build_targets(games), league_avg=0.05, k=250).sort("season", "week")
    wk1, wk2 = f.filter(pl.col("season") == 2024).iter_rows(named=True)
    assert abs(wk1["std_shrunk"] - 0.05) < 1e-9                                # no games yet: league average
    assert abs(wk2["std_shrunk"] - (-10.0 + 250 * 0.05) / (50 + 250)) < 1e-9
    assert abs(wk2["last_shrunk"] - (100.0 + 250 * 0.05) / (500 + 250)) < 1e-9
