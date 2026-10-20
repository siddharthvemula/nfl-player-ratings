import polars as pl

from src.models.baselines import league_average, predict_constant
from src.models.evaluate import mae


def _rows(targets, weights):
    return pl.DataFrame({"target": targets, "weight": weights}, schema={"target": pl.Float64, "weight": pl.UInt32})


def test_mae_weighted_and_unweighted():
    # Errors 0.1 (weight 30) and 0.5 (weight 10).
    df = _rows([0.1, -0.5], [30, 10]).with_columns(pl.lit(0.0).alias("prediction"))
    scores = mae(df)
    assert abs(scores["unweighted"] - 0.3) < 1e-9
    assert abs(scores["weighted"] - (0.1 * 30 + 0.5 * 10) / 40) < 1e-9


def test_league_average_is_dropback_weighted():
    train = _rows([0.4, -0.2], [30, 10])
    assert abs(league_average(train) - (0.4 * 30 - 0.2 * 10) / 40) < 1e-9


def test_constant_prediction_fills_every_row():
    preds = predict_constant(_rows([0.1, 0.2, 0.3], [5, 5, 5]), 0.05)
    assert preds["prediction"].to_list() == [0.05, 0.05, 0.05]


def _game(passer_id, week, dropbacks, epa_sum, season=2024):
    return {
        "season": season, "week": week, "season_type": "REG", "game_id": f"{season}_{week:02d}_{passer_id}",
        "posteam": "KC", "defteam": "BAL", "passer_id": passer_id, "passer": passer_id,
        "dropbacks": dropbacks, "epa_sum": epa_sum, "epa_per_db": epa_sum / dropbacks,
    }


def test_season_to_date_uses_only_earlier_games_and_falls_back():
    from src.features.target import build_targets
    from src.models.baselines import season_to_date

    games = pl.DataFrame([
        _game("qb", 1, 10, 1.0),   # +0.1
        _game("qb", 2, 2, -4.0),   # 2-dropback game: not a target, but counts as history
        _game("qb", 3, 20, 10.0),  # the game itself must never leak into its own prediction
        _game("qb", 1, 10, 9.0, season=2025),  # new season: history resets
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    preds = season_to_date(games, build_targets(games), fallback=0.05).sort("season", "week")
    assert preds["week"].to_list() == [1, 3, 1]
    p = preds["prediction"].to_list()
    assert abs(p[0] - 0.05) < 1e-9                    # first game: fallback
    assert abs(p[1] - (1.0 - 4.0) / (10 + 2)) < 1e-9  # weeks 1-2 only
    assert abs(p[2] - 0.05) < 1e-9                    # 2025 week 1: fallback again


def test_shrinkage_blends_toward_league_average():
    from src.features.target import build_targets
    from src.models.baselines import shrunk_to_date

    games = pl.DataFrame([
        _game("qb", 1, 50, -10.0),  # -0.2 per dropback
        _game("qb", 2, 40, 4.0),
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    preds = shrunk_to_date(games, build_targets(games), league_avg=0.05, k=200).sort("week")
    p = preds["prediction"].to_list()
    assert abs(p[0] - 0.05) < 1e-9                             # no history: pure league average
    assert abs(p[1] - (-10.0 + 200 * 0.05) / (50 + 200)) < 1e-9  # 50 real + 200 imaginary dropbacks


def test_carry_over_starts_from_shrunk_last_season():
    from src.features.target import build_targets
    from src.models.baselines import carry_over

    games = pl.DataFrame([
        _game("vet", 1, 500, 100.0, season=2023),  # +0.2 last season
        _game("vet", 1, 40, -4.0, season=2024),
        _game("vet", 2, 40, 0.0, season=2024),
        _game("rookie", 1, 40, 0.0, season=2024),
    ]).with_columns(pl.col("dropbacks").cast(pl.UInt32))
    preds = carry_over(games, build_targets(games), league_avg=0.05, k=250)
    get = lambda qb, s, w: preds.filter(pl.col("passer_id") == qb, pl.col("season") == s, pl.col("week") == w)["prediction"][0]

    start = (100.0 + 250 * 0.05) / (500 + 250)
    assert abs(get("vet", 2024, 1) - start) < 1e-9                         # week 1: last season, shrunk
    assert abs(get("vet", 2024, 2) - (-4.0 + 250 * start) / (40 + 250)) < 1e-9
    assert abs(get("rookie", 2024, 1) - 0.05) < 1e-9                        # no last season: league average
    assert abs(get("vet", 2023, 1) - 0.05) < 1e-9                           # first season in data
