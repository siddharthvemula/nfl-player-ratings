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
