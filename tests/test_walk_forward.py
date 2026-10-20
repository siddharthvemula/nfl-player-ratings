import polars as pl

from tests.test_evaluate import _game


def _history():
    # Two QBs, every season 2015-2025 (2015 = history only), three games each; numbers vary so fits aren't degenerate.
    rows = []
    for season in range(2015, 2026):
        for week in (1, 2, 3):
            rows.append(_game("a", week, 30 + week, 0.1 * week + 0.01 * (season - 2016), season=season))
            rows.append(_game("b", week, 20 + 2 * week, -0.3 + 0.05 * week * (season % 3), season=season))
    return pl.DataFrame(rows).with_columns(pl.col("dropbacks").cast(pl.UInt32))


def _shifted(games, season, amount):
    return games.with_columns(
        pl.when(pl.col("season") == season).then(pl.col("epa_sum") + amount).otherwise(pl.col("epa_sum")).alias("epa_sum")
    )


def test_walk_forward_scores_2019_to_2024_and_never_looks_ahead():
    from src.features.history import BASE_FEATURES, base_features
    from src.features.target import build_targets
    from src.models.walk_forward import Experiment, walk_forward

    exp = Experiment("raw", BASE_FEATURES, lambda g, t, la, k: base_features(g, t, la))
    games = _history()
    pooled, _ = walk_forward(games, build_targets(games), exp)
    assert sorted(pooled["season"].unique().to_list()) == list(range(2019, 2025))  # 2025 never scored

    # Wildly change 2024 and 2025: every earlier fold's predictions must stay exactly the same.
    changed = _shifted(_shifted(games, 2024, 50.0), 2025, -50.0)
    pooled2, _ = walk_forward(changed, build_targets(changed), exp)
    early = lambda p: p.filter(pl.col("season") < 2024).sort("season", "week", "passer_id")["prediction"]
    assert (early(pooled) - early(pooled2)).abs().max() < 1e-12


def test_residual_keeps_carry_over_weight_at_one():
    from src.models.walk_forward import fit_residual, predict_residual

    # Residual is exactly 0.5 * f, so the adjustment should be 0.5 * f with carry-over untouched.
    df = pl.DataFrame({"carry_over": [0.1, 0.3, -0.2], "f": [1.0, 2.0, 3.0], "weight": [10, 20, 30]})
    df = df.with_columns((pl.col("carry_over") + 0.5 * pl.col("f")).alias("target"))
    pred = predict_residual(fit_residual(df, ["f"]), df, ["f"])
    assert (pred["prediction"] - pred["target"]).abs().max() < 1e-9


def test_2015_is_history_only():
    from src.features.history import base_features
    from src.features.target import build_targets
    from src.models.walk_forward import Experiment, walk_forward

    games = _history()
    targets = build_targets(games)
    assert targets["season"].min() == 2016  # 2015 never becomes a target row

    # ...but it does feed 2016's "last season" feature.
    f = base_features(games, targets, league_avg=0.05)
    assert (f.filter(pl.col("season") == 2016)["last_dropbacks"] > 0).all()

    # And walk-forward still scores only 2019-2024.
    exp = Experiment("raw", ["last_avg"], lambda g, t, la, k: base_features(g, t, la))
    pooled, _ = walk_forward(games, targets, exp)
    assert pooled["season"].min() == 2019
