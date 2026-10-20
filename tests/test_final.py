import numpy as np
import polars as pl

from tests.test_walk_forward import _history


def test_explanation_adds_up_and_matches_shap():
    import shap

    from src.features.history import BASE_FEATURES, base_features
    from src.features.target import build_targets
    from src.models.final import check_explanation_adds_up, explain
    from src.models.walk_forward import Experiment, run_fold

    games = _history()
    exp = Experiment("raw", BASE_FEATURES, lambda g, t, la, k: base_features(g, t, la))
    rows, fitted = run_fold(games, build_targets(games), 2024, exp)
    out = explain(rows, fitted, BASE_FEATURES)
    check_explanation_adds_up(out, BASE_FEATURES)

    # Same numbers from the shap library (its background reduces to the training means for a linear model).
    background = np.array([[fitted["feature_means"][c] for c in BASE_FEATURES]])
    ref = shap.LinearExplainer(fitted["model"], background).shap_values(rows.select(BASE_FEATURES).to_numpy())
    ours = out.select(f"{c}_contrib" for c in BASE_FEATURES).to_numpy()
    assert np.abs(ref - ours).max() < 1e-9
