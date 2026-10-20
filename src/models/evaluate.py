"""Scoring: how far predictions are from the target (see DECISIONS.md, "Locked target").

The official score is dropback-weighted MAE on 2024 val; unweighted MAE is logged
alongside it to show whether a change helps across all games or mostly on
high-dropback ones.
"""

import polars as pl

from src.features.target import TARGET_COL, WEIGHT_COL


def mae(df: pl.DataFrame, pred_col: str = "prediction") -> dict[str, float]:
    """Return {"weighted": ..., "unweighted": ...} mean absolute error of pred_col vs the target."""
    # Step 1: how wrong each game's prediction was, ignoring direction.
    error = (df[TARGET_COL] - df[pred_col]).abs()
    weight = df[WEIGHT_COL].cast(pl.Float64)
    return {
        # Step 2: a 45-dropback game counts 45x as much as a 1-dropback game would.
        "weighted": float((error * weight).sum() / weight.sum()),
        # Step 3: every game counts the same.
        "unweighted": float(error.mean()),
    }


def bootstrap_compare(
    df: pl.DataFrame, pred_a: str, pred_b: str, group_col: str = "passer_id", n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    """How often does pred_b beat pred_a on resampled seasons? (Phase 3, Concept 5)

    Each fake season draws whole groups (default: QBs) at random, with replacement,
    because one QB's games are linked. Returns the observed weighted-MAE gain
    (a minus b; positive = b is better), the share of resamples where b wins, and
    a 95% range for the gain.
    """
    import numpy as np

    # Step 1: per-game weighted errors for both models, and which group each game belongs to.
    weight = df[WEIGHT_COL].cast(pl.Float64).to_numpy()
    err_a = (df[TARGET_COL] - df[pred_a]).abs().to_numpy() * weight
    err_b = (df[TARGET_COL] - df[pred_b]).abs().to_numpy() * weight
    groups, group_of_row = np.unique(df[group_col].to_numpy(), return_inverse=True)

    # Step 2: add those up per group, so a fake season is just a sum over drawn groups.
    sum_a = np.bincount(group_of_row, err_a, len(groups))
    sum_b = np.bincount(group_of_row, err_b, len(groups))
    sum_w = np.bincount(group_of_row, weight, len(groups))

    # Step 3: n_boot fake seasons. counts[i, g] = how many times group g was drawn in season i.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    counts = np.zeros((n_boot, len(groups)))
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1)
    gains = (counts @ sum_a - counts @ sum_b) / (counts @ sum_w)

    return {
        "gain": float((sum_a.sum() - sum_b.sum()) / sum_w.sum()),
        "win_rate": float((gains > 0).mean()),
        "low": float(np.percentile(gains, 2.5)),
        "high": float(np.percentile(gains, 97.5)),
    }
