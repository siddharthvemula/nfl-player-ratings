"""Gradient-boosted trees (Phase 5): LightGBM learning the adjustment on top of carry-over."""

from lightgbm import LGBMRegressor

# (leaves per tree, number of trees). Picked per fold on season S-1 (see walk_forward.pick_param).
LGBM_GRID = tuple((leaves, trees) for leaves in (4, 8, 16) for trees in (100, 300, 600))


def make_lgbm(param: tuple[int, int]) -> LGBMRegressor:
    """A team of small trees, each fixing what the ones before it still get wrong.

    num_leaves         how many yes/no groups each tree may split games into (4 = two questions deep)
    n_estimators       how many trees ("scouts") are added
    learning_rate      each tree's correction is shrunk to 3%, so no single tree can chase a fluke
    min_child_samples  a group needs 100+ games, so a leaf can't be one QB's hot streak
    subsample          each tree sees a random 80% of games (bagging_freq=1: re-drawn every tree)
    """
    leaves, trees = param
    return LGBMRegressor(
        num_leaves=leaves,
        n_estimators=trees,
        learning_rate=0.03,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        random_state=0,
        verbose=-1,
    )
