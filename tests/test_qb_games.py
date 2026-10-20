import polars as pl

from src.features.qb_games import qb_games


def _play(passer_id, epa, qb_dropback=1, posteam="KC"):
    return {
        "season": 2024, "week": 1, "season_type": "REG", "game_id": "2024_01_BAL_KC",
        "posteam": posteam, "defteam": "BAL" if posteam == "KC" else "KC",
        "passer_id": passer_id, "passer": passer_id, "qb_dropback": qb_dropback, "epa": epa,
    }


def test_one_row_per_passer_per_game_with_epa_per_dropback():
    pbp = pl.DataFrame([
        _play("mahomes", 1.0),
        _play("mahomes", -0.4),
        _play("mahomes", 0.6),
        _play("backup", -1.0),
        _play("mahomes", 5.0, qb_dropback=0),  # a run: not a dropback, ignored
        _play("jackson", 0.2, posteam="BAL"),
    ])

    out = qb_games(pbp)

    assert out.height == 3
    mahomes = out.filter(pl.col("passer_id") == "mahomes").row(0, named=True)
    assert mahomes["dropbacks"] == 3
    assert abs(mahomes["epa_per_db"] - 0.4) < 1e-9
