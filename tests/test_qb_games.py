import polars as pl

from src.features.qb_games import qb_games


def _play(passer_id, epa, qb_dropback=1, posteam="KC", cpoe=0.0, sack=0.0):
    return {
        "season": 2024, "week": 1, "season_type": "REG", "game_id": "2024_01_BAL_KC",
        "posteam": posteam, "defteam": "BAL" if posteam == "KC" else "KC",
        "passer_id": passer_id, "passer": passer_id, "qb_dropback": qb_dropback, "epa": epa, "qb_epa": epa, "cpoe": cpoe,
        "sack": sack, "location": "Home", "home_team": "KC",
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


def test_cpoe_skips_blank_throws():
    pbp = pl.DataFrame([
        _play("mahomes", 1.0, cpoe=20.0),
        _play("mahomes", -0.4, cpoe=-10.0),
        _play("mahomes", -2.0, cpoe=None),  # a sack: no CPOE, must not count as 0
    ])
    row = qb_games(pbp).row(0, named=True)
    assert row["cpoe_sum"] == 10.0 and row["cpoe_att"] == 2 and row["dropbacks"] == 3


def test_sacks_and_home_flag():
    pbp = pl.DataFrame([
        _play("mahomes", -2.0, sack=1.0),
        _play("mahomes", 0.5),
        _play("lamar", 0.1, posteam="BAL"),
    ])
    rows = {r["passer_id"]: r for r in qb_games(pbp).to_dicts()}
    assert rows["mahomes"]["sacks"] == 1 and rows["lamar"]["sacks"] == 0
    assert rows["mahomes"]["home"] == 1  # KC is the home team
    assert rows["lamar"]["home"] == -1
    neutral = qb_games(pbp.with_columns(pl.lit("Neutral").alias("location")))
    assert neutral["home"].to_list() == [0, 0]


def test_null_passer_dropbacks_never_become_rows(capsys):
    pbp = pl.DataFrame([
        _play("mahomes", 1.0),
        _play(None, -3.0),  # a dropback nflverse hasn't tagged with a passer
        _play(None, 0.5, posteam="BAL"),
    ])
    out = qb_games(pbp)
    assert out["passer_id"].null_count() == 0
    assert out.height == 1 and out["dropbacks"].to_list() == [1]
    assert "dropped 2 dropbacks with null passer_id (2024: 2)" in capsys.readouterr().out


def test_null_passer_plays_still_count_for_the_defense():
    from src.features.opponent import defense_features
    from src.features.target import build_targets

    def wk(week, passer_id, epa):
        return {**_play(passer_id, epa), "week": week, "game_id": f"2024_0{week}_BAL_KC"}

    pbp = pl.DataFrame([
        *[wk(1, "mahomes", 0.2) for _ in range(3)],
        wk(1, None, -4.0),            # untagged dropback vs BAL in week 1
        *[wk(2, "mahomes", 0.0) for _ in range(3)],
    ])
    games = qb_games(pbp)
    assert games["passer_id"].null_count() == 0
    f = defense_features(games, build_targets(games), league_avg=0.0, k=0)
    wk2 = f.filter(pl.col("week") == 2)["def_shrunk"][0]
    assert abs(wk2 - (3 * 0.2 - 4.0) / 4) < 1e-9  # the None play is in the defense's 4 dropbacks
