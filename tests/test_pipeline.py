from pathlib import Path

import polars as pl

from tests.test_evaluate import _game


def _games():
    # Two teams' QBs, 2015-2020, three games a season; season 2020 is "live" with weeks 1-2 played.
    rows = []
    for season in range(2015, 2021):
        for week in (1, 2, 3):
            if season == 2020 and week == 3:
                continue
            for qb, team, opp, db in (("a", "KC", "BAL", 30), ("b", "BAL", "KC", 25)):
                g = _game(qb, week, db + week, 0.1 * week - (0.2 if qb == "b" else 0) + 0.01 * (season % 4), season=season)
                rows.append({**g, "posteam": team, "defteam": opp, "game_id": f"{season}_{week:02d}_{qb}",
                             "cpoe_sum": 0.0, "cpoe_att": 0, "sacks": float(week % 2 + 1), "home": 1 if qb == "a" else -1})
    return pl.DataFrame(rows).with_columns(pl.col("dropbacks", "cpoe_att").cast(pl.UInt32))


def _schedule(weeks):
    # weeks: list of (week, home, away, result); result None = not played yet.
    return pl.DataFrame([{"season": 2020, "week": w, "game_type": "REG", "game_id": f"2020_{w:02d}_{a}_{h}",
                          "home_team": h, "away_team": a, "result": r} for w, h, a, r in weeks],
                        schema_overrides={"result": pl.Int32})


def test_placeholders_change_no_played_rating_and_stop_before_the_next_game():
    from src.features.target import build_targets
    from src.pipeline.build_site_data import next_game_placeholders, placeholder_targets, rate_season

    games = _games()
    targets = build_targets(games)
    ph = next_game_placeholders(games, _schedule([(2, "KC", "BAL", 3), (3, "KC", "BAL", None)]), 2020)
    assert ph.sort("passer_id").select("passer_id", "week", "defteam").rows() == [("a", 3, "BAL"), ("b", 3, "KC")]

    before, _ = rate_season(games, targets, 2020)
    after, _ = rate_season(pl.concat([games, ph]), pl.concat([targets, placeholder_targets(ph, targets)]), 2020)
    key = ["passer_id", "week"]
    played = after.filter(pl.col("target").is_not_null()).sort(key)
    assert (played["prediction"] - before.sort(key)["prediction"]).abs().max() < 1e-12

    # The upcoming row sees exactly weeks 1-2 of 2020: all of QB a's dropbacks so far.
    nxt = after.filter(pl.col("target").is_null(), pl.col("passer_id") == "a").row(0, named=True)
    assert nxt["std_dropbacks"] == 31 + 32
    assert nxt["evidence"] == nxt["std_dropbacks"] + nxt["last_dropbacks"]


def test_bye_week_and_season_over():
    from src.pipeline.build_site_data import next_game_placeholders

    games = _games()
    # KC on bye in week 3, plays week 4; BAL plays week 3 against someone else.
    ph = next_game_placeholders(games, _schedule([(3, "BAL", "DEN", None), (4, "KC", "LV", None)]), 2020)
    assert ph.sort("passer_id").select("passer_id", "week", "defteam").rows() == [("a", 4, "LV"), ("b", 3, "DEN")]
    # Season over: no unplayed games -> next week, no opponent (neutral).
    ph = next_game_placeholders(games, _schedule([(2, "KC", "BAL", 3)]), 2020)
    assert ph["week"].to_list() == [3, 3] and ph["defteam"].null_count() == 2
    assert (ph["dropbacks"] == 0).all() and (ph["epa_sum"] == 0).all()


def test_rankings_put_bye_week_qbs_with_everyone():
    from src.pipeline.build_site_data import to_records

    def row(qb, week, neutral, target):
        z = {c: 0.0 for c in ["prediction", "carry_over", "avg_adjustment"]}
        return {"season": 2020, "week": week, "season_type": "REG", "game_id": "g", "passer_id": qb, "passer": qb,
                "posteam": "KC", "defteam": "BAL", "neutral": neutral, "target": target, "weight": None, "evidence": 0,
                "confidence": "Low", **z, **{f"{c}_contrib": 0.0 for c in FEATS}, **{c: 0.0 for c in FEATS}}

    from src.models.final import FINAL_FEATURES as FEATS
    recs = to_records(pl.DataFrame([row("a", 3, 0.1, None), row("b", 4, 0.2, None), row("c", 2, 0.3, 0.5)]))
    ranks = {r["passer_id"]: r["rank"] for r in recs}
    assert ranks == {"b": 1, "a": 2, "c": 1}  # b (week 4, after a bye) and a (week 3) share one ranking


def test_experiments_parser_covers_every_table():
    from src.pipeline.experiments import parse_experiments

    md = (Path(__file__).resolve().parents[1] / "experiments.md").read_text()
    sections = parse_experiments(md)
    lines = md.splitlines()
    tables = sum(1 for i, l in enumerate(lines[1:], 1) if l.startswith("|---") and lines[i - 1].startswith("|"))
    data_rows = sum(1 for i, l in enumerate(lines) if l.startswith("|") and not l.startswith("|---")
                    and not (i + 1 < len(lines) and lines[i + 1].startswith("|---")))
    assert len(sections) == tables
    assert sum(len(s["rows"]) for s in sections) == data_rows
    assert all(r["status"] in {"keep", "drop", "reference"} for s in sections for r in s["rows"])
    status = lambda prefix: next(r["status"] for s in sections for r in s["rows"] if r["cells"].get("change", "").startswith(prefix))
    assert status("C7: C6 + home") == "drop"
    assert status("C6: C4 + sack rate") == "keep"
