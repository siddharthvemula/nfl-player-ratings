import json

import pytest
from fastapi.testclient import TestClient


def _rating(pid, season, week, rank, upcoming=False, rating=0.1):
    return {"season": season, "week": week, "season_type": "REG", "game_id": f"{season}_{week}_{pid}", "passer_id": pid,
            "passer": pid.upper(), "team": "KC", "opponent": None if upcoming else "BAL", "upcoming": upcoming, "rank": rank,
            "rating": rating, "matchup_rating": rating, "actual": None if upcoming else 0.2, "dropbacks": None if upcoming else 30,
            "evidence": 500, "confidence": "Med", "carry_over": 0.1, "avg_adjustment": -0.02,
            "contributions": {"def_shrunk": 0.02}, "features": {"def_shrunk": 0.05}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import api.index as api

    ratings = [_rating("a", 2026, 6, 1), _rating("b", 2026, 6, 2), _rating("b", 2026, 8, 1, True, 0.3), _rating("a", 2026, 7, 2, True)]
    meta = {"updated_at": "2026-10-20T00:00:00+00:00", "current": {"season": 2026, "week": 7, "upcoming": True}}
    (tmp_path / "ratings.json").write_text(json.dumps(ratings))
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "experiments.json").write_text(json.dumps([{"title": "t", "rows": []}]))
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    api._data.cache_clear()
    yield TestClient(api.app)
    api._data.cache_clear()


def test_current_rankings_include_bye_week_qbs_in_rank_order(client):
    r = client.get("/api/rankings").json()
    assert (r["season"], r["week"], r["upcoming"]) == (2026, 7, True)
    assert [x["passer_id"] for x in r["rows"]] == ["b", "a"]  # b's next game is week 8 (bye), still listed


def test_past_week_rankings_and_missing_week(client):
    r = client.get("/api/rankings", params={"season": 2026, "week": 6}).json()
    assert [x["passer_id"] for x in r["rows"]] == ["a", "b"] and not r["upcoming"]
    assert client.get("/api/rankings", params={"season": 2001, "week": 1}).status_code == 404


def test_players_and_player_history(client):
    ps = client.get("/api/players").json()
    assert {p["passer_id"]: p["games"] for p in ps} == {"a": 1, "b": 1}  # upcoming games aren't counted as played
    p = client.get("/api/players/b").json()
    assert [g["week"] for g in p["games"]] == [6, 8]
    assert client.get("/api/players/nobody").status_code == 404


def test_meta_and_experiments(client):
    assert client.get("/api/meta").json()["current"]["week"] == 7
    assert client.get("/api/experiments").json()[0]["title"] == "t"
    assert client.get("/api/health").json()["ok"]


def test_real_site_data_fits_the_response_shapes():
    """The committed site_data/ must load and validate (what Vercel will serve)."""
    import api.index as api

    api._data.cache_clear()
    c = TestClient(api.app)
    r = c.get("/api/rankings")
    assert r.status_code == 200 and len(r.json()["rows"]) > 0
    pid = r.json()["rows"][0]["passer_id"]
    assert c.get(f"/api/players/{pid}").status_code == 200
    assert c.get("/api/experiments").status_code == 200
    api._data.cache_clear()
