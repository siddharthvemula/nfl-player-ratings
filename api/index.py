"""FastAPI backend (Phase 6): serves the precomputed JSON in site_data/. No model runs per request.

Local:   .venv/bin/uvicorn api.index:app --reload        (then open http://127.0.0.1:8000/docs)
Vercel:  this file is the Python serverless function; vercel.json routes /api/* here.

Only depends on fastapi (see requirements.txt): the ML packages never ship to the server.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("SITE_DATA_DIR", Path(__file__).resolve().parents[1] / "site_data"))

app = FastAPI(title="NFL QB Ratings API", version="1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")


# ---- Response shapes --------------------------------------------------------------------------------------------

class Rating(BaseModel):
    """One QB-game: the rating made before it, the SHAP pieces behind it, and what happened."""

    season: int
    week: int
    season_type: str
    game_id: str
    passer_id: str
    passer: str  # short name as in the play-by-play ("J.Herbert")
    name: str  # full name ("Justin Herbert"), from nflverse's players table
    headshot: str | None
    team: str
    opponent: str | None
    upcoming: bool
    rank: int
    rating: float  # opponent-neutral: defense set to league average (what the rankings use)
    matchup_rating: float  # vs the actual opponent
    actual: float | None  # EPA per dropback in that game (None until it's played)
    dropbacks: int | None
    evidence: int  # dropbacks of evidence: last season + this season so far
    confidence: str  # Low / Med / High
    carry_over: float
    avg_adjustment: float
    contributions: dict[str, float]  # matchup_rating = carry_over + avg_adjustment + sum(contributions)
    features: dict[str, float]
    raw_epa: float | None  # his raw EPA/dropback this season before this game (None before his first dropback)
    season_dropbacks: int  # dropbacks behind raw_epa
    prev_rank: int | None  # his most recent earlier rank this season


class TestGame(BaseModel):
    """One 2025 game from the one-time test: the forecast (vs the actual opponent) and what happened."""

    passer_id: str
    name: str
    team: str
    week: int
    opponent: str | None
    predicted: float
    actual: float
    dropbacks: int


class Rankings(BaseModel):
    season: int
    week: int
    upcoming: bool
    rows: list[Rating]


class PlayerSummary(BaseModel):
    passer_id: str
    passer: str
    name: str
    headshot: str | None
    team: str
    first_season: int
    last_season: int
    games: int


class Player(BaseModel):
    passer_id: str
    passer: str
    name: str
    headshot: str | None
    team: str
    games: list[Rating]


# ---- Data, loaded once per server start -------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _data() -> dict:
    load = lambda name: json.loads((DATA_DIR / name).read_text())
    ratings = load("ratings.json")
    by_player: dict[str, list[dict]] = {}
    for r in ratings:
        by_player.setdefault(r["passer_id"], []).append(r)
    return {"meta": load("meta.json"), "experiments": load("experiments.json"), "test_2025": load("test_2025.json"),
            "ratings": ratings, "by_player": by_player}


# ---- Endpoints ---------------------------------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "updated_at": _data()["meta"]["updated_at"]}


@app.get("/api/meta")
def meta() -> dict:
    """Current week, played weeks, each season's fitted model, confidence-tier errors, the 2025 test, team colors."""
    return _data()["meta"]


@app.get("/api/rankings", response_model=Rankings)
def rankings(season: int | None = Query(None), week: int | None = Query(None)) -> Rankings:
    """A week's QB rankings by opponent-neutral rating. No season/week = the current (upcoming) week."""
    d = _data()
    current = d["meta"]["current"]
    season = current["season"] if season is None else season
    week = current["week"] if week is None else week
    if current["upcoming"] and (season, week) == (current["season"], current["week"]):
        rows = [r for r in d["ratings"] if r["upcoming"]]
        upcoming = True
    else:
        rows = [r for r in d["ratings"] if r["season"] == season and r["week"] == week and not r["upcoming"]]
        upcoming = False
    if not rows:
        raise HTTPException(404, f"No ratings for {season} week {week}")
    return Rankings(season=season, week=week, upcoming=upcoming, rows=sorted(rows, key=lambda r: r["rank"]))


@app.get("/api/players", response_model=list[PlayerSummary])
def players() -> list[PlayerSummary]:
    """Every rated QB, most recently active first."""
    out = []
    for pid, games in _data()["by_player"].items():
        last = games[-1]
        out.append(PlayerSummary(passer_id=pid, passer=last["passer"], name=last["name"], headshot=last["headshot"],
                                 team=last["team"], first_season=games[0]["season"],
                                 last_season=last["season"], games=sum(not g["upcoming"] for g in games)))
    return sorted(out, key=lambda p: (-p.last_season, p.name))


@app.get("/api/players/{passer_id}", response_model=Player)
def player(passer_id: str) -> Player:
    """One QB's full rating history (oldest first), including his next game if he's in the current rankings."""
    games = _data()["by_player"].get(passer_id)
    if not games:
        raise HTTPException(404, f"No QB with id {passer_id}")
    last = games[-1]
    return Player(passer_id=passer_id, passer=last["passer"], name=last["name"], headshot=last["headshot"], team=last["team"],
                  games=games)


@app.get("/api/experiments")
def experiments() -> list[dict]:
    """Every table in experiments.md: {title, notes, columns, rows: [{cells, status: keep|drop|reference}]}."""
    return _data()["experiments"]


@app.get("/api/test-2025", response_model=list[TestGame])
def test_2025() -> list[TestGame]:
    """Every game of the one-time 2025 test (624 games): what the model forecast and what happened."""
    return _data()["test_2025"]
