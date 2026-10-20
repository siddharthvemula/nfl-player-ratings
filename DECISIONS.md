# Decisions

Why each important choice was made. Newest at the bottom.

## Phase 1: Data

**Target: next-week EPA per dropback.**
EPA (expected points added) measures how much each play changed the offense's expected
score, so it credits a QB for context (a 3rd-and-15 conversion is worth more than a
garbage-time checkdown). Averaging over dropbacks (passes, sacks, scrambles) includes
sacks and scrambles, which are partly on the QB. Predicting *next* week turns a rating into
something testable: a good rating is one that forecasts future play.

**Time-based splits: train 2016–2023, validate 2024, test 2025.**
In real use the model only ever predicts the future from the past. A random split would
let it learn from 2025 games to "predict" 2019 ones, which inflates accuracy. 2025 is held
back untouched until the end so the final score is honest.

**Data source: nflreadpy (nflverse).**
Free, maintained, and its play-by-play already includes EPA and CPOE, so there's no need to
build those models ourselves.

**Cache one parquet file per season in data/raw/.**
Past seasons never change, so each one is downloaded once. One file per season means only
the in-progress season needs refreshing (`load_pbp(refresh_current=True)`). Parquet is
compact and fast to read. Data isn't committed because it's large and can be re-downloaded.

**Python 3.12 in .venv.**
nflreadpy needs 3.10+; the system Python is 3.9.

## Stack

**FastAPI backend + React frontend, free hosting only.**
This is the typical shape of an ML web app: Python serves predictions, and JavaScript
renders the UI. Paid hosting is ruled out; the specific free hosts get chosen in Phase 8.
