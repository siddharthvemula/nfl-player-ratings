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

## Phase 2: Explore + define target

**Explore on train seasons (2016–2023) only.**
Choosing the target definition (dropback minimums, playoffs, etc.) while looking at 2024–2025
would let the val/test seasons shape the model, so their scores would no longer be honest.

**Group by `passer_id`, not `passer_player_id`.**
`passer_player_id` is null on every QB scramble (9,288 in 2016–2025), so grouping on it would
silently drop scrambles from each QB's dropbacks. `passer_id` covers passes, sacks and scrambles.

**QB-game table keeps every passer for now.**
There are 1.23 QB-game rows per team-game in train: backups, plus trick-play passers (Taysom Hill,
punters, receivers). Filtering them out is a sample-size decision, made in Concept 2.

**Minimum 3 dropbacks per QB-game, plus sample weights by dropbacks (Sid's call).**
In train, a cutoff of 3 drops 11.5% of rows: mostly trick-play passers (Taysom Hill, punters,
receivers), plus ~180 cameo rows from real QBs (kneel-downs, mop-up), which carry almost no
signal. A higher cutoff throws away real games: at 20, 19.9% of rows go, including starters
hurt or benched early. Instead of a high cutoff, rows are weighted by dropbacks, so a 45-dropback
game counts more than a 5-dropback one. (Weights are applied when training/evaluating, from
Phase 3; exactly which week's dropbacks set the weight is settled when the target is defined.)
