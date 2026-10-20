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

**Target stays raw EPA per dropback; no per-play cap on the target.**
The rating claims to predict next week's real EPA/dropback, so the target must be the real thing.
Capping the target would also shrink MAE simply because the numbers got smaller, which would make
experiments incomparable. The cost is irreducible noise (a future pick-six can't be predicted).
Capping is still allowed on *features* (past weeks), where a huge play probably says little about
the future. Queued as Phase 4 experiments: (1) per-play EPA cap on features, (2) Huber loss if
raw-target outliers distort Ridge.

**Missing values: check *which* rows are blank before dropping or filling them.**
- Group by `passer_id` (see above). `passer_player_id` is blank on all 7,006 train scrambles,
  which average +0.48 EPA vs +0.02 for other dropbacks. Dropping them would have cut
  Josh Allen from 0.150 to 0.101 EPA/dropback and Justin Fields by 0.10: a bias against mobile QBs.
- The 11 train dropbacks with no `passer_id` (and 1 with no EPA) are dropped: too few and scattered to bias anyone.
- `cpoe` is averaged only over rows where it isn't blank, and never filled with 0. Its 22,590 train
  blanks are sacks (10,389), scrambles (7,006) and passes with no cpoe (5,190). Filling with 0 would
  label them "exactly average accuracy" and pull sack-prone and scrambling QBs toward average.
  Note: `pass_attempt == 1` includes sacks in nflverse, so "pass attempts only" isn't a safe filter.

**Target uses `qb_epa`, not `epa` (Sid's call).**
The two differ only on a completed catch followed by a lost fumble: 580 of 168,928 train dropbacks
(0.34%), where `qb_epa` is +4.7 higher on average (Watson 2019 wk 14: −10.8 → +0.52). That changes
546 of 4,785 QB-games, some a lot (Hurts 2022 wk 10: −0.37 → +0.23), but careers by at most +0.03,
with no pattern: receiver fumbles are noise the QB can't repeat, so they can't be predicted from his
past. This is still real, uncapped EPA (consistent with the raw-target decision), chosen once before
any experiment, so all experiments share one target. Side effects: every value rises slightly (mean
QB-game +0.028 → +0.043), and our MAE isn't directly comparable to sources that use raw `epa`.
A QB's own fumbles (e.g. strip-sacks) still count against him in `qb_epa`. Whether *features*
should use `epa` or `qb_epa` is a separate question, left for Phase 4.

**Playoff games are included (Sid's call).**
Playoff QB-games pool to +0.101 EPA/dropback vs +0.037 in the regular season (train), but that gap
is selection (only good QBs get there): the same QBs are −0.005 vs their own regular season, so they
play about the same. Dropping the 199 playoff QB-games would also remove 197 targets, including 109
regular-season games whose next game was a playoff game. All 109 belong to playoff QBs, so that's
non-random missing data (like the scrambles in Concept 4) that removes good QBs' late-season examples.
And the site has to rate QBs during the playoffs anyway. Playoff weeks continue the week numbers
(18–21 before 2021, 19–22 after), so ordering by (season, week) puts them in sequence.
