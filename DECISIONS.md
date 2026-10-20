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

**Locked target (end of Phase 2; built in `src/features/target.py`).**
Each row is one QB-game we predict, and its features may only use that QB's earlier games
(features for week N use weeks < N, where N is the game being predicted).
- Plays: `qb_dropback == 1` (passes, sacks, scrambles), regular season and playoffs.
- Grouped by `passer_id`; target = mean `qb_epa` over the QB's dropbacks in that game, uncapped.
- Target rows need 3+ dropbacks. Games with 1–2 dropbacks aren't targets, but their plays still
  count in a QB's history when building features (Sid's addition): they're real plays, just too
  few to score a prediction on.
- Weight = dropbacks in the target game, used in training and scoring only. It's only known after
  the game, so it's never a feature: `check_no_leakage()` refuses it (and the other same-game columns)
  and a test enforces that.
- Rows with no history (a QB's first game of a season, or in our data) are kept; baselines and
  models need a fallback for them.
- We only score games the QB actually played: the rating means "how well he'll play if he plays."
- Split by the target game's season: train 2016–2023 (4,785 rows), val 2024 (622), test 2025 (624);
  2026+ is "live" (rated on the site, never scored).
- Official score: dropback-weighted MAE on val. experiments.md also logs unweighted MAE (Sid's
  addition), to show whether a change helps across all games or mostly on high-dropback ones.
Why "row = the game being predicted" rather than "row = game N, target = game N+1": the numbers
are the same, but no last-game-of-season rows get thrown away (they were just the next season's
first game), and it matches the leakage rule exactly.

## Phase 3: Baseline + evaluation

**First baseline: one number for everyone, the train league average (+0.056, dropback-weighted).**
A score only means something next to the score of a simple guess. This one ignores who the QB is,
so any model that can't beat it (val weighted MAE 0.240, unweighted 0.270) has learned nothing.
The number is computed from train only, so nothing from 2024 leaks in. It's weighted because the
official score is weighted. Scoring lives in `src/models/evaluate.py` (`mae()` returns both MAEs),
so every experiment is scored the same way.

**Season-to-date baseline: worse than league average (val 0.259 vs 0.240), kept only as a reference.**
Prediction for week N = the QB's total qb_epa / total dropbacks over his earlier games that season
(1–2 dropback games included, resets each season, league-average fallback in his first game).
On train, it loses badly with 1–100 dropbacks of history (0.334 vs 0.249), still loses at 101–200,
and only edges ahead after 200+ (0.219 vs 0.225). One or two games of EPA are mostly noise, and a raw
average trusts them fully. That points to blending a QB's own average with the league average,
trusting his own more as dropbacks pile up (shrinkage), as the next baseline.

**Shrinkage baseline: K = 250, chosen on train only (val 0.233, best so far).**
Prediction = (his EPA so far + K × league average) / (his dropbacks so far + K), same season,
weeks < N only. K was picked from a grid by train weighted MAE; picking it on val would let 2024 tune
the model. The train curve is flat from about 150 to 400 (0.2284–0.2288), so the exact K barely matters.
It's the first baseline to beat the league average, and on val it wins in every history bucket.
The 2024 weighted MAE to beat is now 0.233. The biggest remaining errors are week-1 games (no history,
val MAE 0.299), which a QB's previous season could fix.

**Carry-over baseline: K = 300, chosen on train (val 0.231, best so far, but only 0.002 better).**
Each QB's season starts from his previous season, shrunk toward the league average with the same K
(rookies and QBs with no dropbacks last season start at the league average). Then this season's
dropbacks pile on as in shrinkage. Train improves by the same small amount (0.2284 → 0.2263), so the
gain isn't a val fluke, but it's small. On val it helps a lot in week 1 (33 games: 0.235 → 0.213), is
slightly worse in weeks 2–4, and slightly better after that, since with K = 300 last season still has
some pull all year. It does nothing for rookies or backups. Kept as the Phase 3 bar to beat.
Open question for later: whether a 0.002 gain is real or noise.

**Bootstrap by QB to judge small gains (`bootstrap_compare()`).**
One val season can't tell whether a 0.002 gain is real. The bootstrap builds 5,000 fake 2024 seasons by
drawing QBs with replacement and counts how often the new model wins. It draws whole QBs, not games,
because one QB's games are linked (carry-over helped or hurt Bryce Young all season), and treating
them as independent would overstate certainty. Results: shrinkage beats league average in 98.0% of fake
seasons (gain 95% range +0.0004 to +0.013); carry-over beats shrinkage in 89.6% (−0.001 to +0.005):
probably real, not certain. Every experiment from Phase 4 on logs its win % against the current best.
The keep/drop call uses the win % together with whether train agrees; there's no fixed cutoff.

**Phase 3 result: carry-over (val 0.231) is the bar to beat.** The whole range from "one number for
everyone" to the best baseline is only 0.009 of MAE, because single games are noisy. Phase 4 gains will
be measured in thousandths, which is why the bootstrap matters.

**Phase 4, Concept 1: linear regression on raw ingredients ties carry-over (val 0.2309 vs 0.2312).**
Features: this-season-so-far EPA/dropback, last season's EPA/dropback, and both dropback counts
(`base_features()`, weeks < N only; "no data" filled with the train league average because a model
can't read a blank). Fit with dropback weights so training matches the weighted MAE. Learned weights:
0.20 on this season and 0.17 on last season, so most of each prediction is pulled toward the average.
The limit: a linear model gives std_avg the same weight at 3 dropbacks as at 300. It over-trusts tiny
samples (Hooker, 3 dropbacks at −1.12, predicted −0.24) and under-trusts big ones (Lamar Jackson, 229
dropbacks at +0.36, predicted +0.15 vs carry-over's +0.21). Carry-over's formula gets that "trust grows with
sample size" pattern built in. The next features should give the model a way to learn it.

**Phase 4, Concept 2: pre-shrunk features lose to carry-over (val 0.2328 vs 0.2312, win 5%). Dropped.**
Shrinking each average (K=250) fixed the within-season trust problem: Hooker's 3-dropback −1.12 became
+0.04. But the model learned fixed weights of 0.86 on this season and 0.51 on last season, and last season
gets 0.51 in week 2 *and* week 17. Carry-over hands off between seasons: last season's pull fades as
this season's dropbacks pile up. Split by week, the shrunk model wins slightly in weeks 1-4 (−0.001)
and loses in weeks 5-9 (+0.004) and 10+ (+0.001). Lesson: a linear model can only express a pattern
that one of its features contains. The next features have to carry the handoff between seasons.
Note on the bootstrap: there's still no fixed win-% cutoff (see above), but 5% is a clear loss.

**Hard keep/drop rule (decided 2026-10-20, replaces "no fixed cutoff" above).**
Keep a change only if BOTH hold:
1. Its bootstrap win rate against the current best (`bootstrap_compare()`, 5,000 QB-resampled 2024 seasons) is **at least 80%**.
2. Train seasons (2016-2023) show a gain **in the same direction** (lower weighted train MAE than the current best).
Otherwise drop it. Why: Phase 4 gains are measured in thousandths, and with one val season it's easy to
talk yourself into a lucky number. A fixed rule, set before seeing the next results, removes that
temptation. Requiring train to agree guards against a change that happens to suit 2024's QBs.
Not retroactive: carry-over (val 0.2312) stays the current best. Concept 1's linear model (win 54.5%)
was kept only as a code base for later features, never as the new best.
