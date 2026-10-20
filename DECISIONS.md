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

**Phase 4, Concept 3: handoff pieces tie carry-over on train, lose on val (0.2322 vs 0.2312, win 13%). Dropped.**
Carry-over split exactly into last_piece + this_piece (dimmer = 300 / (dropbacks so far + 300)). The model
learned 0.83 and 1.03 (carry-over is 1 and 1) with intercept −0.015: it trims last season a bit. That helped
train by 0.0001 and hurt 2024 by 0.001. Reshuffling the same two numbers has run out: three linear
variants (Concepts 1-3) all land within ±0.002 of carry-over. Part of the gap may be that linear regression
minimizes squared error while we score absolute error (Huber is still queued).

**Phase 4 feature plan (decided 2026-10-20): new information, not reshuffled EPA.** In order:
1. Opponent defense strength (the only feature about the game being predicted)
2. CPOE (stabilizes faster than EPA; should help early weeks)
3. Sack rate (QB-owned, fast to stabilize, overlaps EPA)
4. Home/away (known ahead of time, small effect; neutral sites need their own value)
Requirements for every new feature, learned from Concepts 1-3:
- **Shrink by sample size**: e.g. a defense's EPA allowed so far is pulled toward the league average
  when it has played few games, the way std_shrunk is. A linear model can't learn "trust grows with
  sample size" on its own (Concept 1).
- **Only games before week N** (same as all features). Schedule facts known before kickoff (who the
  opponent is, home/away) are allowed; anything measured in the target game is not.
Each feature is one experiment and one commit, judged by the hard keep rule against the current best.

**Phase 4, Concept 4: opponent defense strength fails the rule on 2024 (val 0.2327 vs 0.2312, win 20%). Dropped.**
`defense_features()`: EPA/dropback the opponent allowed to every QB in weeks < N of the same season
(its week N game excluded, since it contains the target game), shrunk toward the league average with
K=200 (best train MAE in a 50-2000 grid; the curve is flat). Learned: 0.98 x carry-over + 0.50 x def_shrunk
− 0.048. Train improved by 0.0018, the biggest train gain in Phase 4, but 2024 got worse, so the hard rule says drop.
Supporting evidence recorded for later: holding out each train season in turn, adding defense beats a
linear refit of carry-over alone in 6 of 8 seasons (average +0.0016); 2024 (−0.0003) and 2018 (−0.0012)
are the off years. About 0.0012 of the 2024 loss comes from refitting carry-over itself (0.98 weight +
intercept), not from the defense term. Not overriding the rule; possible follow-up experiment: keep
carry-over's weight fixed at 1 and learn only the defense adjustment.

**Evaluation protocol v2: walk-forward CV + residual modeling (decided 2026-10-20, before any results under it).**
Replaces single-season val scoring and the earlier hard keep rule.
- **Walk-forward folds:** for each season S in 2019-2024, train only on seasons before S (2016..S-1) and
  predict S. Nothing from a season after S is ever used to score S. That covers model weights *and* every
  number fitted from data: the league average and every K (carry-over K, shrink K, defense K) are
  re-picked inside each fold on its earlier seasons only. Score = dropback-weighted MAE on all six folds
  pooled. Why: one val season is too noisy (Concept 4's train gain vs 2024 loss), and the leave-one-season-
  out check in Concept 4 trained on *future* seasons, which can't happen in real use; walk-forward can't.
- **Keep rule:** keep a change only if BOTH hold against the current best:
  1. pooled bootstrap win rate ≥ 80% (`bootstrap_compare()`, QBs resampled across the pooled folds), and
  2. it beats the current best in a majority of the individual fold seasons (at least 4 of 6).
  Otherwise drop.
- **Residual modeling by default:** carry-over's weight is fixed at 1. Features only learn an adjustment:
  prediction = carry_over + (w0 + w1*f1 + ...), fit on (target − carry_over). Why: in Concepts 1-4 the
  model kept re-learning carry-over's own scale (0.98, 0.83/1.03), which cost val accuracy and hid
  whether the new feature helped.
- **2025 is untouched** until the final test in Phase 5: the walk-forward code drops 2025 from both the
  target rows and the history table before anything else runs.

**Protocol v2 re-run of Concepts 1-4 (results).** Current best is now **carry-over + C1 raw features +
opponent defense, pooled walk-forward MAE 0.2260** (carry-over 0.2287). In order:
- C1 kept (win 98.9%, 5/6 seasons). It was a tie under v1 because the model re-learned carry-over's scale;
  as a residual it finds something real: QBs with little or no last season (backups, rookies) come in
  0.076 EPA/dropback *below* carry-over, because carry-over starts them at the league average. The dropback
  counts let the adjustment push them down. A constant shift alone (control) doesn't do it: 74% win, and
  C1 beats the control in 6 of 6 seasons. So volume is information: teams give more snaps to better QBs.
- C2 and C3 dropped against C1 (0.4% win each): once carry-over is fixed, re-expressing the same two
  averages only adds noise.
- C4 kept (C1 + defense vs C1: win 99.8%, 4/6 seasons; the two "losses" are −0.0001 and −0.0002).
  Under v1 the same feature failed on 2024 alone; the v1 leave-one-season-out check also trained on future
  seasons, which walk-forward avoids. Learned defense weight is stable across folds (0.41-0.61).
**Caveat to fix (data):** play-by-play starts in 2016, so every 2016 QB looks like he had no last season.
That mislabels 2016 veterans as "no history" in the training data of every fold, which blurs C1's
backup/rookie signal. Proposed fix: load 2015 as history only (never a target row).

**Data-quality fix: 2015 loaded as history only (decided 2026-10-20).**
Problem: play-by-play started in 2016, so every 2016 QB looked like he had no previous season. In every
walk-forward fold, 2016 veterans were mislabeled as "no history" (like rookies and backups), which blurs
the signal C1 found. Fix: `load_pbp()` now starts at 2015 (`FIRST_SEASON`), and 2015 is used only for
features (2016's "last season"). It is never a target row: `build_targets()` drops seasons before
`FIRST_TARGET_SEASON` (2016), and `walk_forward()` filters target rows to 2016-2024 as a second guard,
so 2015 can't enter any fold's training rows, league average, or scores. Tested in
tests/test_walk_forward.py. Notebooks 01-02 (Phases 2-3) are left as the historical record on 2016+ data.
Re-run of the v2 table with 2015 history (folds still 2019-2024): conclusions unchanged.
- Carry-over 0.2287 (fold K: 300 for 2019-2022, 400 for 2023-2024).
- C1 kept: 0.2275 (was 0.2276), win 96.8%, 5/6 seasons. QBs with little or no last season now come in
  0.078 below carry-over (was 0.076).
- C2, C3 dropped vs C1 (win 1.3% each, 0/6 seasons).
- C4 kept: **C1 + opponent defense = 0.2259, the current best** (win 99.7% vs C1, 4/6 seasons; defense
  weight 0.43-0.60 across folds).

**Phase 4, Concept 5: CPOE dropped (pooled 0.2260 vs 0.2259, win 35%, 3/6 seasons).**
`cpoe_features()`: carry-over's recipe on CPOE. Each QB starts from last season's CPOE, shrunk toward 0 by
attempts, then this season's throws in weeks < N pile on. Shrinking toward 0 needs no fitted number, since
CPOE is "over expected" by definition. Blanks (sacks, scrambles) are skipped, never filled with 0.
K is picked per fold together with the defense K. The folds chose heavy shrinkage (500-1200).
Why it fails: CPOE is almost entirely a copy of what carry-over already knows. On 2016-2023 it correlates
0.72 with carry-over and −0.01 with carry-over's errors. On top of C1's raw averages, its weight turns
negative (−0.013 to −0.018 per CPOE point, all six folds). It works as a small correction to the EPA averages,
not as new information. Alone on top of carry-over it does a bit better (0.2284 vs 0.2287, win 84%), but only
3/6 seasons, and its sign flips between folds. The early-weeks hypothesis failed: weeks 1-4 got slightly worse.
Lesson: "stabilizes faster" only helps if the faster signal holds something the slower one doesn't. Here,
EPA already includes completions, and carry-over's last season covers the early weeks.
**Leakage note (flagged, not fixed):** nflverse's expected-completion model, which defines CPOE, was fit on
historical seasons that may include fold seasons. The same is true of the EP model behind EPA, which the
target and every feature already use. It's a small, shared risk, not specific to this feature.
Current best is unchanged: C4, 0.2259.

**Phase 4, Concept 6: sack rate kept (pooled 0.2256 vs 0.2259, win 93.2%, 4/6 seasons). New current best.**
`sack_features()`: carry-over's recipe on sacks per dropback. Each QB starts from last season's sack rate, shrunk
toward the league rate, then this season's dropbacks in weeks < N pile on. The league rate comes only from seasons
before the newest one in the table, so a fold never sees the season it scores. Same-game `sacks` is on the leakage
list. Pre-check (training seasons 2016-2023 only, as CPOE's lesson asked): sack rate correlates −0.48 with carry-over,
so half of it repeats what carry-over knows, but +0.044 with C4's errors (CPOE was −0.01). Most folds chose heavy
shrinkage (K=1200).
The weight is *positive*, +2.3 to +2.6 in five folds: a QB with the same past EPA but a 1-point higher sack rate
does about +0.025 EPA/dropback better next week than C4 predicts. Reading: sacks already sit inside EPA, and they
cost a lot of EPA per play, but they repeat less than the rest of a QB's EPA does (offensive line, schedule, game
script all feed them). So carry-over over-punishes them, and this feature gives part of it back. The gain is biggest
in weeks 1-4 (+.0005), when last season's sacks dominate carry-over.
Caveat: the first fold (2019, trained on 2016-2018 only) picked K=50 and a weight near 0, so the signal needs a few
seasons of data to show. The keep rule passes anyway.

**Phase 4, Concept 7: home/away dropped (pooled 0.2257 vs 0.2256, win 17.6%, 4/6 seasons).**
`home_features()`: +1 home, −1 away, 0 neutral site (London, Super Bowl), from the play-by-play's `location` and
`home_team`. It's on the schedule, so it's known before kickoff. Pre-check (2016-2023 only): correlation 0.00 with
carry-over's errors and with C4's; raw EPA/dropback is +0.056 at home vs +0.054 away. Learned weights wander between
−0.0035 and +0.012 across folds. Winning 4 of 6 seasons by ~0.0001 doesn't matter when the pooled bootstrap says it
usually loses. Reading: home field in the NFL is worth ~1.5 points a game at most, and passing EPA per dropback
barely shows it (likely more in run game, penalties, and kicking). Current best stays C6, 0.2256.
