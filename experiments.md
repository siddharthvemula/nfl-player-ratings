# Experiments

Val MAE is on 2024. **Weighted** (by target-game dropbacks) is the official score; unweighted is logged
alongside it to show whether a change helps across all games or mostly on high-dropback ones.
**Win %** = share of 5,000 bootstrapped 2024 seasons (QBs drawn with replacement) where the change beats
the previous best (`bootstrap_compare()` in src/models/evaluate.py); about 50% means a coin flip.
**Keep rule** (DECISIONS.md): win % ≥ 80 AND weighted train MAE also better than the current best. Otherwise drop.

| date | change | hypothesis | val MAE (weighted) | val MAE (unweighted) | keep/drop |
|------|--------|------------|--------------------|----------------------|-----------|
| 2026-10-20 | Baseline: league average (train, weighted, +0.056) for every QB-game | The floor: a rating that ignores who the QB is. Every model must beat it | 0.240 | 0.270 | keep (baseline) |
| 2026-10-20 | Baseline 2: QB season-to-date EPA/dropback (weeks < N, same season), league-average fallback in his first game | Knowing who the QB is should beat one number for everyone | 0.259 | 0.290 | drop (worse than league average, win 0.5%; keep as reference) |
| 2026-10-20 | Baseline 3: shrinkage, season-to-date blended with K=250 imaginary league-average dropbacks (K picked on train) | Trusting a QB only as his dropbacks pile up beats both extremes | 0.233 | 0.263 | keep (win 98.0% vs league average) |
| 2026-10-20 | Baseline 4: carry-over, shrinkage that starts each QB from his own shrunk previous season (K=300, picked on train) | Week 1 had nothing to go on; last season should fix the early weeks | 0.231 | 0.261 | keep (win 89.6% vs shrinkage; same small gain on train). Phase 3 bar to beat |
| 2026-10-20 | Phase 4 C1: linear regression on carry-over's raw ingredients (std_avg, std_dropbacks, last_avg, last_dropbacks), fit on train weighted by dropbacks | Learned weights should beat hand-picked K=300 | 0.231 (0.2309) | 0.260 | keep as Phase 4 base (ties carry-over: win 54.5%, gain −0.006 to +0.006). Fixed weights can't trust std_avg more as dropbacks grow |
| 2026-10-20 | Phase 4 C2: linear regression on pre-shrunk averages (std_shrunk, last_shrunk, K=250) | Shrunk features carry "trust grows with dropbacks", so one fixed weight should beat carry-over | 0.233 (0.2328) | 0.263 | drop (worse than carry-over: win 5.0%; train also slightly worse, 0.2267 vs 0.2263). Last season's weight never fades as this season piles up |
| 2026-10-20 | Phase 4 C3: linear regression on carry-over split into two pieces (last_piece = last season shrunk x dimmer, this_piece = this season x (1 - dimmer), K=300) | Learned weights on the handoff should edge out carry-over's fixed 1 and 1 | 0.232 (0.2322) | 0.262 | drop (win 13.1%; train slightly better 0.2261 vs 0.2263, val worse: fails both halves of the rule). Learned 0.83 / 1.03, intercept −0.015: no new information, just reweighting |
| 2026-10-20 | Phase 4 C4: + opponent defense strength (def_shrunk = EPA/dropback allowed in weeks < N, shrunk K=200 picked on train), linear on [carry_over, def_shrunk] | First genuinely new information; defenses vary almost as much as QBs (season SD 0.09 vs 0.12) | 0.233 (0.2327) | 0.263 | drop (win 20.4%; train better 0.2245 vs 0.2263 but val worse: fails the rule). Held out one train season at a time, defense helps in 6 of 8 (avg +0.0016); 2024 is one of the off years |

## Protocol v2: walk-forward (folds 2019-2024, trained on earlier seasons only), residual on carry-over

Pooled MAE is dropback-weighted over all six folds. Win % = pooled bootstrap (QBs resampled) vs the current best
at the time. Keep = win ≥ 80% AND better in ≥ 4 of 6 seasons. Reproduced in notebooks/03_features_ridge.ipynb.

| date | change | vs | pooled MAE | vs MAE | win % | seasons won | keep/drop |
|------|--------|----|-----------|--------|-------|-------------|-----------|
| 2026-10-20 | Carry-over (K re-picked per fold: 300 in all six) | — | 0.2287 | — | — | — | starting best |
| 2026-10-20 | Control (not a candidate): carry-over + constant shift only | carry-over | 0.2285 | 0.2287 | 74.4 | 4/6 | reference only |
| 2026-10-20 | C1: + raw features (std_avg, std_dropbacks, last_avg, last_dropbacks) | carry-over | 0.2276 | 0.2287 | 98.9 | 5/6 | **keep** (new best). Beats the constant-shift control in 6/6 seasons |
| 2026-10-20 | C2: + shrunk features (K per fold) | C1 | 0.2286 | 0.2276 | 0.4 | 0/6 | drop |
| 2026-10-20 | C3: + handoff pieces (K per fold) | C1 | 0.2286 | 0.2276 | 0.4 | 1/6 | drop |
| 2026-10-20 | C4: C1 + opponent defense (def_shrunk, K per fold: 100-300; weight 0.41-0.61) | C1 | **0.2260** | 0.2276 | 99.8 | 4/6 | **keep** (new best). Per-season gain: +.0038 +.0044 +.0005 −.0001 +.0012 −.0002 |

### Protocol v2 re-run with 2015 loaded as history only (supersedes the table above)

| date | change | vs | pooled MAE | vs MAE | win % | seasons won | per-season gain 2019..2024 | keep/drop |
|------|--------|----|-----------|--------|-------|-------------|----------------------------|-----------|
| 2026-10-20 | Naive baseline: league average (re-computed per fold from earlier seasons: +.058 to +.066) for every QB-game | — | 0.2367 | — | — | — | — | floor (reference only) |
| 2026-10-20 | Carry-over (fold K: 300, 300, 300, 300, 400, 400) | league avg | 0.2287 | 0.2367 | 100.0 | 6/6 | +.0045 +.0141 +.0078 +.0055 +.0079 +.0081 | starting best |
| 2026-10-20 | Control (not a candidate): constant shift only | carry-over | 0.2286 | 0.2287 | 68.0 | 4/6 | +.0002 −.0021 +.0012 +.0021 +.0008 −.0014 | reference only |
| 2026-10-20 | C1: + raw features | carry-over | 0.2275 | 0.2287 | 96.8 | 5/6 | +.0017 −.0021 +.0024 +.0029 +.0021 +.0006 | **keep** |
| 2026-10-20 | C2: + shrunk features | C1 | 0.2287 | 0.2275 | 1.3 | 0/6 | −.0017 −.0001 −.0017 −.0003 −.0015 −.0019 | drop |
| 2026-10-20 | C3: + handoff pieces | C1 | 0.2286 | 0.2275 | 1.3 | 0/6 | −.0015 −.0004 −.0020 −.0000 −.0013 −.0016 | drop |
| 2026-10-20 | C4: C1 + opponent defense | C1 | **0.2259** | 0.2275 | 99.7 | 4/6 | +.0042 +.0047 +.0006 −.0001 +.0008 −.0005 | **keep** (current best) |
| 2026-10-20 | C5: C4 + CPOE (cpoe_carry: carry-over recipe on CPOE, shrunk toward 0 by attempts; fold K (def, CPOE): (200,500) (100,800) (200,800) (200,1200) ×3). Hypothesis: accuracy stabilizes faster than EPA, so it helps most in weeks 1-4 | C4 | 0.2260 | 0.2259 | 35.2 | 3/6 | −.0009 +.0004 −.0001 +.0008 +.0004 −.0012 | drop. CPOE weight negative in all six folds (−0.013 to −0.018 per CPOE point). Weeks 1-4 got *worse* (−.0006), 10+ slightly better (+.0004) |
| 2026-10-20 | C6: C4 + sack rate (sack_carry: carry-over recipe on sacks per dropback, shrunk toward the league rate of earlier seasons; fold K (def, sack): (300,50) (50,1200) (300,1200) (200,1200) ×3). Hypothesis: sack rate is QB-owned and stabilizes fast. Pre-check on 2016-2023: corr −0.48 with carry-over, +0.04 with C4's errors (CPOE: −0.01) | C4 | **0.2256** | 0.2259 | 93.2 | 4/6 | −.0003 −.0006 +.0002 +.0012 +.0007 +.0006 | **keep** (current best). Weight +2.3 to +2.6 in five folds (2019 fold: K=50, weight −0.09). Gain is largest in weeks 1-4 (+.0005), smallest in weeks 10+ (+.0001) |
| 2026-10-20 | C7: C6 + home (+1 home, −1 away, 0 neutral site; same K grid as C6). Hypothesis: home crowds hurt the visiting offense's passing. Pre-check on 2016-2023: corr 0.00 with C4's errors; raw EPA/dropback home +.056 vs away +.054 | C6 | 0.2257 | 0.2256 | 17.6 | 4/6 | −.0010 +.0002 +.0002 +.0001 −.0000 +.0000 | drop. Home weight −0.0035 to +0.012 EPA/dropback across folds: no stable home edge in passing EPA |
| 2026-10-20 | Final Phase 4 model: Ridge on C6's features (scaled to SD 1; alpha picked per fold on season S−1 from 0.1-30000: 30000, 0.1, 0.1, 10000, 10000, 30000). Hypothesis: a penalty on big weights steadies six correlated features | C6 | 0.2257 | 0.2256 | 17.9 | 5/6 | −.0011 +.0000 +.0000 +.0000 +.0002 +.0002 | drop: ties plain linear except 2019, where 3 training seasons picked the biggest penalty. **Final Phase 4 model = C6, plain linear (pooled 0.2256)** |

Floor check (at C4): the then-best (C4) beats the league average by 0.0108 pooled (win 100%, 6/6 seasons; per season +.0105 +.0166 +.0108 +.0083 +.0107 +.0082).

## Phase 5: gradient boosting (same walk-forward protocol, same six features as C6)

| date | change | vs | pooled MAE | vs MAE | win % | seasons won | per-season gain 2019..2024 | keep/drop |
|------|--------|----|-----------|--------|-------|-------------|----------------------------|-----------|
| 2026-10-20 | LightGBM on C6's features, residual on carry-over (lr 0.03, min 100 games per leaf, 80% subsample; (leaves, trees) picked per fold on season S−1 from {4,8,16}×{100,300,600}: (4,100) ×5, (8,100) in 2022). Hypothesis (Sid: tie): trees can learn interactions (defense matters mostly for veterans), but 4,000 noisy rows limit them | C6 | 0.2254 | 0.2256 | 67.6 | 3/6 | −.0004 −.0007 +.0005 −.0003 +.0009 +.0012 | drop (tie, as Sid predicted). Low-history QBs (last season ≤ 212 db): +.0010, win 83.9%; veterans −.0001, win 39.4% |
| 2026-10-20 | Reference only (hindsight, not a candidate): LightGBM with (4,100) fixed in every fold | C6 | 0.2252 | 0.2256 | 80.8 | 4/6 | — | not eligible: the setting was chosen after seeing fold scores |
