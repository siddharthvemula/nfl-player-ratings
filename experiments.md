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
