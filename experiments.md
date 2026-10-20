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
