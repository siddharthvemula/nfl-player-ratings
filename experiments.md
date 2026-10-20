# Experiments

Val MAE is on 2024. **Weighted** (by target-game dropbacks) is the official score; unweighted is logged
alongside it to show whether a change helps across all games or mostly on high-dropback ones.

| date | change | hypothesis | val MAE (weighted) | val MAE (unweighted) | keep/drop |
|------|--------|------------|--------------------|----------------------|-----------|
| 2026-10-20 | Baseline: league average (train, weighted, +0.056) for every QB-game | The floor: a rating that ignores who the QB is. Every model must beat it | 0.240 | 0.270 | keep (baseline) |
| 2026-10-20 | Baseline 2: QB season-to-date EPA/dropback (weeks < N, same season), league-average fallback in his first game | Knowing who the QB is should beat one number for everyone | 0.259 | 0.290 | drop (worse than league average; keep as reference) |
