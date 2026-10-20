# Next steps

## Where we left off
Phase 3 started in notebooks/02_baseline.ipynb.
- Concept 1 done: league-average baseline (+0.056 for everyone): val MAE 0.240 weighted / 0.270 unweighted.
**Next: Concept 2**, season-to-date average per QB (with a fallback for rows with no history).

Phase 2 summary below.

Done so far (notebooks/01_explore.ipynb, train seasons 2016–2023 only; see DECISIONS.md):
- Concept 1: `src/features/qb_games.py` builds one row per passer per game.
- Concept 2: sample size via min 3 dropbacks + dropback weights.
- Concept 3: outliers; target stays raw EPA (feature cap + Huber queued for Phase 4).
- Concept 4: missing values; group by passer_id, cpoe averaged over non-blank rows only.
- Concept 5: target uses `qb_epa` (receiver fumbles after the catch aren't on the QB).
- Concept 6: playoff games included (same QBs play about the same; dropping them is a biased hole).
- Locked target: `build_targets(qb_games(pbp))`; history for features = full `qb_games()` table.

## Phase plan
1. [x] **Data**: download + cache play-by-play with nflreadpy.
2. [x] **Explore + define target**: explore QB dropbacks; pin down next-week EPA/dropback.
3. [ ] **Baseline + evaluation**: simple baselines (e.g. season-to-date average), MAE on 2024 val.
4. [ ] **Features + Ridge**: engineered features (no leakage), Ridge regression.
5. [ ] **LightGBM + SHAP**: gradient-boosted model; SHAP to explain predictions.
6. [ ] **FastAPI backend**: serve ratings, player history, experiments (api/).
7. [ ] **React frontend**: weekly rankings, player pages, Model Lab page showing every
   experiment and its accuracy (frontend/).
8. [ ] **Deploy + weekly auto-update**: free hosting; scheduled job to refresh data and ratings.
