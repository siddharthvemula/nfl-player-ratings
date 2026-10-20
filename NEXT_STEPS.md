# Next steps

## Where we left off
Phase 2 in progress. Concept 1 done: `src/features/qb_games.py` builds one row per passer
per game (dropbacks, epa_per_db); notebook has a Play cell for it. Exploration uses
train seasons (2016–2023) only.
Concept 2 (sample size) covered via the Concept 1 Play cell (min 3 dropbacks + weights).
Concept 3 (outliers) done: target stays raw; feature cap + Huber queued for Phase 4.
**Next:** 4 missing values (passer_player_id is
null on every scramble), 5 epa vs qb_epa, 6 playoffs. Then write the target definition.

## Phase plan
1. [x] **Data**: download + cache play-by-play with nflreadpy.
2. [ ] **Explore + define target**: explore QB dropbacks; pin down next-week EPA/dropback.
3. [ ] **Baseline + evaluation**: simple baselines (e.g. season-to-date average), MAE on 2024 val.
4. [ ] **Features + Ridge**: engineered features (no leakage), Ridge regression.
5. [ ] **LightGBM + SHAP**: gradient-boosted model; SHAP to explain predictions.
6. [ ] **FastAPI backend**: serve ratings, player history, experiments (api/).
7. [ ] **React frontend**: weekly rankings, player pages, Model Lab page showing every
   experiment and its accuracy (frontend/).
8. [ ] **Deploy + weekly auto-update**: free hosting; scheduled job to refresh data and ratings.
