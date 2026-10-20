# Next steps

## Where we left off
Phase 1 is done: 2016–2026 play-by-play is cached in data/raw/, and
notebooks/01_explore.ipynb loads it and filters to QB dropbacks.
**Next:** Phase 2. Sid explores the notebook; then we define the target precisely
(e.g. minimum dropbacks per week, regular season only?, EPA vs qb_epa).

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
