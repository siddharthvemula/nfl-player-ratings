# Next steps

## Where we left off
Phase 2, **Concept 6 (playoffs)** cells + Play cell added; waiting on Sid's playoffs call. Then lock the target. When locking it, switch
`qb_games()` to average `qb_epa` (Concept 5 decision; the notebook still uses `epa` until then).

Done so far (notebooks/01_explore.ipynb, train seasons 2016–2023 only; see DECISIONS.md):
- Concept 1: `src/features/qb_games.py` builds one row per passer per game.
- Concept 2: sample size via min 3 dropbacks + dropback weights.
- Concept 3: outliers; target stays raw EPA (feature cap + Huber queued for Phase 4).
- Concept 4: missing values; group by passer_id, cpoe averaged over non-blank rows only.
- Concept 5: target uses `qb_epa` (receiver fumbles after the catch aren't on the QB).

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
