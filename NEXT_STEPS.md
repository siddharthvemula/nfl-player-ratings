# Next steps

## Where we left off
Where we left off: Phases 6-8 (plan approved). Phases 6-8 built. Waiting on Sid's one-time Vercel import; then add the live URL to README.

Phase 3 started in notebooks/02_baseline.ipynb.
- Concept 1 done: league-average baseline (+0.056 for everyone): val MAE 0.240 weighted / 0.270 unweighted.
- Concept 2 done: season-to-date average: val 0.259 / 0.290, *worse* than league average (small samples are noise).
- Concept 3 done: shrinkage (K=250, picked on train): val 0.233 / 0.263, best baseline so far.
- Concept 4 done: carry-over (K=300, picked on train): val 0.231 / 0.261, best baseline so far (bar to beat).
- Concept 5 done: bootstrap by QB (`bootstrap_compare()`): shrinkage beats league average in 98% of fake seasons, carry-over beats shrinkage in 90%.
Phase 3 done. **Bar to beat: carry-over, val weighted MAE 0.231.**
Phase 4 started in notebooks/03_features_ridge.ipynb.
- Concept 1 done: linear regression on raw ingredients (`base_features()`, `fit_linear()`): val 0.2309, ties carry-over (win 54.5%).
  Fixed weights can't trust this season more as dropbacks pile up.
- Concept 2 done: pre-shrunk features (`shrunk_features()`): val 0.2328, *worse* than carry-over (win 5%). Dropped.
  Fixed weight on last season never fades as this season piles up.
- Hard keep rule added (DECISIONS.md): bootstrap win >= 80% AND train gain in the same direction.
- Concept 3 done: handoff pieces (`handoff_features()`): val 0.2322, win 13%. Dropped. Reshuffling this/last-season EPA has run out.
- Concept 4 done (v1): opponent defense (`defense_features()`), dropped on 2024 alone.
- **Protocol v2** (DECISIONS.md): walk-forward folds 2019-2024, residual on carry-over, keep = pooled win >= 80% AND >= 4/6 seasons.
  `src/models/walk_forward.py`. Re-run of C1-C4: C1 kept, C2/C3 dropped, C4 kept.
- Data fix: 2015 loaded as history only (features, never a target row). v2 table re-run, same conclusions.
  **Current best: carry-over + raw features + opponent defense, pooled walk-forward MAE 0.2259** (carry-over 0.2287).
  Reproduce: last section of notebooks/03_features_ridge.ipynb.
- Walk-forward table now starts with the naive floor: league average (per-fold, earlier seasons only) = 0.2367 pooled.
  Carry-over beats it in 6/6 seasons; the current best beats it by 0.0108.
- Concept 5 done: CPOE (`cpoe_features()`), pooled 0.2260 vs 0.2259, win 35%, 3/6. Dropped: it mostly repeats
  carry-over (corr 0.72), and its weight comes out negative. **Current best still C4, 0.2259.**
- Concept 6 done: sack rate (`sack_features()`), pooled 0.2256 vs 0.2259, win 93.2%, 4/6. **Kept: current best.**
  Weight is positive: carry-over over-punishes sacks, which repeat less than the rest of EPA.
- Concept 7 done: home/away (`home_features()`), win 17.6%, dropped. Raw home edge is only +0.002 EPA/dropback.
- Final Ridge (`make_ridge()`, alpha per fold on season S−1): 0.2257 vs 0.2256, win 17.9%. Dropped: ties plain linear.
Phase 4 done. **Final Phase 4 model: C6 plain linear, pooled walk-forward MAE 0.2256.**
Phase 5 started in notebooks/04_lightgbm.ipynb.
- Concept 1 done: LightGBM (`make_lgbm()`), pooled 0.2254 vs C6 0.2256, win 67.6%, 3/6. Dropped (a tie, as Sid predicted).
  Gain only on low-history QBs (+.0010). Folds picked the smallest grid setting; (4,100) fixed would pass (win 80.8%), but
  that's hindsight, not eligible.
- Smaller LightGBM grid skipped (Sid: it would be shaped by results already seen). **Final model: C6 plain linear.**
- Concept 2 done: SHAP (`explain()` in `src/models/final.py`), weight × (value − training mean) on top of carry-over.
  Matches `shap.LinearExplainer` to 2e-17 on all 2024 games.
- Final model spec frozen in DECISIONS.md (own commit, before 2025 was computed).
- **2025 test (run once): final 0.2382 vs carry-over 0.2431 (win 99.3%) vs league average 0.2491 (win 99.8%).**
  The gain is almost all on low-history QBs (0.2740 → 0.2559); veterans are a wash. Reproduce: `.venv/bin/python -m src.models.final_test`.
  2025 is now used up as a test set.
Phase 5 done.
Phases 6-8 plan approved (DECISIONS.md, "Phases 6-8").
- [x] Weekly data build: `.venv/bin/python -m src.pipeline.build_site_data [--refresh]` → site_data/ (meta, ratings, experiments).
- [x] FastAPI backend: `.venv/bin/uvicorn api.index:app --reload` → http://127.0.0.1:8000/api/docs. Setup: `pip install -r requirements-ml.txt`.
- [x] React frontend: `cd frontend && npm install && npm run dev` (with uvicorn running) → http://localhost:5173.
- [x] vercel.json + GitHub Actions (weekly.yml: Tuesdays 14:00 UTC + manual; ci.yml on push). Live once Sid imports the repo on Vercel.
- [x] README (results, method, limitations, data credit). Resume bullets given in chat.
- [x] Live on Vercel: https://nfl-player-ratings.vercel.app (link in README).

Frontend redesign (approved 2026-10-20), one commit per page:
- [x] Site data + API: full names, headshots, team colors, raw EPA, previous rank, /api/test-2025.
- [x] Rankings (QB landscape scatter + week slider, top-3 cards, sortable table), dark theme + light toggle, Vercel Analytics.
- [x] Player, Players, Method (rating explanation fixed), Model Lab cards.
- [ ] Model Lab calibration plot (2025, 10 bins): explain + Sid's prediction first (CLAUDE.md, ML work).
- [ ] Sid: `gh auth login`, then: repo About link, one manual run of the weekly workflow (confirm commit + redeploy).
- [ ] Sid: enable Web Analytics in the Vercel project (Analytics tab).

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
3. [x] **Baseline + evaluation**: simple baselines (e.g. season-to-date average), MAE on 2024 val.
4. [x] **Features + Ridge**: engineered features (no leakage), Ridge regression.
5. [x] **LightGBM + SHAP**: gradient-boosted model; SHAP to explain predictions.
6. [x] **FastAPI backend**: serve ratings, player history, experiments (api/).
7. [x] **React frontend**: weekly rankings, player pages, Model Lab page showing every
   experiment and its accuracy (frontend/).
8. [ ] **Deploy + weekly auto-update**: free hosting; scheduled job to refresh data and ratings.
