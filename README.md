# NFL QB Ratings

Weekly NFL quarterback ratings where **a rating is a forecast**: the model's prediction of a QB's EPA per dropback in his
next game. Because every rating is a prediction, it can be checked against what happened. The model was frozen and then
tested once on a full held-out season (2025).

Full-stack ML app: Python data pipeline + model → precomputed JSON → **FastAPI** backend → **React** frontend, deployed
on Vercel's free tier and refreshed every Tuesday by GitHub Actions.

**Live site:** _link added after the first deploy_

## Results: the one-time 2025 test

The final model was frozen (spec committed first) and then scored once on all 624 QB-games of 2025, trained on 2016-2024.
MAE = mean absolute error in EPA per dropback, weighted by dropbacks. Lower is better.

| Model | 2025 MAE | vs carry-over |
|---|---|---|
| League average (same number for every QB) | 0.2491 | |
| Carry-over (QB's own shrunk average, last season → this season) | 0.2431 | |
| **Final model** (carry-over + learned adjustment, 6 features) | **0.2382** | better in 99.3% of bootstrap resamples (gain +0.0049, 95% range +0.0009 to +0.0093) |

Where the gain comes from: QBs with little history (≤ 212 dropbacks last season) went from 0.2740 to 0.2559. For
established starters the model is about the same as carry-over (0.2322 vs 0.2319). Results held up against the
six-season walk-forward estimate (final 0.2256 vs carry-over 0.2287, better in 6 of 6 seasons).

## Method

- **Target:** QB EPA per dropback in a game (passes, sacks and scrambles; `qb_epa`, so receiver fumbles aren't charged
  to the QB). Regular season and playoffs, games with 3+ dropbacks.
- **No leakage:** features for week N use only weeks < N. Every season is predicted by a model trained only on earlier
  seasons (walk-forward), so every rating on the site was made before its game.
- **Model:** carry-over (last season's EPA/dropback shrunk toward the league average, handing off to this season as
  dropbacks pile up), plus a linear adjustment fit on the leftover error, using six features: this-season and
  last-season EPA/dropback and dropbacks, opponent defense (EPA allowed so far, shrunk), and sack rate.
- **Evaluation:** walk-forward folds 2019-2024. A change was kept only if it beat the current best in ≥ 80% of bootstrap
  resamples (QBs resampled) **and** in ≥ 4 of 6 seasons. Every experiment is logged in [experiments.md](experiments.md)
  and shown on the site's Model Lab page. The reasoning behind each choice is in [DECISIONS.md](DECISIONS.md).
- **Explanations:** each rating splits exactly into carry-over + one SHAP piece per feature (weight × (value − training
  average)), verified against `shap.LinearExplainer`.
- **Confidence:** based on dropbacks of evidence (last season + this season). Low/Med/High tiers had walk-forward errors
  of 0.248 / 0.227 / 0.217.

### Tried and dropped

Pre-shrunk features, re-weighting carry-over's pieces, completion % over expected (CPOE), home field, Ridge regression,
and gradient-boosted trees (LightGBM) were all tested and failed the keep rule. LightGBM tied the linear model (0.2254 vs
0.2256, 68% bootstrap win); its only edge was on low-history QBs. Two findings: QBs with little history do worse than
carry-over assumes, and at the same past EPA a QB with a higher sack rate does slightly *better* next week (sacks are
costly but repeat less than the rest of a QB's play).

## Known limitations

- EPA and CPOE come from nflverse's own models, which were fit on seasons that overlap the ones used here. The target
  and every feature share this small leakage risk.
- 2025 has been used as the one-time test set. Any later model change can only be judged by walk-forward on 2019-2024.
- The model knows nothing about injuries, depth charts, weather or coaching changes, and can't know who will start.
- Single games are noisy: even the best forecasts miss by about 0.22 EPA/dropback per game on average.
- The improvement over carry-over is concentrated in rookies and backups.

## Project layout

```
src/data/        nflreadpy download + parquet cache (data/ is not committed)
src/features/    per-game QB table, target, features (no week-N data)
src/models/      baselines, walk-forward, final model + SHAP, one-time 2025 test
src/pipeline/    weekly build: every rating → site_data/*.json
api/             FastAPI app (reads site_data/; no model at request time)
frontend/        React + Vite + TypeScript (rankings, player pages, Model Lab, method)
notebooks/       exploration and one notebook per phase
tests/           pytest
.github/         weekly refresh (Tuesdays) + CI
```

## Run it locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt
.venv/bin/python -m src.data.load                   # download play-by-play (2015 → now), cached as parquet
.venv/bin/python -m src.pipeline.build_site_data    # recompute every rating → site_data/
.venv/bin/python -m pytest                          # tests
.venv/bin/uvicorn api.index:app --reload            # API at http://127.0.0.1:8000/api/docs
cd frontend && npm install && npm run dev           # site at http://localhost:5173
```

Reproduce the final test: `.venv/bin/python -m src.models.final_test` (deterministic; the result is already recorded).

## Data

Data: [nflverse](https://github.com/nflverse) / nflfastR (CC-BY 4.0), loaded with
[nflreadpy](https://github.com/nflverse/nflreadpy). Code: MIT license.
