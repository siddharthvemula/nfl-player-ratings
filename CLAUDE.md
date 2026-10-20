# NFL Player Ratings (full-stack ML learning project)

## Goal
Weekly QB ratings where a rating = model's prediction of next-week EPA per dropback.
Accuracy = prediction error on held-out seasons. Primary purpose: Sid LEARNS ML.
Shipped as a typical ML web app: FastAPI backend + React frontend. No paid hosting.

## Layout
- src/data/      data loading + parquet cache
- src/features/  feature engineering
- src/models/    training + evaluation
- api/           FastAPI backend
- frontend/      React frontend
- notebooks/     exploration
- tests/         pytest

## How to work with Sid
You build everything, including the ML code. How you do it depends on the kind of work:

- ML work (features, target, models, evaluation). Before writing code:
  1. Explain the concept in plain language with a concrete analogy.
  2. Give a small worked example using real QB numbers from the data.
  3. Ask Sid to predict the result, and wait for the answer.
  4. Build it, with comments explaining each step.
  5. Report the result in chat. No quizzes.
- Notebooks: keep a Play cell per concept for optional hands-on use, but never require Sid
  to open a notebook. Everything he needs (numbers, tables, results) goes in the chat reply.
- Non-ML work (backend, frontend, deployment, automation): just build it and give a
  one-line summary.
- Phases 6–8 (backend, frontend, deploy): propose one combined plan for Sid's approval, then
  build, test, and commit end to end. Write the README and resume bullets along the way.
- Sid is new to Git/tooling: explain commands in plain language.

## Rules
- Data: nflreadpy (returns Polars). Cache raw data to data/raw/ as parquet; never commit /data/.
- Time-based splits only: train 2016-2023, val 2024, test 2025. Features for week N
  may only use data from weeks < N. Flag any leakage risk loudly.
- Every experiment gets a row in experiments.md: change, hypothesis, val MAE, keep/drop.
- DECISIONS.md: record why each important choice was made (especially ML choices).
- NEXT_STEPS.md: keep the phase plan current and note where we left off at the end of each step.
- After each completed step, run the tests and, if they pass, commit and push without asking.
  One experiment per commit.
