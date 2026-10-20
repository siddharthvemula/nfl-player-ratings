# NFL Player Ratings (ML learning project)

## Goal
Weekly QB ratings where a rating = model's prediction of next-week EPA per dropback.
Accuracy = prediction error on held-out seasons. Primary purpose: Sid LEARNS ML.

## Division of labor (strict)
- YOU implement: data loading/caching (src/data/), website (web/), automation, tests.
- SID decides: anything in src/features/ and src/models/. For these, explain the
  concept, propose options with tradeoffs, and ask Sid to predict the result.
  Do NOT write feature/model code until Sid approves the approach, then keep it
  minimal and explain each line in comments.

## Rules
- Data: nflreadpy (returns Polars). Cache raw data to data/raw/ as parquet; never commit data/.
- Time-based splits only: train 2016-2023, val 2024, test 2025. Features for week N
  may only use data from weeks < N. Flag any leakage risk loudly.
- Every experiment gets a row in experiments.md: change, hypothesis, val MAE, keep/drop.
- Sid is new to Git/tooling: explain commands in plain language.
- Commit and push after each completed step once tests pass, one experiment per commit.