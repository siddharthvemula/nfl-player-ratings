"""Download NFL play-by-play data with nflreadpy and cache it locally as parquet.

Each season is stored as its own file (data/raw/pbp_<season>.parquet), so a
season is only downloaded once. The in-progress season can be re-downloaded
with refresh_current=True to pick up newly played weeks.

Run from the project root to download everything:
    .venv/bin/python -m src.data.load
"""

from pathlib import Path

import nflreadpy
import polars as pl

FIRST_SEASON = 2016
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _cache_path(season: int) -> Path:
    return RAW_DIR / f"pbp_{season}.parquet"


def load_pbp(seasons: list[int] | None = None, refresh_current: bool = False) -> pl.DataFrame:
    """Return play-by-play data for the given seasons (default: 2016 through current).

    Seasons already cached in data/raw/ are read from disk; missing ones are
    downloaded and saved. If refresh_current is True, the current season is
    re-downloaded even if cached.
    """
    current = nflreadpy.get_current_season()
    if seasons is None:
        seasons = list(range(FIRST_SEASON, current + 1))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = _cache_path(season)
        if not path.exists() or (refresh_current and season == current):
            print(f"Downloading {season}...")
            nflreadpy.load_pbp(season).write_parquet(path)
        frames.append(pl.read_parquet(path))

    # Column types can differ slightly between seasons; relax them so they stack.
    return pl.concat(frames, how="diagonal_relaxed")


if __name__ == "__main__":
    df = load_pbp()
    print(f"Loaded {df.height:,} plays across seasons {df['season'].min()}-{df['season'].max()}")
