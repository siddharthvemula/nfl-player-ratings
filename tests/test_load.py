import polars as pl

from src.data import load


def test_load_pbp_downloads_each_season_only_once(tmp_path, monkeypatch):
    calls = []

    def fake_download(season):
        calls.append(season)
        return pl.DataFrame({"season": [season], "epa": [0.1]})

    monkeypatch.setattr(load, "RAW_DIR", tmp_path)
    monkeypatch.setattr(load.nflreadpy, "load_pbp", fake_download)
    monkeypatch.setattr(load.nflreadpy, "get_current_season", lambda: 2017)

    first = load.load_pbp()
    second = load.load_pbp()

    assert calls == [2016, 2017]
    assert first.equals(second)
    assert (tmp_path / "pbp_2016.parquet").exists()


def test_refresh_current_redownloads_only_current_season(tmp_path, monkeypatch):
    calls = []

    def fake_download(season):
        calls.append(season)
        return pl.DataFrame({"season": [season]})

    monkeypatch.setattr(load, "RAW_DIR", tmp_path)
    monkeypatch.setattr(load.nflreadpy, "load_pbp", fake_download)
    monkeypatch.setattr(load.nflreadpy, "get_current_season", lambda: 2017)

    load.load_pbp()
    load.load_pbp(refresh_current=True)

    assert calls == [2016, 2017, 2017]
