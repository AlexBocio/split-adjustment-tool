"""Unit tests for split_adjustment_tool.providers -- synthetic fixtures only, using pytest's tmp_path
for the CSV/Parquet round-trip tests (no real ticker symbols, no network)."""
from __future__ import annotations

import datetime as dt

import polars as pl

from split_adjustment_tool.providers import (
    BAR_SCHEMA,
    CSVActionSource,
    CSVBarProvider,
    InMemoryActionSource,
    InMemoryBarProvider,
    InMemoryTruthProvider,
    ParquetActionSource,
    ParquetBarProvider,
)

_DATES = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(5)]


def _bar_df() -> pl.DataFrame:
    return pl.DataFrame({
        "date": _DATES, "open": [100.0, 101, 102, 101, 103], "high": [101.0, 102, 103, 102, 104],
        "low": [99.0, 100, 101, 100, 102], "close": [100.5, 101.5, 102.5, 101.5, 103.5],
        "volume": [1000, 1100, 900, 1200, 1050],
    }).with_columns(pl.col("date").cast(pl.Date))


def _action_df(symbol: str) -> pl.DataFrame:
    return pl.DataFrame({
        "symbol": [symbol], "date": [dt.date(2022, 6, 1)], "ratio_from": [1], "ratio_to": [4],
        "source": ["feed_a"], "type": ["split"],
    }).with_columns(pl.col("date").cast(pl.Date))


def test_in_memory_bar_provider_round_trip():
    provider = InMemoryBarProvider({"TEST1": _bar_df()})
    assert provider.symbols() == ["TEST1"]
    got = provider.get_bars("TEST1")
    assert got.height == 5
    missing = provider.get_bars("UNKNOWN")
    assert missing.height == 0
    assert set(missing.columns) == set(BAR_SCHEMA)


def test_in_memory_action_source_round_trip():
    provider = InMemoryActionSource(_action_df("TEST2"))
    got = provider.get_actions()
    assert got.height == 1
    assert got["symbol"][0] == "TEST2"


def test_in_memory_truth_provider_round_trip():
    provider = InMemoryTruthProvider({("TEST3", dt.date(2022, 1, 1)): {"high": 110.0, "low": 90.0, "n_bars": 300}})
    assert provider.get_truth("TEST3", dt.date(2022, 1, 1)) == {"high": 110.0, "low": 90.0, "n_bars": 300}
    assert provider.get_truth("TEST3", dt.date(2022, 1, 2)) is None


def test_csv_bar_provider_round_trip(tmp_path):
    directory = tmp_path / "bars"
    directory.mkdir()
    _bar_df().write_csv(directory / "TEST4.csv")
    provider = CSVBarProvider(directory)
    assert provider.symbols() == ["TEST4"]
    got = provider.get_bars("TEST4")
    assert got.height == 5
    assert got["date"][0] == _DATES[0]
    assert provider.get_bars("UNKNOWN").height == 0


def test_csv_action_source_round_trip(tmp_path):
    path = tmp_path / "actions.csv"
    _action_df("TEST5").write_csv(path)
    got = CSVActionSource(path).get_actions()
    assert got.height == 1
    assert got["symbol"][0] == "TEST5"
    assert CSVActionSource(tmp_path / "missing.csv").get_actions().height == 0


def test_csv_action_source_fills_optional_columns(tmp_path):
    path = tmp_path / "actions_minimal.csv"
    pl.DataFrame({
        "symbol": ["TEST6"], "date": [dt.date(2022, 6, 1)], "ratio_from": [1], "ratio_to": [2],
    }).write_csv(path)
    got = CSVActionSource(path).get_actions()
    assert "source" in got.columns and "type" in got.columns
    assert got["source"][0] is None


def test_parquet_bar_provider_hive_partitioned(tmp_path):
    root = tmp_path / "bars_hive"
    sym_dir = root / "symbol=TEST7"
    sym_dir.mkdir(parents=True)
    _bar_df().write_parquet(sym_dir / "data.parquet")
    provider = ParquetBarProvider(root)
    assert provider.symbols() == ["TEST7"]
    got = provider.get_bars("TEST7")
    assert got.height == 5
    assert provider.get_bars("UNKNOWN").height == 0


def test_parquet_bar_provider_single_file(tmp_path):
    path = tmp_path / "all_bars.parquet"
    combined = _bar_df().with_columns(pl.lit("TEST8").alias("symbol"))
    combined.write_parquet(path)
    provider = ParquetBarProvider(path)
    assert provider.symbols() == ["TEST8"]
    got = provider.get_bars("TEST8")
    assert got.height == 5
    assert "symbol" not in got.columns


def test_parquet_action_source_round_trip(tmp_path):
    path = tmp_path / "actions.parquet"
    _action_df("TEST9").write_parquet(path)
    got = ParquetActionSource(path).get_actions()
    assert got.height == 1
    assert ParquetActionSource(tmp_path / "missing.parquet").get_actions().height == 0
