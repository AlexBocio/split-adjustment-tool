"""Pluggable IO for split-adjustment-tool.

split-adjustment-tool never reads a database, a REST API, or a hardcoded file path. Every module that
needs bar data, corporate-action claims, or independent sub-daily "truth" takes a small
object instead -- you decide where your data actually lives (CSV, Parquet, a database, an
in-memory DataFrame, a live API client) and hand split-adjustment-tool an object shaped like one of the
three ``Protocol``\\ s below. Reference implementations for CSV, Parquet, and plain
in-memory DataFrames/dicts are included so most users never need to write their own.

The three roles, and why they're separate:

- :class:`BarProvider` -- your own RAW (as-traded) daily OHLCV bars. This is "the tape":
  the thing D1 says is truth, because a real corporate action shows up in it as a literal
  price discontinuity, immune to any adjustment-factor bug by construction.
- :class:`ActionSource` -- corporate-action CLAIMS (splits, reverse splits, ...) from
  wherever you get them (a vendor feed, a filing parser, a spreadsheet). Claims are
  hypotheses, never applied at face value -- that's the whole point of the collapse chain
  in :mod:`split_adjustment_tool.chain`.
- :class:`TruthProvider` -- optional, independent sub-daily price extremes (e.g. minute
  bars) used by :mod:`split_adjustment_tool.guard` to repair or confirm a daily bar's high/low against
  a source that isn't the daily print itself.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

__all__ = [
    "BarProvider",
    "ActionSource",
    "TruthProvider",
    "InMemoryBarProvider",
    "InMemoryActionSource",
    "InMemoryTruthProvider",
    "CSVBarProvider",
    "CSVActionSource",
    "ParquetBarProvider",
    "ParquetActionSource",
    "BAR_SCHEMA",
    "ACTION_SCHEMA",
]

#: Canonical bar schema every provider must return. `date` ascending, RAW/as-traded units.
BAR_SCHEMA = {
    "date": pl.Date, "open": pl.Float64, "high": pl.Float64,
    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
}

#: Canonical action-claim schema. `ratio_from`/`ratio_to` use conventional old:new share
#: notation -- a 10-for-1 forward split is ratio_from=1, ratio_to=10; a 1-for-10 reverse
#: split is ratio_from=10, ratio_to=1. `source` and `type` are optional metadata (may be
#: null) -- never required for the math, only for source-rank tie-breaking and bookkeeping.
ACTION_SCHEMA = {
    "symbol": pl.String, "date": pl.Date, "ratio_from": pl.Float64,
    "ratio_to": pl.Float64, "source": pl.String, "type": pl.String,
}


@runtime_checkable
class BarProvider(Protocol):
    """Daily OHLCV bars for a symbol, RAW / as-traded -- unadjusted for any corporate
    action. This is the tape D1 refers to."""

    def get_bars(self, symbol: str) -> pl.DataFrame:
        """Returns columns date/open/high/low/close/volume (see :data:`BAR_SCHEMA`), sorted
        by date ascending. Return an EMPTY DataFrame (not None, not an exception) for an
        unknown symbol."""
        ...

    def symbols(self) -> list[str]:
        """All symbols this provider can serve."""
        ...


@runtime_checkable
class ActionSource(Protocol):
    """Corporate-action CLAIMS -- hypotheses to be confirmed or refuted against the tape,
    never applied at face value (see :mod:`split_adjustment_tool.chain`)."""

    def get_actions(self) -> pl.DataFrame:
        """Returns columns symbol/date/ratio_from/ratio_to/source/type (see
        :data:`ACTION_SCHEMA`), one row per claimed action."""
        ...


@runtime_checkable
class TruthProvider(Protocol):
    """Independent sub-daily (e.g. minute-bar) price extremes for a (symbol, date) pair --
    used by :mod:`split_adjustment_tool.guard` to repair or confirm a daily bar's high/low. Optional
    everywhere it's accepted: pass ``None`` and the guard falls back to envelope-only
    repair."""

    def get_truth(self, symbol: str, date: _dt.date) -> dict | None:
        """Returns ``{"high": float, "low": float, "n_bars": int}`` in the SAME price-unit
        system as the bars you're validating (raw or adjusted -- your choice, just be
        consistent across both), or ``None`` if there's no sub-daily coverage for that
        (symbol, date)."""
        ...


# ---------------------------------------------------------------------------------------
# Reference implementations: in-memory, CSV, Parquet.
# ---------------------------------------------------------------------------------------


class InMemoryBarProvider:
    """:class:`BarProvider` backed by a plain ``dict[symbol, DataFrame]`` already in
    memory. The natural glue for tests and for :mod:`split_adjustment_tool.gauntlet`."""

    def __init__(self, bars: dict[str, pl.DataFrame]):
        self._bars = bars

    def get_bars(self, symbol: str) -> pl.DataFrame:
        df = self._bars.get(symbol)
        return df if df is not None else pl.DataFrame(schema=BAR_SCHEMA)

    def symbols(self) -> list[str]:
        return list(self._bars.keys())


class InMemoryActionSource:
    """:class:`ActionSource` backed by a single DataFrame already in memory."""

    def __init__(self, actions: pl.DataFrame):
        self._actions = actions

    def get_actions(self) -> pl.DataFrame:
        return self._actions


class InMemoryTruthProvider:
    """:class:`TruthProvider` backed by a ``dict[(symbol, date), {"high","low","n_bars"}]``."""

    def __init__(self, truth: dict[tuple[str, _dt.date], dict]):
        self._truth = truth

    def get_truth(self, symbol: str, date: _dt.date) -> dict | None:
        return self._truth.get((symbol, date))


class CSVBarProvider:
    """:class:`BarProvider` reading one CSV per symbol from a directory:
    ``<directory>/<symbol>.csv`` with columns ``date,open,high,low,close,volume``."""

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._cache: dict[str, pl.DataFrame] = {}

    def get_bars(self, symbol: str) -> pl.DataFrame:
        if symbol not in self._cache:
            f = self._dir / f"{symbol}.csv"
            if f.exists():
                df = pl.read_csv(f, try_parse_dates=True).with_columns(pl.col("date").cast(pl.Date))
                self._cache[symbol] = df.sort("date")
            else:
                self._cache[symbol] = pl.DataFrame(schema=BAR_SCHEMA)
        return self._cache[symbol]

    def symbols(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(f.stem for f in self._dir.glob("*.csv"))


class ParquetBarProvider:
    """:class:`BarProvider` reading Hive-partitioned parquet:
    ``<root>/symbol=<SYMBOL>/*.parquet`` -- or, if ``root`` points directly at a file, a
    single combined parquet with a ``symbol`` column."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._cache: dict[str, pl.DataFrame] = {}
        self._single_file = self._root.is_file()
        self._single_df: pl.DataFrame | None = None

    def _load_single(self) -> pl.DataFrame:
        if self._single_df is None:
            self._single_df = pl.read_parquet(self._root)
        return self._single_df

    def get_bars(self, symbol: str) -> pl.DataFrame:
        if self._single_file:
            df = self._load_single()
            return df.filter(pl.col("symbol") == symbol).sort("date").drop("symbol")
        if symbol not in self._cache:
            d = self._root / f"symbol={symbol}"
            files = sorted(d.glob("*.parquet")) if d.exists() else []
            if files:
                self._cache[symbol] = pl.concat([pl.read_parquet(f) for f in files]).sort("date")
            else:
                self._cache[symbol] = pl.DataFrame(schema=BAR_SCHEMA)
        return self._cache[symbol]

    def symbols(self) -> list[str]:
        if self._single_file:
            return sorted(self._load_single()["symbol"].unique().to_list())
        if not self._root.exists():
            return []
        return sorted(p.name.split("=", 1)[1] for p in self._root.glob("symbol=*") if p.is_dir())


def _with_optional_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    for col in columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    return df


class CSVActionSource:
    """:class:`ActionSource` reading a single CSV:
    ``symbol,date,ratio_from,ratio_to[,source][,type]`` (``source``/``type`` optional)."""

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def get_actions(self) -> pl.DataFrame:
        if not self._path.exists():
            return pl.DataFrame(schema=ACTION_SCHEMA)
        df = pl.read_csv(self._path, try_parse_dates=True).with_columns(
            pl.col("date").cast(pl.Date),
            # ratios are exact numbers (1:1.0526, 2:3 ...), never truncated to integers
            pl.col("ratio_from").cast(pl.Float64), pl.col("ratio_to").cast(pl.Float64))
        return _with_optional_columns(df, ("source", "type"))


class ParquetActionSource:
    """:class:`ActionSource` reading a single parquet file of all action claims."""

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def get_actions(self) -> pl.DataFrame:
        if not self._path.exists():
            return pl.DataFrame(schema=ACTION_SCHEMA)
        return _with_optional_columns(pl.read_parquet(self._path), ("source", "type"))
