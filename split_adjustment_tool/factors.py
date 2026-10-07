"""Split adjustment: turn confirmed corporate actions into adjustment factors and adjusted bars.

This is the step the rest of the package exists to make safe. :func:`split_adjustment_tool.chain.collapse_all`
decides WHICH split claims are real (right ratio, right date); this module applies them.

Convention (CRSP-style, documented so there is no ambiguity):

* A claim ``ratio_from:ratio_to`` implies the price moves by ``m = ratio_from / ratio_to`` at the
  open of its ex-date (a 1-for-10 forward split is ``1:10`` -> ``m = 0.1``; a 1-for-5 reverse
  split is ``5:1`` -> ``m = 5``).
* Factors are anchored at the most recent bar: ``price_factor = 1`` on and after the last event.
  For any earlier date, ``price_factor`` is the product of ``m`` over every event whose ex-date is
  AFTER that date. ``adjusted_price = raw_price * price_factor``.
* An event applies FROM its ex-date: the ex-date's own bar already trades post-split, so it is not
  adjusted for that event.
* ``volume_factor = 1 / price_factor`` -- share counts and volume scale the other way, so
  ``raw_price * raw_volume == adjusted_price * adjusted_volume`` on every bar (dollar volume is
  invariant to a split).

Splits and reverse splits only. Cash dividends, spin-offs and rights need a separate price factor
computed from the distributed value and are NOT handled here (see docs/STANDARD.md).
"""
from __future__ import annotations

import polars as pl

__all__ = ["build_factor_table", "apply_split_adjustment"]

_PRICE_COLS = ("open", "high", "low", "close")


def _implied(actions: pl.DataFrame) -> pl.DataFrame:
    return (actions.select("symbol", pl.col("date").cast(pl.Date),
                           (pl.col("ratio_from").cast(pl.Float64) / pl.col("ratio_to").cast(pl.Float64)).alias("m"))
            .filter(pl.col("m") > 0)
            .group_by("symbol", "date").agg(pl.col("m").product())  # same-day events compose
            .sort("symbol", "date"))


def build_factor_table(actions: pl.DataFrame, start_date=None) -> pl.DataFrame:
    """Sparse factor table, one row per event plus an optional starting row:
    ``symbol, date, price_factor, volume_factor`` where each row's factors apply from that date
    until the next row. This is the usual vendor format and what :mod:`split_adjustment_tool.reconcile`
    compares (it as-of aligns sparse tables).

    `actions` should already be cleaned (pass it through :func:`split_adjustment_tool.chain.collapse_all`).
    `start_date`, if given, adds a row at that date carrying the fully-compounded factor for all
    history before the first event.
    """
    ev = _implied(actions)
    out = []
    for sym in ev["symbol"].unique().sort().to_list():
        rows = ev.filter(pl.col("symbol") == sym)
        ms = rows["m"].to_list()
        ds = rows["date"].to_list()
        # factor in force FROM ds[i] = product of m over events strictly after ds[i]
        tail = 1.0
        after = []
        for m in reversed(ms):
            after.append(tail)
            tail *= m
        after.reverse()
        if start_date is not None and ds and start_date < ds[0]:
            out.append({"symbol": sym, "date": start_date, "price_factor": tail})
        for d, f in zip(ds, after, strict=True):
            out.append({"symbol": sym, "date": d, "price_factor": f})
    schema = {"symbol": pl.String, "date": pl.Date, "price_factor": pl.Float64}
    df = pl.DataFrame(out, schema=schema) if out else pl.DataFrame(schema=schema)
    return df.with_columns((1.0 / pl.col("price_factor")).alias("volume_factor"))


def _trading_date(bars: pl.DataFrame, timestamp_col: str | None, exchange_tz: str,
                  tz_by_symbol: dict[str, str] | None, naive_tz: str) -> pl.Expr:
    """Expression giving each row's exchange-local TRADING DATE, whatever the input time zone.

    * `timestamp_col` (or a Datetime-typed ``date`` column) is converted to the symbol's exchange
      time zone and truncated to a date. Naive timestamps are first interpreted as `naive_tz`.
    * A Date-typed ``date`` column is taken as already being the trading date.
    """
    col = timestamp_col if timestamp_col is not None else "date"
    dtype = bars.schema[col]
    if not isinstance(dtype, pl.Datetime):
        return pl.col(col).cast(pl.Date)
    ts = pl.col(col)
    if dtype.time_zone is None:
        ts = ts.dt.replace_time_zone(naive_tz)
    zones = {exchange_tz}
    if tz_by_symbol:
        zones |= set(tz_by_symbol.values())
    if len(zones) == 1:
        return ts.dt.convert_time_zone(exchange_tz).dt.replace_time_zone(None).dt.date()
    zone_of = pl.col("symbol").replace_strict(tz_by_symbol or {}, default=exchange_tz, return_dtype=pl.String)
    expr = None
    for z in sorted(zones):
        part = ts.dt.convert_time_zone(z).dt.replace_time_zone(None).dt.date()
        expr = pl.when(zone_of == z).then(part) if expr is None else expr.when(zone_of == z).then(part)
    return expr.otherwise(None)


def apply_split_adjustment(
    bars: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    timestamp_col: str | None = None,
    exchange_tz: str = "America/New_York",
    tz_by_symbol: dict[str, str] | None = None,
    naive_timestamps_tz: str = "UTC",
) -> pl.DataFrame:
    """Add ``price_factor``, ``volume_factor`` and ``adj_open/adj_high/adj_low/adj_close``
    (+ ``adj_volume`` when a ``volume`` column exists) to `bars`.

    Works on any timeframe and any time zone. A split takes effect at the start of its ex-date in
    the EXCHANGE's local time, so each row is matched by its exchange-local trading date:

    * daily bars: a Date-typed ``date`` column is used as is;
    * intraday bars: pass ``timestamp_col`` (or give ``date`` a Datetime type). Time-zone-aware
      timestamps are converted to the exchange zone; naive ones are read as ``naive_timestamps_tz``
      (UTC by default) first. ``exchange_tz`` defaults to New York; ``tz_by_symbol`` overrides it
      per symbol (e.g. ``{"VOD.L": "Europe/London"}``).

    `bars` needs ``symbol`` and any of ``open/high/low/close`` (raw, as traded); every other column,
    including your own ``date``/timestamp, passes through unchanged. `actions` should already be
    cleaned by :func:`split_adjustment_tool.chain.collapse_all`. Symbols with no actions get factor 1.
    """
    ev = _implied(actions)
    b = bars.with_columns(
        _trading_date(bars, timestamp_col, exchange_tz, tz_by_symbol, naive_timestamps_tz).alias("_tdate")
    ).with_row_index("_ri")
    if ev.height == 0:
        f = b.with_columns(pl.lit(1.0).alias("price_factor"))
    else:
        # price_factor(d) = product of m over events with ex-date > d. Computed by direct
        # multiplication (no log/exp round trip) so simple ratios stay exact.
        ev = ev.with_columns(
            pl.col("m").reverse().cum_prod().reverse().shift(-1).fill_null(1.0).over("symbol").alias("after"),
        )
        totals = ev.group_by("symbol").agg(pl.col("m").product().alias("total"))
        # for bar date d, take the LAST event with ex-date <= d: its `after` is the answer;
        # dates before the first event take the symbol's total.
        j = b.sort("symbol", "_tdate").join_asof(
            ev.select("symbol", pl.col("date").alias("_tdate"), "after").sort("symbol", "_tdate"),
            on="_tdate", by="symbol", strategy="backward", check_sortedness=False,
        ).join(totals, on="symbol", how="left")
        f = j.with_columns(
            pl.when(pl.col("after").is_not_null()).then(pl.col("after"))
              .when(pl.col("total").is_not_null()).then(pl.col("total"))
              .otherwise(1.0).alias("price_factor")
        ).drop("after", "total")
    f = f.with_columns((1.0 / pl.col("price_factor")).alias("volume_factor"))
    adds = [(pl.col(c) * pl.col("price_factor")).alias(f"adj_{c}") for c in _PRICE_COLS if c in f.columns]
    if "volume" in f.columns:
        adds.append((pl.col("volume") * pl.col("volume_factor")).alias("adj_volume"))
    return f.with_columns(adds).sort("_ri").drop("_ri", "_tdate")
