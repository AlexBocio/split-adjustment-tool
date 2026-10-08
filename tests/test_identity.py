"""v0.4 I1: permanent security IDs (ticker changes, reused tickers). Synthetic fixtures only."""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from split_adjustment_tool import (
    SecurityBarProvider,
    apply_split_adjustment,
    collapse_all,
    make_close_series_fn,
    make_gap_fn,
    rekey_by_security,
    symbol_history,
)
from split_adjustment_tool.providers import ACTION_SCHEMA, InMemoryBarProvider

T0 = dt.date(2024, 1, 1)


def _days(n, start=T0):
    return [start + dt.timedelta(days=i) for i in range(n)]


def _bars(dates, closes):
    return pl.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes,
                         "volume": [1000] * len(dates)})


def _reuse_world():
    """Ticker TSTR: OLDCO trades it for 100 days (1-for-10 reverse split on day 50), then is delisted;
    NEWCO, an unrelated company, lists under TSTR on day 130 and never splits."""
    d = _days(200)
    old = [5.0] * 50 + [50.0] * 50
    new = [20.0] * 70
    bars = InMemoryBarProvider({"TSTR": _bars(d[:100] + d[130:], old + new)})
    hist = symbol_history([("OLDCO", "TSTR", d[0], d[99]), ("NEWCO", "TSTR", d[130], None)])
    claims = pl.DataFrame([("TSTR", d[50], 10.0, 1.0, "v", "reverse_split")], schema=ACTION_SCHEMA, orient="row")
    return d, bars, hist, claims


def test_reused_ticker_old_companys_split_is_not_applied_to_the_new_company():
    d, bars, hist, claims = _reuse_world()
    keyed = rekey_by_security(claims, hist)
    assert keyed["symbol"].to_list() == ["OLDCO"] and keyed["ticker"].to_list() == ["TSTR"]
    sbars = SecurityBarProvider(bars, hist)
    out, _ = collapse_all(keyed, gap_fn=make_gap_fn(series_fn=make_close_series_fn(sbars)))
    assert out["symbol"].to_list() == ["OLDCO"]                      # confirmed on OLDCO's own tape
    new_adj = apply_split_adjustment(sbars.get_bars("NEWCO").with_columns(pl.lit("NEWCO").alias("symbol")), out)
    assert new_adj["adj_close"].to_list() == [20.0] * 70              # NEWCO untouched
    old_adj = apply_split_adjustment(sbars.get_bars("OLDCO").with_columns(pl.lit("OLDCO").alias("symbol")), out)
    assert set(old_adj["adj_close"].to_list()) == {50.0}              # OLDCO continuous across its split


def test_without_identity_a_later_companys_split_rewrites_the_earlier_companys_prices():
    """The failure this module exists for, measured: keyed on the ticker, NEWX's 2-for-1 split reaches back
    across the hand-over and halves OLDX's prices (8 -> 4); keyed on the security, OLDX is untouched."""
    d = _days(200)
    raw = _bars(d[:100] + d[130:], [8.0] * 100 + [40.0] * 35 + [20.0] * 35)
    claims = pl.DataFrame([("TSTX", d[165], 1.0, 2.0, "v", "split")], schema=ACTION_SCHEMA, orient="row")
    by_ticker = apply_split_adjustment(raw.with_columns(pl.lit("TSTX").alias("symbol")), claims)
    assert set(by_ticker.filter(pl.col("date") <= d[99])["adj_close"].to_list()) == {4.0}


def test_renamed_security_is_one_continuous_history():
    d = _days(100)
    bars = InMemoryBarProvider({"OLDT": _bars(d[:60], [10.0] * 60), "NEWT": _bars(d[60:], [10.0] * 40)})
    hist = symbol_history([("SEC1", "OLDT", d[0], d[59]), ("SEC1", "NEWT", d[60], None)])
    b = SecurityBarProvider(bars, hist).get_bars("SEC1")
    assert b.height == 100 and b["date"].to_list() == d


def test_new_companys_split_does_not_reach_back_into_the_old_company():
    d = _days(200)
    bars = InMemoryBarProvider({"TSTX": _bars(d[:100] + d[130:], [8.0] * 100 + [40.0] * 35 + [20.0] * 35)})
    hist = symbol_history([("OLDX", "TSTX", d[0], d[99]), ("NEWX", "TSTX", d[130], None)])
    claims = pl.DataFrame([("TSTX", d[165], 1.0, 2.0, "v", "split")], schema=ACTION_SCHEMA, orient="row")
    keyed = rekey_by_security(claims, hist)
    sb = SecurityBarProvider(bars, hist)
    old = apply_split_adjustment(sb.get_bars("OLDX").with_columns(pl.lit("OLDX").alias("symbol")), keyed)
    assert set(old["adj_close"].to_list()) == {8.0}                   # keyed on ticker it would read 4.0
    new = apply_split_adjustment(sb.get_bars("NEWX").with_columns(pl.lit("NEWX").alias("symbol")), keyed)
    assert new["adj_close"].to_list() == [20.0] * 70                  # NEWX's own split still applied


def test_unmatched_rows_keep_their_ticker_by_default_and_can_be_dropped_or_refused():
    d = _days(10)
    hist = symbol_history([("SECA", "TSTA", d[0], d[4])])
    df = pl.DataFrame({"symbol": ["TSTA", "TSTA", "TSTB"], "date": [d[1], d[8], d[1]]})
    assert rekey_by_security(df, hist)["symbol"].to_list() == ["SECA", "TSTA", "TSTB"]
    assert rekey_by_security(df, hist, unmatched="drop")["symbol"].to_list() == ["SECA"]
    with pytest.raises(ValueError):
        rekey_by_security(df, hist, unmatched="raise")


def test_overlapping_ownership_of_one_ticker_is_refused():
    d = _days(10)
    with pytest.raises(ValueError):
        symbol_history([("S1", "TSTO", d[0], d[5]), ("S2", "TSTO", d[5], None)])
    with pytest.raises(ValueError):
        symbol_history([("S1", "TSTO", d[5], d[0])])
