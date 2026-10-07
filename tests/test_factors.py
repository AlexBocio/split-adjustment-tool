"""Unit tests for split_adjustment_tool.factors -- synthetic fixtures only."""
from __future__ import annotations

import datetime as dt
import math

import polars as pl

from split_adjustment_tool.factors import apply_split_adjustment, build_factor_table

_D = [dt.date(2023, 1, 2) + dt.timedelta(days=i) for i in range(10)]


def _bars(closes, vols=None, sym="TESTF"):
    vols = vols or [1000] * len(closes)
    return pl.DataFrame({"symbol": [sym] * len(closes), "date": _D[:len(closes)], "open": closes,
                         "high": closes, "low": closes, "close": closes, "volume": vols})


def _acts(rows):
    return pl.DataFrame(rows, schema={"symbol": pl.String, "date": pl.Date,
                                      "ratio_from": pl.Int64, "ratio_to": pl.Int64})


def test_forward_split_is_continuous_and_anchored_at_last_bar():
    # 1-for-10 forward split effective on _D[5]: raw 100 -> 10
    out = apply_split_adjustment(_bars([100.0] * 5 + [10.0] * 5), _acts([("TESTF", _D[5], 1, 10)]))
    assert out["adj_close"].to_list() == [10.0] * 10            # no jump anywhere
    assert out["price_factor"].to_list() == [0.1] * 5 + [1.0] * 5  # ex-date bar not adjusted
    assert out["price_factor"][-1] == 1.0                       # anchored at the last bar


def test_reverse_split_and_dollar_volume_invariant():
    # 1-for-5 reverse split: raw 1.0 -> 5.0, volume 5000 -> 1000
    bars = _bars([1.0] * 5 + [5.0] * 5, [5000] * 5 + [1000] * 5)
    out = apply_split_adjustment(bars, _acts([("TESTF", _D[5], 5, 1)]))
    assert out["adj_close"].to_list() == [5.0] * 10
    assert out["adj_volume"].to_list() == [1000.0] * 10
    for r in out.iter_rows(named=True):
        assert math.isclose(r["close"] * r["volume"], r["adj_close"] * r["adj_volume"])


def test_two_events_compose_hand_computed():
    # 1:2 forward at D3 (price halves), then 3:1 reverse at D7 (price triples)
    closes = [60.0] * 3 + [30.0] * 4 + [90.0] * 3
    out = apply_split_adjustment(_bars(closes), _acts([("TESTF", _D[3], 1, 2), ("TESTF", _D[7], 3, 1)]))
    # before D3: x0.5 then x3 -> 1.5 ; D3..D6: x3 ; from D7: 1
    assert out["price_factor"].to_list() == [1.5] * 3 + [3.0] * 4 + [1.0] * 3
    assert out["adj_close"].to_list() == [90.0] * 10


def test_symbol_without_actions_is_untouched_and_row_order_kept():
    a = _bars([10.0, 11.0], sym="TESTA")
    b = _bars([20.0, 21.0], sym="TESTB")
    bars = pl.concat([b, a])  # deliberately not sorted by symbol
    out = apply_split_adjustment(bars, _acts([("TESTB", _D[1], 1, 2)]))
    assert out["symbol"].to_list() == ["TESTB", "TESTB", "TESTA", "TESTA"]
    assert out.filter(pl.col("symbol") == "TESTA")["price_factor"].to_list() == [1.0, 1.0]


def test_factor_table_sparse_rows():
    t = build_factor_table(_acts([("TESTF", _D[3], 1, 2), ("TESTF", _D[7], 3, 1)]), start_date=_D[0])
    assert t["date"].to_list() == [_D[0], _D[3], _D[7]]
    assert t["price_factor"].to_list() == [1.5, 3.0, 1.0]
    assert [round(v, 9) for v in t["volume_factor"].to_list()] == [round(1 / 1.5, 9), round(1 / 3.0, 9), 1.0]


def test_intraday_bars_use_their_trading_date():
    # minute bars: previous session's after-hours stay pre-split; ex-date pre-market is post-split
    ts = [dt.datetime(2023, 1, 6, 19, 58), dt.datetime(2023, 1, 6, 19, 59),
          dt.datetime(2023, 1, 9, 4, 0), dt.datetime(2023, 1, 9, 4, 1)]
    bars = pl.DataFrame({"symbol": ["TESTI"] * 4, "ts": ts,
                         "date": [t.date() for t in ts],  # exchange-local trading date
                         "close": [200.0, 200.2, 20.03, 20.01], "volume": [10, 10, 100, 100]})
    out = apply_split_adjustment(bars, _acts([("TESTI", dt.date(2023, 1, 9), 1, 10)]))
    assert out["price_factor"].to_list() == [0.1, 0.1, 1.0, 1.0]
    assert [round(v, 3) for v in out["adj_close"].to_list()] == [20.0, 20.02, 20.03, 20.01]
    assert out["ts"].to_list() == ts  # extra columns and row order pass through


# --- time-zone agnostic trading dates ---
def _utc(*a):
    return dt.datetime(*a, tzinfo=dt.UTC)


def _intraday(ts, sym="TESTZ"):
    n = len(ts)
    return pl.DataFrame({"symbol": [sym] * n, "ts": ts, "close": [200.0, 200.0, 20.0, 20.0][:n],
                         "volume": [10] * n})


def test_utc_timestamps_mapped_to_new_york_trading_date():
    # 23:30 UTC Fri = 19:30 ET Fri (pre-split); 00:30 UTC Sat = 20:30 ET Fri (still pre-split, though
    # its UTC date is Saturday); 09:00 UTC Mon = 05:00 ET Mon pre-market (post-split)
    ts = [_utc(2023, 1, 6, 23, 30), _utc(2023, 1, 7, 0, 30), _utc(2023, 1, 9, 9, 0), _utc(2023, 1, 9, 15, 0)]
    out = apply_split_adjustment(_intraday(ts), _acts([("TESTZ", dt.date(2023, 1, 9), 1, 10)]), timestamp_col="ts")
    assert out["price_factor"].to_list() == [0.1, 0.1, 1.0, 1.0]
    assert out["ts"].to_list() == ts


def test_naive_timestamps_read_as_utc_by_default_or_as_given_zone():
    naive = [dt.datetime(2023, 1, 9, 2, 0), dt.datetime(2023, 1, 9, 14, 0)]
    bars = _intraday(naive)
    acts = _acts([("TESTZ", dt.date(2023, 1, 9), 1, 10)])
    # 02:00 UTC Mon = 21:00 ET Sun -> trading date Sunday (before the Monday ex-date)
    assert apply_split_adjustment(bars, acts, timestamp_col="ts")["price_factor"].to_list() == [0.1, 1.0]
    # same naive clock read as New York local time -> both on Monday
    assert apply_split_adjustment(bars, acts, timestamp_col="ts",
                                  naive_timestamps_tz="America/New_York")["price_factor"].to_list() == [1.0, 1.0]


def test_per_symbol_exchange_zone():
    # 23:30 UTC Sun = 08:30 Mon in Tokyo (post-split there) but 18:30 Sun in New York (pre-split)
    t = _utc(2023, 1, 8, 23, 30)
    bars = pl.concat([_intraday([t], "TESTJP"), _intraday([t], "TESTUS")])
    acts = _acts([("TESTJP", dt.date(2023, 1, 9), 1, 10), ("TESTUS", dt.date(2023, 1, 9), 1, 10)])
    out = apply_split_adjustment(bars, acts, timestamp_col="ts", tz_by_symbol={"TESTJP": "Asia/Tokyo"})
    assert dict(zip(out["symbol"].to_list(), out["price_factor"].to_list(), strict=True)) == {"TESTJP": 1.0, "TESTUS": 0.1}
