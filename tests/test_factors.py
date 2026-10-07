"""Unit tests for tapetruth.factors -- synthetic fixtures only."""
from __future__ import annotations

import datetime as dt
import math

import polars as pl

from tapetruth.factors import apply_split_adjustment, build_factor_table

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
