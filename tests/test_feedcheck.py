"""v0.4 F1 (frozen prices) + F2 (wrong-day stamps). Synthetic fixtures only."""
from __future__ import annotations

import datetime as dt
import random

import polars as pl

from split_adjustment_tool import scan_date_shifts, scan_stale_runs

DAYS = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(260)]


def _walk(seed=1, n=260, vol=0.015, start=50.0):
    rng = random.Random(seed)
    p, out = start, []
    for _ in range(n):
        p *= 1 + rng.gauss(0, vol)
        out.append(round(p, 4))
    return out


def _fn(d):
    return lambda s: (None if s not in d else pl.DataFrame({"date": DAYS[:len(d[s])], "close": d[s]}))


def _freeze(closes, start, n):
    c = list(closes)
    for i in range(start, start + n):
        c[i] = c[start - 1]
    return c


def test_frozen_runs_of_3_5_10_bars_are_flagged_with_start_and_end_as_stale_feed():
    true = _walk(1)
    for n in (3, 5, 10):
        bad = _freeze(true, 100, n)
        r = scan_stale_runs(_fn({"TSTS": bad}), ["TSTS"], witness_series_fn=_fn({"TSTS": true}))
        assert r.height == 1
        assert (r["start"][0], r["end"][0]) == (DAYS[99], DAYS[99 + n])   # the last real bar repeated n times
        assert r["classification"][0] == "STALE_FEED"


def test_illiquid_name_flat_on_both_feeds_is_illiquid_not_stale():
    true = _freeze(_walk(2), 120, 6)                 # the stock genuinely did not trade for 6 sessions
    r = scan_stale_runs(_fn({"TSTI": true}), ["TSTI"], witness_series_fn=_fn({"TSTI": true}))
    assert r.height == 1 and r["classification"][0] == "ILLIQUID"


def test_no_witness_means_unverified():
    r = scan_stale_runs(_fn({"TSTU": _freeze(_walk(3), 50, 4)}), ["TSTU"])
    assert r["classification"].to_list() == ["UNVERIFIED"]


def test_a_name_that_often_sits_flat_has_a_higher_bar():
    """Many 3-bar flat stretches are this name's normal; only a run longer than its own norm is reported."""
    c = _walk(4, n=260)
    for k in range(10, 250, 10):
        c[k + 1] = c[k + 2] = c[k]                   # 24 flat 3-bar stretches
    c = _freeze(c, 200, 8)
    r = scan_stale_runs(_fn({"TSTN": c}), ["TSTN"])
    assert r["n_bars"].to_list() == [9]


def test_clean_series_has_no_stale_runs():
    assert scan_stale_runs(_fn({"TSTC": _walk(5)}), ["TSTC"]).height == 0


def _shift_late(closes, start, end):
    """Bars in [start, end) carry the PREVIOUS session's close (stamped one session late)."""
    c = list(closes)
    for i in range(start, end):
        c[i] = closes[i - 1]
    return c


def test_one_day_late_stamps_are_found_with_span_and_direction():
    true = _walk(6)
    bad = _shift_late(true, 100, 160)
    r = scan_date_shifts(_fn({"TSTD": bad}), _fn({"TSTD": true}), ["TSTD"])
    assert r.height >= 1 and set(r["shift"].to_list()) == {1}
    assert r["start"].min() <= DAYS[110] and r["end"].max() >= DAYS[150]
    assert r["start"].min() >= DAYS[80] and r["end"].max() <= DAYS[185]


def test_one_day_early_stamps_are_found_as_shift_minus_one():
    true = _walk(7)
    bad = list(true)
    for i in range(100, 160):
        bad[i] = true[i + 1]
    r = scan_date_shifts(_fn({"TSTE": bad}), _fn({"TSTE": true}), ["TSTE"])
    assert r.height >= 1 and set(r["shift"].to_list()) == {-1}


def test_clean_vendors_with_small_noise_are_not_shifted():
    rng = random.Random(9)
    true = _walk(8)
    noisy = [round(c * (1 + rng.gauss(0, 0.0005)), 4) for c in true]   # a second vendor, 5 bps noise
    assert scan_date_shifts(_fn({"TSTK": noisy}), _fn({"TSTK": true}), ["TSTK"]).height == 0
