"""Unit tests for split_adjustment_tool.breaks -- synthetic fixtures only."""
from __future__ import annotations

import datetime as dt
import random

import polars as pl

from split_adjustment_tool.breaks import (
    check_claims_against_witness,
    classify_jumps,
    scan_unexplained_jumps,
)

_DAYS = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(200)]


def _walk(seed=1, n=200, vol=0.01, start=50.0):
    rng = random.Random(seed)
    p, out = start, []
    for _ in range(n):
        p *= 1 + rng.gauss(0, vol)
        out.append(p)
    return out


def _fn(series_by_symbol):
    def f(sym):
        c = series_by_symbol.get(sym)
        return None if c is None else pl.DataFrame({"date": _DAYS[:len(c)], "close": c})
    return f


def _claims(rows):
    return pl.DataFrame(rows, schema={"symbol": pl.String, "date": pl.Date, "ratio_from": pl.Int64,
                                      "ratio_to": pl.Int64}, orient="row")


def test_provider_break_from_day_x_is_flagged_with_its_start_date():
    true = _walk()
    bad = [c * 10 if i >= 120 else c for i, c in enumerate(true)]
    j = scan_unexplained_jumps(_fn({"TESTPB": bad}), ["TESTPB"])
    assert j.height == 1 and j["date"][0] == _DAYS[120] and j["reverted_on"][0] is None
    c = classify_jumps(j, _fn({"TESTPB": true}))
    assert c["classification"][0] == "PROVIDER_BREAK"


def test_provider_break_that_reverts_gives_a_bad_data_window():
    true = _walk(2)
    bad = [c * 10 if 120 <= i < 150 else c for i, c in enumerate(true)]
    j = scan_unexplained_jumps(_fn({"TESTPW": bad}), ["TESTPW"])
    assert j.height == 1
    assert (j["date"][0], j["reverted_on"][0]) == (_DAYS[120], _DAYS[150])


def test_real_split_missing_from_claims_is_real_move():
    real = [c / 10 if i >= 100 else c for i, c in enumerate(_walk(3))]
    j = scan_unexplained_jumps(_fn({"TESTMC": real}), ["TESTMC"])
    c = classify_jumps(j, _fn({"TESTMC": real}))   # witness sees the same real split
    assert c.height == 1 and c["classification"][0] == "REAL_MOVE"


def test_claimed_real_split_is_explained_and_not_reported():
    real = [c / 10 if i >= 100 else c for i, c in enumerate(_walk(4))]
    j = scan_unexplained_jumps(_fn({"TESTEX": real}), ["TESTEX"], _claims([("TESTEX", _DAYS[100], 1, 10)]))
    assert j.height == 0


def test_no_witness_means_unexplained_not_acted_on():
    bad = [c * 10 if i >= 120 else c for i, c in enumerate(_walk(5))]
    c = classify_jumps(scan_unexplained_jumps(_fn({"TESTNW": bad}), ["TESTNW"]), None)
    assert c["classification"].to_list() == ["UNEXPLAINED"]


def test_bogus_claim_aligned_with_provider_break_is_rejected_by_witness():
    true = _walk(6)
    claims = _claims([("TESTBC", _DAYS[120], 10, 1)])          # bogus 10:1 reverse split
    kept, st = check_claims_against_witness(claims, _fn({"TESTBC": true}))
    assert kept.height == 0 and len(st["witness_contradicted"]) == 1


def test_real_claim_confirmed_by_witness_is_kept():
    real = [c / 10 if i >= 100 else c for i, c in enumerate(_walk(7))]
    kept, st = check_claims_against_witness(_claims([("TESTRC", _DAYS[100], 1, 10)]), _fn({"TESTRC": real}))
    assert kept.height == 1 and len(st["witness_confirmed"]) == 1


def test_volatile_but_clean_series_raises_no_jumps():
    for seed in range(20):
        clean = _walk(seed, vol=0.05)   # a volatile small cap, no breaks
        assert scan_unexplained_jumps(_fn({"TESTVC": clean}), ["TESTVC"]).height == 0


def test_one_day_window_is_reported_and_classified_with_witness():
    true = _walk(8)
    s = list(true)
    s[120] *= 10   # one whole day delivered 10x off, next day correct again
    j = scan_unexplained_jumps(_fn({"TESTSP": s}), ["TESTSP"])
    assert j.height == 1 and (j["date"][0], j["reverted_on"][0]) == (_DAYS[120], _DAYS[121])
    assert classify_jumps(j, _fn({"TESTSP": true}))["classification"][0] == "PROVIDER_BREAK"


def test_witness_that_moves_but_disagrees_is_still_provider_break():
    true = _walk(9)
    true = [c * 0.75 if i >= 120 else c for i, c in enumerate(true)]  # real -25% day in the witness
    bad = [c * 10 if i >= 120 else c for i, c in enumerate(true)]
    c = classify_jumps(scan_unexplained_jumps(_fn({"TESTVD": bad}), ["TESTVD"]), _fn({"TESTVD": true}))
    assert c["classification"].to_list() == ["PROVIDER_BREAK"]
