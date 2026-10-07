"""Unit tests for split_adjustment_tool.locator.find_unique_boundary -- synthetic fixtures only."""
from __future__ import annotations

import datetime as dt
import math

import polars as pl

from split_adjustment_tool.locator import find_unique_boundary


def _series(prices: list[float], start: dt.date = dt.date(2022, 1, 3)) -> pl.DataFrame:
    dates = [start + dt.timedelta(days=i) for i in range(len(prices))]
    return pl.DataFrame({"date": dates, "close": prices}).with_columns(pl.col("date").cast(pl.Date))


def test_finds_clean_unique_split_boundary():
    prices = [100.0] * 10 + [25.0] * 10  # 1:4 forward split at index 10 (ratio 0.25)
    s = _series(prices)
    center = s["date"][10]
    hit = find_unique_boundary(s, center, implied_gap=0.25)
    assert hit is not None
    assert hit["tape_date"] == center
    assert abs(hit["measured_1d"] - 0.25) < 1e-6


def test_no_candidate_returns_none():
    prices = [100.0] * 20  # flat, no boundary anywhere
    s = _series(prices)
    center = s["date"][10]
    assert find_unique_boundary(s, center, implied_gap=0.25) is None


def test_ambiguous_multiple_candidates_returns_none():
    # two independent 0.25-ratio drops, both inside one wide search window
    prices = [100.0] * 5 + [25.0] * 5 + [100.0] * 5 + [25.0] * 5
    s = _series(prices)
    center = s["date"][12]
    hit = find_unique_boundary(s, center, implied_gap=0.25, back_days=15, fwd_days=7)
    assert hit is None


def test_reversal_rejected_by_persistence_gate():
    # day10: 100 -> 25 (looks like a split); day11 reverses most of the way back -- a lone
    # bad print, not a real boundary.
    prices = [100.0] * 10 + [25.0, 95.0] + [96.0] * 8
    s = _series(prices)
    center = s["date"][10]
    assert find_unique_boundary(s, center, implied_gap=0.25) is None


def test_tolerance_respected():
    prices = [100.0] * 10 + [26.0] * 10  # ratio 0.26, close to but not exactly 0.25
    s = _series(prices)
    center = s["date"][10]
    # loose tolerance: matches
    hit_loose = find_unique_boundary(s, center, implied_gap=0.25, tol_ln=math.log(1.35))
    assert hit_loose is not None
    # tight tolerance: rejects
    hit_tight = find_unique_boundary(s, center, implied_gap=0.25, tol_ln=math.log(1.02))
    assert hit_tight is None


def test_none_on_short_or_invalid_input():
    assert find_unique_boundary(None, dt.date(2022, 1, 1), 0.25) is None
    assert find_unique_boundary(_series([1.0, 2.0]), dt.date(2022, 1, 1), 0.25) is None
    s = _series([100.0] * 10)
    assert find_unique_boundary(s, s["date"][5], implied_gap=0.0) is None
    assert find_unique_boundary(s, s["date"][5], implied_gap=-1.0) is None
