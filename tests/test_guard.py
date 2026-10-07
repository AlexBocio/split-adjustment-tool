"""Unit tests for tapetruth.guard.apply_bad_print_guard -- synthetic fixtures only."""
from __future__ import annotations

import datetime as dt

import polars as pl

from tapetruth.guard import GuardConfig, apply_bad_print_guard
from tapetruth.providers import InMemoryTruthProvider


def _row(symbol, date, o, h, l, c):
    return {"symbol": symbol, "date": date, "open": o, "high": h, "low": l, "close": c}


def _frame(*rows) -> pl.DataFrame:
    return pl.DataFrame(list(rows)).with_columns(pl.col("date").cast(pl.Date))


def test_oob_high_clamped_to_close_without_truth():
    df = _frame(_row("TEST1", dt.date(2022, 1, 1), 100.0, 250.0, 99.0, 100.0))
    out = apply_bad_print_guard(df)
    row = out.row(0, named=True)
    assert row["bad_print_flag"] == 1
    assert row["repair_source"] == "envelope"
    assert row["high"] == 100.0  # clamped to close -- no truth available to preserve it


def test_oob_low_clamped_to_close_without_truth():
    df = _frame(_row("TEST2", dt.date(2022, 1, 1), 100.0, 101.0, 40.0, 100.0))
    out = apply_bad_print_guard(df)
    row = out.row(0, named=True)
    assert row["bad_print_flag"] == 1
    assert row["low"] == 100.0


def test_geometry_violation_repaired():
    # high(95) < max(open,close)=100 -- geometry can't hold
    df = _frame(_row("TEST3", dt.date(2022, 1, 1), 100.0, 95.0, 99.0, 100.0))
    out = apply_bad_print_guard(df)
    row = out.row(0, named=True)
    assert row["bad_print_flag"] == 1


def test_truth_confirmed_wick_preserved_not_deleted():
    # A 2.2x wick that dense truth confirms is a real traded price: kept unchanged and flagged
    # (v0.1.1; earlier versions overwrote it with the 1.8x band edge).
    df = _frame(_row("TEST4", dt.date(2022, 1, 1), 100.0, 220.0, 97.0, 100.0))
    truth = InMemoryTruthProvider({("TEST4", dt.date(2022, 1, 1)): {"high": 220.0, "low": 97.0, "n_bars": 390}})
    out = apply_bad_print_guard(df, truth=truth)
    row = out.row(0, named=True)
    assert row["high"] == 220.0
    assert row["bad_print_flag"] == 0
    assert row["repair_source"] == "confirmed"
    assert row["confirmed_extreme"] == 1


def test_truth_confirmed_3x_wick_kept_not_deleted():
    df = _frame(_row("TEST4B", dt.date(2022, 1, 1), 100.0, 300.0, 97.0, 100.0))
    truth = InMemoryTruthProvider({("TEST4B", dt.date(2022, 1, 1)): {"high": 300.0, "low": 97.0, "n_bars": 390}})
    row = apply_bad_print_guard(df, truth=truth).row(0, named=True)
    assert row["high"] == 300.0 and row["confirmed_extreme"] == 1


def test_dense_exceed_catches_in_band_bad_low():
    df = _frame(_row("TEST5", dt.date(2022, 1, 1), 100.0, 103.0, 75.0, 100.0))
    truth = InMemoryTruthProvider({("TEST5", dt.date(2022, 1, 1)): {"high": 102.0, "low": 98.5, "n_bars": 390}})
    out = apply_bad_print_guard(df, truth=truth)
    row = out.row(0, named=True)
    assert row["bad_print_flag"] == 1
    assert row["repair_source"] == "truth"
    assert abs(row["low"] - 98.5) < 1.0


def test_dense_exceed_requires_dense_coverage_jurisdiction():
    # identical bad print, but SPARSE sub-daily coverage -- outside the dense-exceed
    # jurisdiction, so it must NOT be repaired (a thin session's true range can legitimately
    # exceed a handful of sub-daily prints).
    df = _frame(_row("TEST6", dt.date(2022, 1, 1), 100.0, 103.0, 75.0, 100.0))
    truth = InMemoryTruthProvider({("TEST6", dt.date(2022, 1, 1)): {"high": 102.0, "low": 98.5, "n_bars": 10}})
    out = apply_bad_print_guard(df, truth=truth)
    row = out.row(0, named=True)
    assert row["bad_print_flag"] == 0


def test_range_insane_but_truth_agrees_left_alone():
    df = _frame(_row("TEST7", dt.date(2022, 1, 1), 100.0, 125.0, 58.0, 100.0))  # ratio 2.155
    truth = InMemoryTruthProvider({("TEST7", dt.date(2022, 1, 1)): {"high": 125.0, "low": 58.0, "n_bars": 50}})
    out = apply_bad_print_guard(df, truth=truth)
    row = out.row(0, named=True)
    assert row["bad_print_flag"] == 0


def test_range_insane_and_truth_contradicts_repaired():
    df = _frame(_row("TEST8", dt.date(2022, 1, 1), 100.0, 125.0, 58.0, 100.0))  # ratio 2.155
    truth = InMemoryTruthProvider({("TEST8", dt.date(2022, 1, 1)): {"high": 103.0, "low": 98.0, "n_bars": 50}})
    out = apply_bad_print_guard(df, truth=truth)
    row = out.row(0, named=True)
    assert row["bad_print_flag"] == 1


def test_clean_data_untouched():
    dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    df = pl.DataFrame({
        "symbol": ["TEST9"] * 5, "date": dates,
        "open": [100.0, 101.0, 102.0, 101.0, 103.0], "high": [101.0, 102.0, 103.0, 102.0, 104.0],
        "low": [99.0, 100.0, 101.0, 100.0, 102.0], "close": [100.5, 101.5, 102.5, 101.5, 103.5],
    }).with_columns(pl.col("date").cast(pl.Date))
    out = apply_bad_print_guard(df)
    assert out["bad_print_flag"].sum() == 0
    assert out["repair_source"].null_count() == 5


def test_config_overrides_are_respected():
    # a value that's OOB under default config (band_high=1.8) but IN-band under a wider one
    df = _frame(_row("TEST10", dt.date(2022, 1, 1), 100.0, 190.0, 99.0, 100.0))
    default_out = apply_bad_print_guard(df)
    assert default_out.row(0, named=True)["bad_print_flag"] == 1
    wide_out = apply_bad_print_guard(df, config=GuardConfig(band_high=2.0))
    assert wide_out.row(0, named=True)["bad_print_flag"] == 0


# --- Regression tests for the crash-day fix (body-relative wicks, truth-confirmed opens) ---
# Found on real data: a small-cap that opened at 6.66 and closed at 1.13 (a genuine one-day
# crash, confirmed by sub-daily bars) had its open, high and low flattened because every
# limit was measured against the close. Fixtures below are synthetic (repo rule).

def test_real_crash_day_confirmed_by_truth_is_kept():
    day = dt.date(2022, 3, 1)
    df = _frame(_row("TESTCRASH", day, 6.66, 6.72, 1.00, 1.13))
    truth = InMemoryTruthProvider({("TESTCRASH", day): {"high": 6.75, "low": 1.00, "n_bars": 17}})
    row = apply_bad_print_guard(df, truth=truth).row(0, named=True)
    assert row["bad_print_flag"] == 0
    assert row["repair_source"] == "confirmed"
    assert (row["open"], row["high"], row["low"]) == (6.66, 6.72, 1.00)


def test_crash_day_open_without_truth_still_repaired():
    # With no evidence, an open 5.9x the close is indistinguishable from a bad print.
    day = dt.date(2022, 3, 1)
    df = _frame(_row("TESTCRASH2", day, 6.66, 6.72, 1.00, 1.13))
    row = apply_bad_print_guard(df).row(0, named=True)
    assert row["bad_print_flag"] == 1
    assert row["repair_source"] == "envelope"
    assert row["open"] == 1.13


def test_wicks_judged_against_body_not_close():
    # Big gap-down day: body 10 -> 6; high 10.1 is a normal wick above the body (1.01x),
    # even though it is 1.68x the close. Must be left alone.
    day = dt.date(2022, 3, 2)
    df = _frame(_row("TESTBODY", day, 10.0, 10.1, 5.9, 6.0))
    row = apply_bad_print_guard(df).row(0, named=True)
    assert row["bad_print_flag"] == 0


def test_bad_low_far_below_body_still_repaired_from_truth():
    # A decimal-slip low (69.0 on a ~690 day) sticks out far beyond the body: repaired from truth.
    day = dt.date(2022, 3, 3)
    df = _frame(_row("TESTSLIP", day, 689.6, 696.9, 69.0, 695.4))
    truth = InMemoryTruthProvider({("TESTSLIP", day): {"high": 697.5, "low": 681.9, "n_bars": 17}})
    row = apply_bad_print_guard(df, truth=truth).row(0, named=True)
    assert row["bad_print_flag"] == 1
    assert row["repair_source"] == "truth"
    assert abs(row["low"] - 681.9) < 1e-9


def test_truth_that_does_not_contain_the_close_is_ignored():
    # A truth range that doesn't even contain the day's close is a degraded measurement.
    day = dt.date(2022, 3, 4)
    df = _frame(_row("TESTGARB", day, 100.0, 101.0, 40.0, 100.0))
    truth = InMemoryTruthProvider({("TESTGARB", day): {"high": 60.0, "low": 30.0, "n_bars": 390}})
    row = apply_bad_print_guard(df, truth=truth).row(0, named=True)
    assert row["repair_source"] == "envelope"
    assert row["low"] == 100.0
