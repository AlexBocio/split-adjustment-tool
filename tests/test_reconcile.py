"""Unit tests for tapetruth.reconcile -- synthetic fixtures only."""
from __future__ import annotations

import datetime as dt

import polars as pl

from tapetruth.reconcile import (
    ReconcileConfig,
    ReconciliationClass,
    reconcile_all,
    reconcile_symbol,
    reconciliation_gate,
)

_DATES = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(10)]


def _series(dates: list[dt.date], factors: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, "factor": factors}).with_columns(pl.col("date").cast(pl.Date))


def test_exact_match():
    ours = _series(_DATES, [1.0] * 5 + [4.0] * 5)
    ref = _series(_DATES, [1.0] * 5 + [4.0] * 5)
    r = reconcile_symbol("TEST1", ours, ref)
    assert r.classification == ReconciliationClass.EXACT


def test_minor_disagreement():
    ours = _series(_DATES, [1.0] * 10)
    ref = _series(_DATES, [1.01] * 10)  # ~1% constant offset -- within minor tolerance
    r = reconcile_symbol("TEST2", ours, ref)
    assert r.classification == ReconciliationClass.MINOR


def test_exact_adjusted_equiv_parallel_offset():
    ours = _series(_DATES, [1.0] * 5 + [4.0] * 5)
    ref = _series(_DATES, [2.0] * 5 + [8.0] * 5)  # exactly 2x ours everywhere -- parallel curves
    r = reconcile_symbol("TEST3", ours, ref)
    assert r.classification == ReconciliationClass.EXACT_ADJUSTED_EQUIV


def test_mismatch():
    ours = _series(_DATES, [1.0] * 5 + [4.0] * 5)
    ref = _series(_DATES, [1.0] * 5 + [10.0] * 5)  # genuinely different event magnitude
    r = reconcile_symbol("TEST4", ours, ref)
    assert r.classification == ReconciliationClass.MISMATCH


def test_explained_via_explain_fn():
    ours = _series(_DATES, [1.0] * 5 + [4.0] * 5)
    ref = _series(_DATES, [1.0] * 5 + [10.0] * 5)
    r = reconcile_symbol("TEST5", ours, ref, explain_fn=lambda s: "known scope difference")
    assert r.classification == ReconciliationClass.EXPLAINED
    assert "known scope difference" in r.detail


def test_not_comparable_missing_side():
    ours = _series(_DATES, [1.0] * 10)
    r = reconcile_symbol("TEST6", ours, None)
    assert r.classification == ReconciliationClass.NOT_COMPARABLE


def test_not_comparable_insufficient_overlap():
    ours = _series(_DATES[:2], [1.0, 1.0])
    ref = _series(_DATES[5:7], [1.0, 1.0])  # no overlapping dates at all
    r = reconcile_symbol("TEST7", ours, ref)
    assert r.classification == ReconciliationClass.NOT_COMPARABLE


def test_reconcile_all_and_gate():
    def ours_fn(symbol):
        return _series(_DATES, [1.0] * 10) if symbol != "MISSING" else None

    def reference_fn(symbol):
        if symbol == "TESTA":
            return _series(_DATES, [1.0] * 10)  # EXACT
        if symbol == "TESTB":
            return _series(_DATES, [1.01] * 10)  # MINOR
        if symbol == "TESTC":
            # NOT a constant multiple of ours (half the series agrees, half doesn't) -- a
            # genuinely non-parallel divergence, unlike a constant offset (which classifies
            # as EXACT_ADJUSTED_EQUIV, not MISMATCH -- see test_exact_adjusted_equiv_parallel_offset)
            return _series(_DATES, [1.0] * 5 + [5.0] * 5)  # MISMATCH
        return None  # MISSING -> NOT_COMPARABLE

    results = reconcile_all(["TESTA", "TESTB", "TESTC", "MISSING"], ours_fn, reference_fn)
    assert results.height == 4

    gate = reconciliation_gate(results, min_pass_rate=0.6)
    assert gate["n_comparable"] == 3
    assert gate["n_passing"] == 2  # EXACT + MINOR
    assert gate["n_mismatch"] == 1
    assert abs(gate["pass_rate"] - (2 / 3)) < 1e-9
    assert gate["gate_passed"] is True


def test_reconciliation_gate_fails_below_threshold():
    results = pl.DataFrame({
        "symbol": ["A", "B"],
        "classification": [ReconciliationClass.EXACT, ReconciliationClass.MISMATCH],
        "detail": ["", ""], "n_dates_compared": [10, 10],
        "median_abs_ln_diff": [0.0, 0.5], "max_abs_ln_diff": [0.0, 0.5],
        "mean_ln_diff": [0.0, 0.5], "std_ln_diff": [0.0, 0.0],
    })
    gate = reconciliation_gate(results, min_pass_rate=0.99)
    assert gate["gate_passed"] is False


def test_config_tolerances_are_respected():
    ours = _series(_DATES, [1.0] * 10)
    ref = _series(_DATES, [1.01] * 10)
    tight = ReconcileConfig(exact_tol_ln=0.5)  # very loose exact tolerance
    r = reconcile_symbol("TEST8", ours, ref, config=tight)
    assert r.classification == ReconciliationClass.EXACT


# --- M2 regressions (specialist findings C1, C2) ---
_LONG = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(100)]


def test_single_wrong_event_on_minority_of_dates_is_mismatch():
    # ours wrong by 2x on the earliest 30% of dates (an extra event) -- a median would hide it
    ours = _series(_LONG, [0.5] * 30 + [1.0] * 70)
    ref = _series(_LONG, [1.0] * 100)
    r = reconcile_symbol("TESTC1", ours, ref)
    assert r.classification == ReconciliationClass.MISMATCH
    assert any(i.startswith("ONLY_OURS") for i in r.event_issues)


def test_five_day_misdate_is_date_differs():
    ours = _series(_LONG, [0.1] * 50 + [1.0] * 50)
    ref = _series(_LONG, [0.1] * 55 + [1.0] * 45)
    r = reconcile_symbol("TESTC1B", ours, ref)
    assert r.classification == ReconciliationClass.DATE_DIFFERS


def test_sparse_event_table_is_comparable_with_dense_series():
    dense = _series(_LONG, [0.1] * 50 + [1.0] * 50)
    sparse = _series([_LONG[0], _LONG[50]], [0.1, 1.0])  # one row per event, typical vendor format
    r = reconcile_symbol("TESTC2", dense, sparse)
    assert r.classification == ReconciliationClass.EXACT
