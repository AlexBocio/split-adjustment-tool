"""The honest-classes reconciler.

A reconciler that can only say "match" or "mismatch" hides information: a symbol with no
date overlap between two sources looks identical to a symbol whose adjustment history is
flatly wrong, unless the reconciler tells you which. This module compares two independent
cumulative-adjustment-factor series for the same symbol -- yours and a reference's -- and
classifies the result into one of six honest classes instead of a boolean.

Both sides are just factor-series functions -- ``fn(symbol) -> pl.DataFrame(date, factor) |
None``. tapetruth's own output (see :mod:`tapetruth.chain`) can be one side; a vendor, a
broker, or a second independent pipeline can be the other. Nothing here is vendor-specific.

The six classes
----------------
``EXACT``
    The two series agree everywhere they overlap, within tight tolerance.
``MINOR``
    Small, bounded disagreement -- day-count conventions, rounding, timezone-of-record
    differences. Worth knowing about, not worth alarming over.
``EXACT_ADJUSTED_EQUIV``
    The two series are PARALLEL but offset -- every period-over-period ratio matches (so a
    price adjusted by either series moves identically), they just anchor their cumulative
    product to a different starting point/vintage. This is a real, common, benign pattern:
    two pipelines that agree on every split event but started compounding from a different
    base date.
``EXPLAINED``
    A real disagreement exists, but you supplied (via `explain_fn`) a specific, citable
    reason for it -- a known scope difference (e.g. one side only tracks splits, not
    spin-offs), a post-coverage action on one side, or any other accepted, documented cause.
``NOT_COMPARABLE``
    There isn't enough overlapping data to say anything -- missing on one side, or too few
    overlapping dates to be statistically meaningful.
``MISMATCH``
    A real, unexplained disagreement. This is the only class that means "you (or the
    reference) might actually be wrong" -- everything else is either agreement or a
    labeled, accepted reason for disagreement.

The acceptance gate (:func:`reconciliation_gate`) evaluates only the COMPARABLE set (every
class except `NOT_COMPARABLE`); uncomparable symbols are reported, never silently hidden and
never counted as failures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import polars as pl

__all__ = [
    "ReconciliationClass",
    "ReconcileConfig",
    "SymbolReconciliation",
    "reconcile_symbol",
    "reconcile_all",
    "reconciliation_gate",
]


class ReconciliationClass:
    """The six honest classes. Plain string constants (not an Enum) so results serialize to
    CSV/Parquet/JSON without any special-casing."""

    EXACT = "EXACT"
    MINOR = "MINOR"
    EXACT_ADJUSTED_EQUIV = "EXACT_ADJUSTED_EQUIV"
    EXPLAINED = "EXPLAINED"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    MISMATCH = "MISMATCH"

    ALL = (EXACT, MINOR, EXACT_ADJUSTED_EQUIV, EXPLAINED, NOT_COMPARABLE, MISMATCH)
    #: Classes that count as "agreement or accepted" for the acceptance gate.
    PASSING = (EXACT, MINOR, EXACT_ADJUSTED_EQUIV, EXPLAINED)


@dataclass
class ReconcileConfig:
    """Tolerances for :func:`reconcile_symbol`. All thresholds are on
    ``ln(ours/reference)`` -- log-space so a 2x-too-high error and a 2x-too-low error are
    symmetric."""

    #: median |ln diff| at/below this -> EXACT. Default ~0.1%.
    exact_tol_ln: float = math.log(1.001)
    #: median |ln diff| at/below this (but above exact) -> MINOR. Default ~2%.
    minor_tol_ln: float = math.log(1.02)
    #: For EXACT_ADJUSTED_EQUIV: the two series are "parallel" when the STANDARD DEVIATION of
    #: the per-date ln-difference is at/below this, even if the mean offset is large --
    #: i.e. every period-over-period step agrees, only the total anchor differs.
    parallel_tol_ln: float = math.log(1.001)
    #: Fewer overlapping dates than this -> NOT_COMPARABLE (too little evidence to classify).
    min_date_overlap: int = 5


@dataclass
class SymbolReconciliation:
    """One symbol's reconciliation result."""

    symbol: str
    classification: str
    detail: str
    n_dates_compared: int
    median_abs_ln_diff: float | None
    max_abs_ln_diff: float | None
    mean_ln_diff: float | None
    std_ln_diff: float | None


def _stats(diffs: list[float]) -> tuple[float, float, float, float]:
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / n
    std = math.sqrt(var)
    sorted_abs = sorted(abs(x) for x in diffs)
    mid = n // 2
    median_abs = (sorted_abs[mid] if n % 2 == 1
                 else (sorted_abs[mid - 1] + sorted_abs[mid]) / 2.0)
    max_abs = sorted_abs[-1]
    return median_abs, max_abs, mean, std


def reconcile_symbol(
    symbol: str,
    ours: pl.DataFrame | None,
    reference: pl.DataFrame | None,
    config: ReconcileConfig = ReconcileConfig(),
    explain_fn: Callable[[str], str | None] | None = None,
) -> SymbolReconciliation:
    """Reconcile one symbol's two cumulative-adjustment-factor series.

    Parameters
    ----------
    symbol : str
    ours, reference : pl.DataFrame | None
        Columns ``date``, ``factor`` (cumulative adjustment factor, any positive float
        convention as long as both sides use the SAME convention). `None` or empty means
        "no data on this side."
    config : ReconcileConfig
    explain_fn : callable, optional
        ``fn(symbol) -> str | None``. Called only when the comparison would otherwise be a
        MISMATCH; if it returns a non-None string, the classification becomes EXPLAINED and
        the returned string becomes `detail`. Use this for scope differences you already
        know about (e.g. "reference does not track spin-offs").
    """
    if ours is None or reference is None or ours.height == 0 or reference.height == 0:
        return SymbolReconciliation(symbol, ReconciliationClass.NOT_COMPARABLE,
                                     "no data on one or both sides", 0, None, None, None, None)

    joined = ours.rename({"factor": "_ours"}).join(
        reference.rename({"factor": "_ref"}), on="date", how="inner"
    )
    if joined.height < config.min_date_overlap:
        return SymbolReconciliation(
            symbol, ReconciliationClass.NOT_COMPARABLE,
            f"only {joined.height} overlapping date(s), need >= {config.min_date_overlap}",
            joined.height, None, None, None, None,
        )

    ours_vals = joined["_ours"].to_list()
    ref_vals = joined["_ref"].to_list()
    diffs = []
    for o, r in zip(ours_vals, ref_vals):
        if o is None or r is None or o <= 0 or r <= 0:
            continue
        diffs.append(math.log(o) - math.log(r))
    if len(diffs) < config.min_date_overlap:
        return SymbolReconciliation(
            symbol, ReconciliationClass.NOT_COMPARABLE,
            f"only {len(diffs)} usable (positive, non-null) overlapping value(s), "
            f"need >= {config.min_date_overlap}",
            len(diffs), None, None, None, None,
        )

    median_abs, max_abs, mean_ln, std_ln = _stats(diffs)
    n = len(diffs)

    if median_abs <= config.exact_tol_ln:
        return SymbolReconciliation(symbol, ReconciliationClass.EXACT,
                                     "median log-difference within exact tolerance",
                                     n, median_abs, max_abs, mean_ln, std_ln)

    if median_abs <= config.minor_tol_ln:
        return SymbolReconciliation(symbol, ReconciliationClass.MINOR,
                                     "median log-difference within minor tolerance",
                                     n, median_abs, max_abs, mean_ln, std_ln)

    # Only reached once the disagreement is bigger than "minor" -- a small constant offset is
    # already MINOR above; EXACT_ADJUSTED_EQUIV means a LARGE but perfectly parallel offset
    # (every period-over-period step still agrees), the "different vintage/anchor" signature.
    if std_ln <= config.parallel_tol_ln:
        return SymbolReconciliation(
            symbol, ReconciliationClass.EXACT_ADJUSTED_EQUIV,
            "series are parallel (every period-over-period step agrees) but offset by a "
            "constant -- consistent with a different vintage/anchor date, not a disagreement "
            "about any actual event",
            n, median_abs, max_abs, mean_ln, std_ln,
        )

    if explain_fn is not None:
        explanation = explain_fn(symbol)
        if explanation:
            return SymbolReconciliation(symbol, ReconciliationClass.EXPLAINED, explanation,
                                         n, median_abs, max_abs, mean_ln, std_ln)

    return SymbolReconciliation(
        symbol, ReconciliationClass.MISMATCH,
        f"median |ln diff| {median_abs:.4f} exceeds minor tolerance "
        f"{config.minor_tol_ln:.4f} with no accepted explanation",
        n, median_abs, max_abs, mean_ln, std_ln,
    )


def reconcile_all(
    symbols: list[str],
    ours_fn: Callable[[str], pl.DataFrame | None],
    reference_fn: Callable[[str], pl.DataFrame | None],
    config: ReconcileConfig = ReconcileConfig(),
    explain_fn: Callable[[str], str | None] | None = None,
) -> pl.DataFrame:
    """Runs :func:`reconcile_symbol` for every symbol; returns one row per symbol as a
    DataFrame (columns match :class:`SymbolReconciliation`'s fields)."""
    rows = []
    for sym in symbols:
        r = reconcile_symbol(sym, ours_fn(sym), reference_fn(sym), config=config, explain_fn=explain_fn)
        rows.append({
            "symbol": r.symbol, "classification": r.classification, "detail": r.detail,
            "n_dates_compared": r.n_dates_compared, "median_abs_ln_diff": r.median_abs_ln_diff,
            "max_abs_ln_diff": r.max_abs_ln_diff, "mean_ln_diff": r.mean_ln_diff,
            "std_ln_diff": r.std_ln_diff,
        })
    schema = {"symbol": pl.String, "classification": pl.String, "detail": pl.String,
              "n_dates_compared": pl.Int64, "median_abs_ln_diff": pl.Float64,
              "max_abs_ln_diff": pl.Float64, "mean_ln_diff": pl.Float64, "std_ln_diff": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def reconciliation_gate(results: pl.DataFrame, min_pass_rate: float = 0.99) -> dict:
    """Aggregate acceptance gate over a :func:`reconcile_all` result: is the COMPARABLE set's
    pass rate (EXACT + MINOR + EXACT_ADJUSTED_EQUIV + EXPLAINED) at or above `min_pass_rate`?
    `NOT_COMPARABLE` rows are excluded from the denominator entirely -- reported, never
    counted as failures.

    Returns a dict: n_total, n_comparable, n_passing, n_mismatch, pass_rate, gate_passed.
    """
    n_total = results.height
    comparable = results.filter(pl.col("classification") != ReconciliationClass.NOT_COMPARABLE)
    n_comparable = comparable.height
    n_passing = comparable.filter(pl.col("classification").is_in(list(ReconciliationClass.PASSING))).height
    n_mismatch = comparable.filter(pl.col("classification") == ReconciliationClass.MISMATCH).height
    pass_rate = (n_passing / n_comparable) if n_comparable else 1.0
    return {
        "n_total": n_total, "n_comparable": n_comparable, "n_not_comparable": n_total - n_comparable,
        "n_passing": n_passing, "n_mismatch": n_mismatch, "pass_rate": pass_rate,
        "min_pass_rate": min_pass_rate, "gate_passed": pass_rate >= min_pass_rate,
    }
