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
``DATE_DIFFERS``
    Both sides record the same events with the same ratios, but at least one event sits on a
    different date (within ``event_date_window_days``) -- e.g. an announcement date vs the
    market ex-date. Not passing: every date-anchored consumer reads the wrong day.
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
    DATE_DIFFERS = "DATE_DIFFERS"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    MISMATCH = "MISMATCH"

    ALL = (EXACT, MINOR, EXACT_ADJUSTED_EQUIV, EXPLAINED, DATE_DIFFERS, NOT_COMPARABLE, MISMATCH)
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
    #: Overlap is counted AFTER an as-of (forward-fill) alignment, so a sparse one-row-per-event
    #: reference table is comparable against a dense daily series.
    min_date_overlap: int = 5
    #: A period-over-period factor change larger than this (log) is an EVENT (a split, etc.).
    event_step_tol_ln: float = math.log(1.01)
    #: Two events (one per side) with matching ratios count as the SAME event when their dates
    #: are within this many calendar days; same event on different dates -> DATE_DIFFERS.
    event_date_window_days: int = 10


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
    n_events_ours: int = 0
    n_events_reference: int = 0
    #: One line per event-level disagreement, e.g. "RATIO_DIFFERS 2024-06-10 ours x0.1000 ref
    #: x0.0500", "DATE_DIFFERS ours 2024-07-11 ref 2024-07-15 x0.1000", "ONLY_OURS ...",
    #: "ONLY_REFERENCE ...".
    event_issues: list[str] = field(default_factory=list)


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

    # As-of alignment: union of dates, each side forward-filled from its own last known value,
    # so a sparse one-row-per-event table compares against a dense daily series.
    o = ours.select(pl.col("date").cast(pl.Date), pl.col("factor").cast(pl.Float64).alias("_ours"))
    r = reference.select(pl.col("date").cast(pl.Date), pl.col("factor").cast(pl.Float64).alias("_ref"))
    dates = pl.concat([o.select("date"), r.select("date")]).unique().sort("date")
    joined = (dates.join(o, on="date", how="left").join(r, on="date", how="left")
              .with_columns(pl.col("_ours").forward_fill(), pl.col("_ref").forward_fill())
              .filter(pl.col("_ours").is_not_null() & pl.col("_ref").is_not_null()
                      & (pl.col("_ours") > 0) & (pl.col("_ref") > 0)))
    if joined.height < config.min_date_overlap:
        return SymbolReconciliation(
            symbol, ReconciliationClass.NOT_COMPARABLE,
            f"only {joined.height} overlapping date(s) after as-of alignment, need >= "
            f"{config.min_date_overlap}",
            joined.height, None, None, None, None,
        )

    d = joined["date"].to_list()
    ov = joined["_ours"].to_list()
    rv = joined["_ref"].to_list()
    diffs = [math.log(a) - math.log(b) for a, b in zip(ov, rv)]
    median_abs, max_abs, mean_ln, std_ln = _stats(diffs)
    n = len(diffs)

    def _events(vals):
        ev = []
        for i in range(1, len(vals)):
            step = math.log(vals[i]) - math.log(vals[i - 1])
            if abs(step) > config.event_step_tol_ln:
                ev.append((d[i], step))
        return ev

    eo, er = _events(ov), _events(rv)
    issues, used = [], set()
    for (do_, so) in eo:
        best = None
        for k, (dr_, sr) in enumerate(er):
            if k in used or abs(so - sr) > config.minor_tol_ln:
                continue
            gap = abs((dr_ - do_).days)
            if gap <= config.event_date_window_days and (best is None or gap < best[1]):
                best = (k, gap)
        if best is not None:
            used.add(best[0])
            if best[1] > 0:
                issues.append(f"DATE_DIFFERS ours {do_} ref {er[best[0]][0]} x{math.exp(so):.4f}")
            continue
        same_day = [k for k, (dr_, _) in enumerate(er) if k not in used and dr_ == do_]
        if same_day:
            used.add(same_day[0])
            issues.append(f"RATIO_DIFFERS {do_} ours x{math.exp(so):.4f} "
                          f"ref x{math.exp(er[same_day[0]][1]):.4f}")
        else:
            issues.append(f"ONLY_OURS {do_} x{math.exp(so):.4f}")
    for k, (dr_, sr) in enumerate(er):
        if k not in used:
            issues.append(f"ONLY_REFERENCE {dr_} x{math.exp(sr):.4f}")

    def _res(cls, detail):
        return SymbolReconciliation(symbol, cls, detail, n, median_abs, max_abs, mean_ln, std_ln,
                                    len(eo), len(er), issues)

    if not issues:
        # Every event agrees on date and ratio; judge the LEVEL on the worst date, not the median.
        if max_abs <= config.exact_tol_ln:
            return _res(ReconciliationClass.EXACT,
                        "every event and every date agree within exact tolerance")
        if max_abs <= config.minor_tol_ln:
            return _res(ReconciliationClass.MINOR,
                        "every event agrees; worst-date level difference within minor tolerance")
        if std_ln <= config.parallel_tol_ln:
            return _res(ReconciliationClass.EXACT_ADJUSTED_EQUIV,
                        "every event agrees; the series are parallel but offset by a constant "
                        "(different anchor/vintage, not a disagreement about any event)")
    if explain_fn is not None:
        explanation = explain_fn(symbol)
        if explanation:
            return _res(ReconciliationClass.EXPLAINED, explanation)
    if issues and all(i.startswith("DATE_DIFFERS") for i in issues):
        return _res(ReconciliationClass.DATE_DIFFERS, "; ".join(issues))
    if issues:
        return _res(ReconciliationClass.MISMATCH, "; ".join(issues))
    return _res(ReconciliationClass.MISMATCH,
                f"no event disagreement but the worst-date level difference {max_abs:.4f} exceeds "
                f"minor tolerance {config.minor_tol_ln:.4f} and the series are not parallel")


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
            "std_ln_diff": r.std_ln_diff, "n_events_ours": r.n_events_ours,
            "n_events_reference": r.n_events_reference, "event_issues": "; ".join(r.event_issues),
        })
    schema = {"symbol": pl.String, "classification": pl.String, "detail": pl.String,
              "n_dates_compared": pl.Int64, "median_abs_ln_diff": pl.Float64,
              "max_abs_ln_diff": pl.Float64, "mean_ln_diff": pl.Float64, "std_ln_diff": pl.Float64,
              "n_events_ours": pl.Int64, "n_events_reference": pl.Int64, "event_issues": pl.String}
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
