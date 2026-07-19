"""The shared tape-boundary locator.

One function, used by two callers in a full deployment of this method: the load-time snap
pass (:func:`tapetruth.chain.snap_actions_to_tape`) and a separate reconciliation-repair
engine you might build on top of :mod:`tapetruth.reconcile` -- kept as ONE implementation so
the two never drift apart (see ``docs/STANDARD.md`` D3: "one chain, one chokepoint").
"""
from __future__ import annotations

import datetime as _dt
import math as _math

import polars as pl

__all__ = ["find_unique_boundary"]


def find_unique_boundary(
    series: pl.DataFrame,
    center_date: _dt.date,
    implied_gap: float,
    back_days: int = 15,
    fwd_days: int = 7,
    tol_ln: float | None = None,
) -> dict | None:
    """Scan a raw close series for a UNIQUE single-day boundary matching ``implied_gap``
    within ``[center_date - back_days, center_date + fwd_days]`` (calendar days).

    Three gates, all required:

    - **MATCH** -- the day's close-over-close ratio is within ``tol_ln`` (default
      ``ln(1.35)``) of ``implied_gap``.
    - **UNIQUE** -- exactly one date in the window matches. Two or more candidate days means
      the search can't tell which one is the real boundary, so it refuses to guess.
    - **PERSISTENCE** -- the NEXT bar must not reverse the jump: an opposite-direction move
      recovering more than half the jump's log-magnitude means the "boundary" was a lone bad
      print, not a real, sticking step. A same-direction drift the next day is fine (that's
      ordinary volatility, not a reversal).

    Why this matters: a corporate action logged on the wrong date breaks every
    date-anchored consumer downstream (a refuter reads "no gap here" and kills a real split;
    a vintage classifier reads the same false negative and assumes the series is already
    adjusted). Relocating the action's date to where the tape actually moves fixes all of
    them at once, from one function.

    Parameters
    ----------
    series : pl.DataFrame
        Columns ``date`` (ascending) and ``close``. Typically your raw/as-traded tape --
        the whole point of using the raw series is that it's immune to adjustment-factor
        bugs by construction (D1).
    center_date : datetime.date
        The claimed (possibly wrong) action date to search around.
    implied_gap : float
        The ratio a genuine boundary at this action would produce:
        ``close_after / close_before``. For a ``ratio_from:ratio_to`` split claim this is
        ``ratio_from / ratio_to`` (see :mod:`tapetruth.chain`).
    back_days, fwd_days : int
        Search window, calendar days, before/after ``center_date``.
    tol_ln : float, optional
        Log-tolerance for the MATCH gate. Defaults to ``ln(1.35)``.

    Returns
    -------
    dict | None
        ``{"tape_date": date, "measured_1d": float}`` on a unique, persistent match, else
        ``None``.
    """
    if tol_ln is None:
        tol_ln = _math.log(1.35)
    if series is None or series.height < 3 or implied_gap is None or implied_gap <= 0:
        return None

    lo = center_date - _dt.timedelta(days=back_days)
    hi = center_date + _dt.timedelta(days=fwd_days)
    s = series.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).alias("_r"),
        (pl.col("close").shift(-1) / pl.col("close")).alias("_r_next"),
    )
    win = s.filter(
        (pl.col("date") >= lo) & (pl.col("date") <= hi)
        & pl.col("_r").is_not_null() & (pl.col("_r") > 0)
    )
    if win.height == 0:
        return None

    ln_implied = _math.log(implied_gap)
    cands = [
        row for row in win.iter_rows(named=True)
        if abs(_math.log(row["_r"]) - ln_implied) < tol_ln
    ]
    if len(cands) != 1:
        return None

    c = cands[0]
    r_next = c["_r_next"]
    if r_next is None or r_next <= 0:
        return None
    ln_next = _math.log(r_next)
    if ln_next * ln_implied < 0 and abs(ln_next) > 0.5 * abs(ln_implied):
        return None  # reversal -- lone bad print, not a real boundary

    return {"tape_date": c["date"], "measured_1d": round(float(c["_r"]), 6)}
