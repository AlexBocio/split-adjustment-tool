"""The OHLC bad-print sanity guard.

A real daily OHLC field is never a wild multiple away from its own close -- and within one
day's candle, high/low must bound open/close by construction. This module catches and
repairs the prints that violate those invariants: odd-lot / out-of-sequence / erroneous
trade prints that leak into a daily high or low field even though the close itself (and the
rest of the session) is fine.

Repair order, each gate independent (any can fire on its own):

1. **Band-out-of-bounds** -- a field more than ``config.band_high`` above, or below
   ``config.band_low`` times, its own close.
2. **Geometry violation** -- ``high`` below ``max(open, close)``, or ``low`` above
   ``min(open, close)`` (a candle that doesn't bound its own body).
3. **Range-vs-truth contradiction** -- an implausible high/low spread
   (``> config.range_insanity_ratio``) that an independent sub-daily truth source
   contradicts by more than ``config.range_contradiction_mult``. A genuine wild day that
   agrees with its own sub-daily prints is left alone -- only a spread the truth source
   itself refutes gets repaired.
4. **Dense-exceed** -- an in-band, sub-2x-range value that's still wrong: with DENSE
   sub-daily coverage (``>= config.dense_min_bars``), the sub-daily extremes effectively ARE
   the session, so a daily field exceeding them by more than ``config.dense_exceed_tol``
   is defective by definition. Sparse days are never judged this way (a thin session's true
   range can legitimately exceed a handful of sub-daily prints).

When triggered AND a truth source confirms a value, the repair uses the truth value --
**clamped to the band contract**, not the raw truth measurement. This is deliberate: the
measurement-sanity bound (``config.truth_sanity_band``, looser, validates that the
*measurement itself* is plausible) and the contract bound (``band_low``/``band_high``,
tighter, is the promise every downstream consumer relies on) are different things. A
real, truth-confirmed wick beyond the contract band lands AT the band edge, not deleted down
to close -- much closer to the truth than the old envelope-only behavior (see the module
docstring's history note below), while still respecting the contract every consumer counts
on. When no truth source is available (or none confirms this row), the guard falls back to
snapping the offending field to close (the "envelope" repair).

**A prior version of this exact mechanism deleted real wicks**: without any independent
truth source, "the offending field must be wrong, clamp it to close" is the only defensible
fallback -- but it means a genuinely enormous, real move that happens to lack sub-daily
corroboration gets flattened. Wiring in a :class:`~tapetruth.providers.TruthProvider` is
what turns "delete the wick" into "confirm and clamp the wick" -- see the `envelope_wick`
class in :mod:`tapetruth.gauntlet` for a runnable demonstration of the difference.

**Crash-day fix (v0.1.1).** Earlier versions measured every limit against the CLOSE. On a real
one-day crash the open itself sits far from the close, so the open, high and low all looked
"out of bounds" and a genuine move was flattened; a confirming truth source was then discarded by
a fixed 2.5x-of-close sanity check. Now: (1) the OPEN is kept whenever a usable truth range shows
trading there; (2) HIGH/LOW are judged as wicks against the day's BODY (max/min of open, close);
(3) each truth extreme is sanity-checked on its own, and a truth range must contain the close to be
usable at all. Without any truth source a real crash is still indistinguishable from a bad print --
give the guard sub-daily data for volatile small caps.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

__all__ = ["GuardConfig", "apply_bad_print_guard"]


@dataclass
class GuardConfig:
    """Every tolerance the guard uses. Defaults are the values this engine was tuned and
    measured against."""

    #: A field further than this multiple below its own close is out-of-bounds.
    band_low: float = 0.55
    #: A field further than this multiple above its own close is out-of-bounds.
    band_high: float = 1.8

    #: A truth-source measurement more than this multiple away from close (either direction)
    #: is treated as an unreliable measurement, not a real repair target -- protects against
    #: a degraded or garbage truth source injecting its own bad values.
    truth_sanity_band: float = 2.5

    #: high/low ratio above this is "range-insane" -- a candidate for repair, but ONLY when a
    #: truth source also contradicts it (see `range_contradiction_mult`).
    range_insanity_ratio: float = 2.0
    #: A range-insane day only repairs when the daily high/low ratio exceeds the truth
    #: source's own high/low ratio by more than this multiple.
    range_contradiction_mult: float = 1.5

    #: Sub-daily bar count at/above which coverage counts as "dense" -- dense enough that the
    #: sub-daily extremes ARE effectively the session's true range.
    dense_min_bars: int = 300
    #: Tolerance band (fraction) around the truth extreme under dense coverage -- e.g. 0.02
    #: means a daily low more than 2% below the truth low is defective.
    dense_exceed_tol: float = 0.02

    #: Relative tolerance for "the truth source shows trading at this price": an open inside the
    #: truth range (within this fraction), or a truth range that contains the close, counts as
    #: confirmed. Confirmed real moves are never flattened.
    confirm_tol: float = 0.02

    #: Floating-point slack for the geometry check (``high >= max(open,close)``,
    #: ``low <= min(open,close)``).
    geometry_tol: float = 0.0001


def _gather_truth(df: pl.DataFrame, truth) -> pl.DataFrame:
    """Runs `truth.get_truth(symbol, date)` for every distinct (symbol, date) pair in `df`
    and assembles the results into a small joinable DataFrame. Returns an empty (but
    correctly-typed) frame when `truth` is None or yields nothing."""
    schema = {"symbol": pl.String, "date": pl.Date,
              "truth_high": pl.Float64, "truth_low": pl.Float64, "truth_n": pl.Int64}
    if truth is None or df.height == 0:
        return pl.DataFrame(schema=schema)
    pairs = df.select(["symbol", "date"]).unique()
    rows = []
    for r in pairs.iter_rows(named=True):
        t = truth.get_truth(r["symbol"], r["date"])
        if t is not None and t.get("high") is not None and t.get("low") is not None:
            rows.append({
                "symbol": r["symbol"], "date": r["date"],
                "truth_high": float(t["high"]), "truth_low": float(t["low"]),
                "truth_n": int(t.get("n_bars", 0) or 0),
            })
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def apply_bad_print_guard(
    df: pl.DataFrame, truth=None, config: GuardConfig = GuardConfig()
) -> pl.DataFrame:
    """THE shared chokepoint for OHLC sanity repair (D3 applied to bars, not just actions --
    run this at every point your pipeline writes or re-writes daily bars).

    Parameters
    ----------
    df : pl.DataFrame
        Columns ``symbol, date, open, high, low, close`` (any other columns, e.g. `volume`,
        pass through untouched). Whatever unit system these prices are in (raw or already
        adjusted) is up to you -- the guard doesn't care, it only checks internal
        consistency and, optionally, agreement with `truth`.
    truth : TruthProvider, optional
        Independent sub-daily extremes, in the SAME unit system as `df`. Omit to run
        envelope-only repair (see module docstring).
    config : GuardConfig
        Tolerances; see :class:`GuardConfig`.

    Returns
    -------
    pl.DataFrame
        `df` with `open`/`high`/`low` repaired in place where triggered, plus two new
        columns: ``bad_print_flag`` (Int8, 1 iff ANY field was changed) and
        ``repair_source`` (String, ``"truth"`` / ``"envelope"`` / null).
    """
    band_low, band_high = config.band_low, config.band_high
    tol = config.confirm_tol

    truth_df = _gather_truth(df, truth)
    if truth_df.height > 0:
        out = df.join(truth_df, on=["symbol", "date"], how="left")
    else:
        out = df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("truth_high"),
            pl.lit(None, dtype=pl.Float64).alias("truth_low"),
            pl.lit(0, dtype=pl.Int64).alias("truth_n"),
        ])

    # A truth source is only usable at all if it is internally sane AND its range contains the
    # day's own close -- a sub-daily record of the same session must bracket where it ended.
    truth_present = (
        pl.col("truth_high").is_not_null() & pl.col("truth_low").is_not_null()
        & (pl.col("truth_low") > 0) & (pl.col("truth_high") >= pl.col("truth_low"))
        & (pl.col("close") > 0)
        & (pl.col("close") >= pl.col("truth_low") * (1 - tol))
        & (pl.col("close") <= pl.col("truth_high") * (1 + tol))
    )

    # 1. OPEN. A real gap or crash moves the open away from the close; an open outside the band
    #    is only repaired when no usable truth shows the session actually trading there.
    open_oob = (pl.col("close") > 0) & (
        (pl.col("open") > band_high * pl.col("close")) | (pl.col("open") < band_low * pl.col("close"))
    )
    open_confirmed = truth_present & (pl.col("open") >= pl.col("truth_low") * (1 - tol))         & (pl.col("open") <= pl.col("truth_high") * (1 + tol))
    out = out.with_columns(pl.col("open").alias("_o0")).with_columns(
        pl.when(open_oob & ~open_confirmed).then(pl.col("close")).otherwise(pl.col("open")).alias("_o"),
        (open_oob & open_confirmed).alias("_open_confirmed"),
    )

    # 2. HIGH / LOW are WICKS: judged against the day's BODY (max/min of open and close), not
    #    against the close alone -- a real crash moves the whole body, a bad print sticks out
    #    beyond it. Measuring wicks against the close flattened genuine crash days.
    body_hi = pl.max_horizontal("_o", "close")
    body_lo = pl.min_horizontal("_o", "close")
    high_oob = (body_hi > 0) & (pl.col("high") > band_high * body_hi)
    low_oob = (body_lo > 0) & (pl.col("low") < band_low * body_lo)

    # Each truth extreme is sanity-checked on its own, relative to the body: one wild extreme
    # must not discard the other, and a truth value far outside even the loose sanity band is
    # treated as a degraded measurement (it may carry the same bad print).
    th_ok = truth_present & (pl.col("truth_high") <= config.truth_sanity_band * body_hi)         & (pl.col("truth_high") >= body_hi * (1 - tol))
    tl_ok = truth_present & (pl.col("truth_low") >= body_lo / config.truth_sanity_band)         & (pl.col("truth_low") <= body_lo * (1 + tol))
    truth_ok = th_ok & tl_ok

    geom_bad = (
        (pl.col("high") < body_hi * (1 - config.geometry_tol))
        | (pl.col("low") > body_lo * (1 + config.geometry_tol))
    )
    range_bad = (
        (pl.col("low") > 0) & ((pl.col("high") / pl.col("low")) > config.range_insanity_ratio)
        & truth_ok
        & ((pl.col("high") / pl.col("low"))
           > config.range_contradiction_mult * (pl.col("truth_high") / pl.col("truth_low")))
    )
    dense = (pl.col("truth_n") >= config.dense_min_bars) & truth_present
    dense_low = dense & tl_ok & (pl.col("low") < (1 - config.dense_exceed_tol) * pl.col("truth_low"))
    dense_high = dense & th_ok & (pl.col("high") > (1 + config.dense_exceed_tol) * pl.col("truth_high"))

    fix_high = high_oob | geom_bad | range_bad | dense_high
    fix_low = low_oob | geom_bad | range_bad | dense_low

    out = out.with_columns([
        pl.when(fix_high & th_ok)
          .then(pl.min_horizontal(pl.col("truth_high"), band_high * body_hi))
          .when(fix_high).then(body_hi)
          .otherwise(pl.col("high")).alias("_h"),
        pl.when(fix_low & tl_ok)
          .then(pl.max_horizontal(pl.col("truth_low"), band_low * body_lo))
          .when(fix_low).then(body_lo)
          .otherwise(pl.col("low")).alias("_l"),
        ((fix_high & th_ok) | (fix_low & tl_ok)).alias("_by_truth"),
        ((fix_high & ~th_ok) | (fix_low & ~tl_ok)).alias("_by_env"),
    ]).with_columns([
        pl.col("_o").alias("open"),
        pl.max_horizontal("_o", "_h", "close").alias("_H"),
        pl.min_horizontal("_o", "_l", "close").alias("_L"),
    ])
    out = out.with_columns([
        ((pl.col("_o") != pl.col("_o0")) | (pl.col("_H") != pl.col("high"))
         | (pl.col("_L") != pl.col("low"))).cast(pl.Int8).alias("bad_print_flag"),
    ]).with_columns([
        pl.when((pl.col("bad_print_flag") == 1) & pl.col("_by_truth")).then(pl.lit("truth"))
          .when(pl.col("bad_print_flag") == 1).then(pl.lit("envelope"))
          .when(pl.col("_open_confirmed")).then(pl.lit("confirmed"))
          .otherwise(pl.lit(None, dtype=pl.String)).alias("repair_source"),
        pl.col("_H").alias("high"),
        pl.col("_L").alias("low"),
    ]).drop(["_o", "_h", "_l", "_H", "_L", "_by_truth", "_by_env", "_o0", "_open_confirmed",
             "truth_high", "truth_low", "truth_n"])
    return out
