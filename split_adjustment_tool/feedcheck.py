"""Feed checks: frozen prices and wrong-day stamps.

Two ways a price feed goes wrong without any split being involved:

* **Frozen prices** (:func:`scan_stale_runs`) -- the feed repeats the same close for days while the
  real market moved. A run of identical closes longer than the symbol's own normal is reported; an
  independent WITNESS series decides what it was: the witness moved -> ``STALE_FEED``; the witness was
  flat too -> ``ILLIQUID`` (the stock genuinely did not trade); no witness -> ``UNVERIFIED``.
* **Wrong-day stamps** (:func:`scan_date_shifts`) -- bars carry the wrong date (each bar one session
  late or early). Over a rolling window the primary's daily moves are compared with the witness's
  moves on the same day and one session either side; a window that matches far better shifted is
  part of a ``DATE_SHIFT`` span, reported with its start, end and direction.

Both take ``series_fn(symbol) -> DataFrame[date, close]`` (e.g.
:func:`split_adjustment_tool.chain.make_close_series_fn`) for the primary and, optionally, the same for a
witness (a second vendor, or daily closes built from your own intraday tape). Thresholds scale to each
symbol's own behaviour; nothing is acted on, everything is reported.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

__all__ = ["FeedCheckConfig", "scan_stale_runs", "scan_date_shifts"]

STALE_FEED = "STALE_FEED"
ILLIQUID = "ILLIQUID"
UNVERIFIED = "UNVERIFIED"
DATE_SHIFT = "DATE_SHIFT"


@dataclass
class FeedCheckConfig:
    #: A run of identical closes is reported when it is at least this many bars long...
    min_stale_run: int = 3
    #: ...and longer than this quantile of the symbol's OWN run lengths (an illiquid name that often
    #: sits flat for a few days has a higher bar than a liquid one).
    own_run_quantile: float = 0.99
    #: The witness "moved" during a run when its close changed by more than this (log) from the bar
    #: before the run to the run's last bar.
    witness_move_ln: float = math.log(1.02)
    #: Rolling window (in shared sessions) for the date-shift test.
    shift_window: int = 20
    #: A window is shifted when the best one-session-shifted error is below this fraction of the
    #: aligned error...
    shift_improvement: float = 0.5
    #: ...and the aligned error is above this floor (mean |log move| difference): two clean vendors that
    #: agree to within normal noise are never "shifted".
    shift_min_error_ln: float = 0.004


def _runs(dates, closes):
    """[(start_idx, end_idx)] of maximal runs of identical consecutive closes (length >= 2)."""
    out, i, n = [], 0, len(closes)
    while i < n:
        j = i
        while j + 1 < n and closes[j + 1] == closes[i]:
            j += 1
        if j > i:
            out.append((i, j))
        i = j + 1
    return out


def scan_stale_runs(
    series_fn: Callable[[str], pl.DataFrame | None],
    symbols: list[str],
    witness_series_fn: Callable[[str], pl.DataFrame | None] | None = None,
    config: FeedCheckConfig = FeedCheckConfig(),
) -> pl.DataFrame:
    """Report runs of identical closes longer than the symbol's own normal, classified against a
    witness. Columns: ``symbol, start, end, n_bars, close, witness_move, classification``."""
    rows = []
    for sym in symbols:
        s = series_fn(sym)
        if s is None or s.height < config.min_stale_run:
            continue
        s = s.sort("date")
        dates, closes = s["date"].to_list(), s["close"].to_list()
        runs = _runs(dates, closes)
        if not runs:
            continue
        lengths = sorted(j - i + 1 for i, j in runs)
        own = lengths[min(len(lengths) - 1, int(config.own_run_quantile * len(lengths)))]
        # with few runs the quantile is the longest run itself; never let a single run set its own bar
        bar = max(config.min_stale_run, own if len(lengths) >= 20 else config.min_stale_run)
        w = witness_series_fn(sym) if witness_series_fn is not None else None
        for i, j in runs:
            n = j - i + 1
            if n < bar:
                continue
            move, cls = None, UNVERIFIED
            if w is not None and w.height:
                before_date = dates[i - 1] if i > 0 else dates[i]
                wb = w.filter(pl.col("date") <= before_date).tail(1)["close"]
                we = w.filter(pl.col("date") <= dates[j]).tail(1)["close"]
                win = w.filter((pl.col("date") >= dates[i]) & (pl.col("date") <= dates[j]))["close"]
                if len(wb) and len(we) and wb[0] > 0 and we[0] > 0:
                    move = math.log(we[0] / wb[0])
                    moved = abs(move) > config.witness_move_ln or (len(win) and win.n_unique() > 1
                                                                  and (win.max() / win.min()) > math.exp(config.witness_move_ln))
                    cls = STALE_FEED if moved else ILLIQUID
            rows.append({"symbol": sym, "start": dates[i], "end": dates[j], "n_bars": n, "close": closes[i],
                         "witness_move": move, "classification": cls})
    schema = {"symbol": pl.String, "start": pl.Date, "end": pl.Date, "n_bars": pl.Int64, "close": pl.Float64,
              "witness_move": pl.Float64, "classification": pl.String}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def scan_date_shifts(
    series_fn: Callable[[str], pl.DataFrame | None],
    witness_series_fn: Callable[[str], pl.DataFrame | None],
    symbols: list[str],
    config: FeedCheckConfig = FeedCheckConfig(),
) -> pl.DataFrame:
    """Report spans where the primary matches the witness better shifted by one session.

    ``shift = +1`` means the primary is stamped one session LATE (its bar dated D is the witness's
    D-1); ``-1`` means one session early. Columns: ``symbol, start, end, n_sessions, shift,
    aligned_error, shifted_error, classification``.
    """
    out = []
    for sym in symbols:
        p, w = series_fn(sym), witness_series_fn(sym)
        if p is None or w is None or p.height < config.shift_window + 2 or w.height < config.shift_window + 2:
            continue
        j = (p.sort("date").select("date", pl.col("close").alias("p"))
               .join(w.sort("date").select("date", pl.col("close").alias("w")), on="date", how="inner")
               .sort("date").filter((pl.col("p") > 0) & (pl.col("w") > 0)))
        if j.height < config.shift_window + 2:
            continue
        j = j.with_columns((pl.col("p") / pl.col("p").shift(1)).log().alias("rp"),
                           (pl.col("w") / pl.col("w").shift(1)).log().alias("rw"))
        j = j.with_columns(pl.col("rw").shift(1).alias("rw_prev"), pl.col("rw").shift(-1).alias("rw_next"))
        j = j.drop_nulls(["rp", "rw", "rw_prev", "rw_next"])
        n = config.shift_window
        e0 = (pl.col("rp") - pl.col("rw")).abs().rolling_mean(n)
        ep = (pl.col("rp") - pl.col("rw_prev")).abs().rolling_mean(n)
        em = (pl.col("rp") - pl.col("rw_next")).abs().rolling_mean(n)
        j = j.with_columns(e0.alias("e0"), ep.alias("ep"), em.alias("em")).drop_nulls(["e0"])
        dates = j["date"].to_list()
        flags = []
        for e_0, e_p, e_m in zip(j["e0"].to_list(), j["ep"].to_list(), j["em"].to_list(), strict=True):
            best, shift = (e_p, +1) if e_p <= e_m else (e_m, -1)
            ok = e_0 > config.shift_min_error_ln and best < config.shift_improvement * e_0
            flags.append((shift if ok else 0, e_0, best))
        # merge consecutive flagged window-ends with the same direction into spans; a window ending at
        # index k covers sessions [k-n+1, k], so a span starts n-1 sessions before its first flag
        k = 0
        while k < len(flags):
            if flags[k][0] == 0:
                k += 1
                continue
            sh, k0 = flags[k][0], k
            while k + 1 < len(flags) and flags[k + 1][0] == sh:
                k += 1
            seg = flags[k0:k + 1]
            all_dates = j["date"].to_list()
            start = all_dates[max(0, k0 - n + 1)] if k0 - n + 1 >= 0 else all_dates[0]
            out.append({"symbol": sym, "start": start, "end": dates[k], "n_sessions": (k - k0) + n,
                        "shift": sh, "aligned_error": max(f[1] for f in seg), "shifted_error": min(f[2] for f in seg),
                        "classification": DATE_SHIFT})
            k += 1
    schema = {"symbol": pl.String, "start": pl.Date, "end": pl.Date, "n_sessions": pl.Int64, "shift": pl.Int64,
              "aligned_error": pl.Float64, "shifted_error": pl.Float64, "classification": pl.String}
    return pl.DataFrame(out, schema=schema) if out else pl.DataFrame(schema=schema)
