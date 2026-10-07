"""Provider breaks: when the price tape itself goes wrong.

Everything else in this package checks split CLAIMS against your price tape and assumes the tape
is honest. This module covers the case where it is not: a provider delivers correct prices up to
some day X and then, from the next day on, prices on the wrong basis (10x off, a split applied
early, a split never applied) -- with or without a matching bogus claim.

Two steps:

1. :func:`scan_unexplained_jumps` finds every PERSISTENT level shift in a close series that is far
   outside that symbol's own normal day-to-day moves and that no claim explains. It also records
   whether the shift later reverses (the provider fixed itself), which bounds a bad-data window.
2. :func:`classify_jumps` asks an independent WITNESS series (a second vendor, or daily closes built
   from your own intraday tape) what happened on that date:

   * witness shows the same jump  -> ``REAL_MOVE``      (real: a split your claims feed missed, or a
                                                         genuine one-day crash/surge -- a split moves
                                                         volume inversely to price, a crash does not)
   * witness shows no jump        -> ``PROVIDER_BREAK`` (the primary feed is wrong from that date)
   * no witness / ambiguous       -> ``UNEXPLAINED``    (review it; a real crash also looks like this)

:func:`check_claims_against_witness` applies the same witness to the claims themselves, so a bogus
claim that happens to line up with a provider break is rejected instead of "confirmed" by bad data.

With a single source, a provider break and a real one-day crash are indistinguishable: both are a
persistent level shift. The witness is what separates them; without one, findings are reported as
``UNEXPLAINED`` for review, never silently acted on.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

__all__ = [
    "BreakConfig",
    "scan_unexplained_jumps",
    "classify_jumps",
    "check_claims_against_witness",
]

REAL_MOVE = "REAL_MOVE"
PROVIDER_BREAK = "PROVIDER_BREAK"
UNEXPLAINED = "UNEXPLAINED"


@dataclass
class BreakConfig:
    """Tolerances for break detection. Thresholds scale to each symbol's own volatility."""

    #: A day's close-over-close log move is a candidate jump when it exceeds this many robust
    #: standard deviations of the symbol's trailing daily moves...
    z_threshold: float = 8.0
    #: ...and is at least this large in absolute log terms (ln 1.5 by default), so a quiet
    #: series cannot turn ordinary noise into "jumps".
    min_jump_ln: float = math.log(1.5)
    #: Trailing window (bars) for the robust volatility estimate; the jump day itself is excluded.
    vol_lookback: int = 60
    #: Bars on each side used to confirm the shift PERSISTS (median after vs median before).
    persist_bars: int = 3
    #: The persisted shift must stay within this log-tolerance of the one-day jump.
    persist_tol_ln: float = math.log(1.35)
    #: A claim explains a jump when its implied ratio matches within this tolerance and its date
    #: is within ``claim_window_days`` calendar days of the jump.
    claim_tol_ln: float = math.log(1.35)
    claim_window_days: int = 15
    #: Look this many calendar days ahead for the provider reverting the break.
    revert_window_days: int = 120
    #: Witness agrees ("same jump") within ``claim_tol_ln``. It DISAGREES -- a provider break --
    #: when its move differs from the primary's by more than this (log) AND by more than half the
    #: primary jump: a micro-cap's witness may itself move 20-30% that day without agreeing with a
    #: 10x primary jump.
    witness_disagree_ln: float = math.log(1.6)
    #: A one-bar excursion that reverses the next bar within ``persist_tol_ln`` is reported as a
    #: ONE-DAY window (a whole bar delivered on the wrong basis looks internally normal, so the
    #: bad-print guard cannot see it). Requires the same size/volatility thresholds as a jump.
    report_one_day_windows: bool = True


def _robust_sigma(values: list[float]) -> float | None:
    if len(values) < 10:
        return None
    s = sorted(values)
    med = s[len(s) // 2]
    mad = sorted(abs(v - med) for v in values)[len(values) // 2]
    return 1.4826 * mad if mad > 0 else None


def _implied(claims: pl.DataFrame | None, symbol: str) -> list[tuple]:
    if claims is None or claims.height == 0:
        return []
    rows = claims.filter(pl.col("symbol") == symbol)
    out = []
    for r in rows.iter_rows(named=True):
        rf, rt = float(r["ratio_from"]), float(r["ratio_to"])
        if rf > 0 and rt > 0:
            out.append((r["date"], math.log(rf / rt)))
    return out


def scan_unexplained_jumps(
    series_fn: Callable[[str], pl.DataFrame | None],
    symbols: list[str],
    claims: pl.DataFrame | None = None,
    config: BreakConfig = BreakConfig(),
) -> pl.DataFrame:
    """Find persistent level shifts that no claim explains.

    `series_fn(symbol)` returns ``date, close`` (raw, as traded), e.g. from
    :func:`split_adjustment_tool.chain.make_close_series_fn`. `claims` (``symbol, date, ratio_from,
    ratio_to``) are the cleaned split claims; a jump matching one is "explained" and not reported.

    Returns one row per unexplained jump: ``symbol, date, prev_date, measured_ratio, z_score,
    reverted_on`` (date the shift reversed within ``revert_window_days``, or null).
    """
    rows = []
    for sym in symbols:
        s = series_fn(sym)
        if s is None or s.height < config.persist_bars * 2 + 2:
            continue
        dates = s["date"].to_list()
        closes = s["close"].to_list()
        rets = [None] + [
            math.log(closes[i] / closes[i - 1]) if closes[i] > 0 and closes[i - 1] > 0 else None
            for i in range(1, len(closes))
        ]
        explained = _implied(claims, sym)
        candidates = []
        for i in range(1, len(closes)):
            r = rets[i]
            if r is None or abs(r) < config.min_jump_ln:
                continue
            trail = [x for x in rets[max(1, i - config.vol_lookback):i] if x is not None]
            sigma = _robust_sigma(trail)
            z = abs(r) / sigma if sigma else float("inf")
            if z < config.z_threshold:
                continue
            before = closes[max(0, i - config.persist_bars):i]
            after = closes[i:i + config.persist_bars]
            if len(before) < 1 or len(after) < 1:
                continue
            mb = sorted(before)[len(before) // 2]
            ma = sorted(after)[len(after) // 2]
            if mb <= 0 or ma <= 0:
                continue
            shift = math.log(ma / mb)
            if abs(shift - r) > config.persist_tol_ln:
                # not a level shift -- but a whole bar off by the jump and back the next bar is a
                # one-day window (e.g. a single day delivered on the wrong basis)
                nxt = rets[i + 1] if i + 1 < len(rets) else None
                if (config.report_one_day_windows and nxt is not None
                        and abs(r + nxt) <= config.persist_tol_ln):
                    candidates.append((i, r, z, i + 1))
                continue
            if any(abs((dates[i] - d).days) <= config.claim_window_days and abs(shift - m) <= config.claim_tol_ln
                   for d, m in explained):
                continue
            candidates.append((i, shift, z, None))
        used = set()
        for i, shift, z, one_day_end in candidates:
            if i in used:
                continue
            reverted = dates[one_day_end] if one_day_end is not None else None
            for j, s2, _, _ in ([] if one_day_end is not None else candidates):
                if j > i and j not in used and (dates[j] - dates[i]).days <= config.revert_window_days \
                        and abs(shift + s2) <= config.persist_tol_ln:
                    reverted = dates[j]
                    used.add(j)
                    break
            rows.append({"symbol": sym, "date": dates[i], "prev_date": dates[i - 1],
                         "measured_ratio": math.exp(shift), "z_score": z, "reverted_on": reverted})
    schema = {"symbol": pl.String, "date": pl.Date, "prev_date": pl.Date, "measured_ratio": pl.Float64,
              "z_score": pl.Float64, "reverted_on": pl.Date}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _witness_move(witness_series_fn, symbol, prev_date, date, n=1) -> float | None:
    """Witness close on `date` vs on `prev_date` (one bar each side: a wider window could reach into
    a neighbouring real event and blur exactly the day being judged)."""
    if witness_series_fn is None:
        return None
    w = witness_series_fn(symbol)
    if w is None or w.height == 0:
        return None
    before = w.filter(pl.col("date") <= prev_date).tail(n)["close"]
    after = w.filter(pl.col("date") >= date).head(n)["close"]
    if len(before) == 0 or len(after) == 0:
        return None
    b, a = float(before.median()), float(after.median())
    if b <= 0 or a <= 0:
        return None
    return math.log(a / b)


def classify_jumps(
    jumps: pl.DataFrame,
    witness_series_fn: Callable[[str], pl.DataFrame | None] | None,
    config: BreakConfig = BreakConfig(),
) -> pl.DataFrame:
    """Add ``classification`` and ``witness_ratio`` to :func:`scan_unexplained_jumps` output by
    asking an independent witness series (``date, close``) what happened across each jump."""
    out = []
    for r in jumps.iter_rows(named=True):
        wm = _witness_move(witness_series_fn, r["symbol"], r["prev_date"], r["date"])
        shift = math.log(r["measured_ratio"])
        if wm is None:
            cls = UNEXPLAINED
        elif abs(wm - shift) <= config.claim_tol_ln:
            cls = REAL_MOVE
        elif abs(wm - shift) > max(config.witness_disagree_ln, 0.5 * abs(shift)):
            cls = PROVIDER_BREAK
        else:
            cls = UNEXPLAINED
        out.append(r | {"witness_ratio": math.exp(wm) if wm is not None else None, "classification": cls})
    schema = dict(jumps.schema) | {"witness_ratio": pl.Float64, "classification": pl.String}
    return pl.DataFrame(out, schema=schema) if out else pl.DataFrame(schema=schema)


def check_claims_against_witness(
    claims: pl.DataFrame,
    witness_series_fn: Callable[[str], pl.DataFrame | None] | None,
    config: BreakConfig = BreakConfig(),
) -> tuple[pl.DataFrame, dict]:
    """Drop claims an independent witness flatly contradicts (no move where the claim implies a
    large one) -- the case where a bogus claim lines up with a provider break and the primary
    tape "confirms" it. Claims the witness cannot speak to are kept. Returns
    ``(kept_claims, stats)`` with ``witness_confirmed`` / ``witness_contradicted`` lists."""
    stats = {"n_input_rows": claims.height, "witness_confirmed": [], "witness_contradicted": [],
             "witness_unavailable": 0}
    if witness_series_fn is None or claims.height == 0:
        stats["witness_unavailable"] = claims.height
        return claims, stats
    keep = []
    for i, r in enumerate(claims.iter_rows(named=True)):
        rf, rt = float(r["ratio_from"]), float(r["ratio_to"])
        if rf <= 0 or rt <= 0:
            keep.append(i)
            continue
        implied = math.log(rf / rt)
        w = witness_series_fn(r["symbol"])
        prev = None
        if w is not None and w.height:
            prior = w.filter(pl.col("date") < r["date"])
            prev = prior["date"][-1] if prior.height else None
        wm = _witness_move(witness_series_fn, r["symbol"], prev, r["date"]) if prev else None
        key = {k: r[k] for k in ("symbol", "date", "ratio_from", "ratio_to")}
        if wm is None:
            stats["witness_unavailable"] += 1
            keep.append(i)
        elif abs(wm - implied) <= config.claim_tol_ln:
            stats["witness_confirmed"].append(key)
            keep.append(i)
        elif abs(implied) >= config.min_jump_ln and abs(wm - implied) > max(config.witness_disagree_ln,
                                                                            0.5 * abs(implied)):
            stats["witness_contradicted"].append(key | {"witness_ratio": round(math.exp(wm), 4)})
        else:
            keep.append(i)
    out = claims.with_row_index("_ri").filter(pl.col("_ri").is_in(keep)).drop("_ri")
    return out, stats
