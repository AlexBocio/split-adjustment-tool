"""The tape-truth collapse chain.

Corporate-action de-duplication, conflict resolution, and phantom refutation -- all
measured against your own raw price tape rather than trusted at face value.

**D1 -- tape is truth.** Corporate-action records (any source: vendor feeds, filings, NLP
extraction) are *claims*. A claim is applied only where your own as-traded price series
confirms it: the right ratio, at the right date, with next-bar persistence. Claims the tape
contradicts are refuted; claims the tape locates elsewhere are snapped to where the tape
actually moves.

Every function here takes and returns a plain ``polars.DataFrame`` with columns
``symbol/date/ratio_from/ratio_to`` (+ optional ``source``/``type``/anything else, passed
through untouched), so each pass can be called standalone -- but :func:`collapse_all` is the
intended entry point. **D3 -- one chain, one chokepoint.** Run the SAME chain everywhere you
consume corporate-action claims; a fix applied to one consumer and not another is the
single most common way this class of bug reappears.

See ``docs/STANDARD.md`` for the full doctrine and the incidents that forged each rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import polars as pl

from tapetruth.locator import find_unique_boundary

__all__ = [
    "ChainConfig",
    "TAPE_CONFIRMED_SOURCE",
    "collapse_duplicate_actions",
    "collapse_same_date_conflicts",
    "collapse_near_date_conflicts",
    "snap_actions_to_tape",
    "refute_phantom_actions",
    "drop_post_coverage_actions",
    "drop_recorded_mislabels",
    "collapse_all",
    "make_close_series_fn",
    "make_gap_fn",
    "make_last_bar_date_fn",
]

#: Rows written by a repair/reconciliation pass AFTER measuring a boundary directly on the
#: tape are, by construction, more authoritative than any externally-sourced claim. If you
#: ever feed tape-confirmed findings back in as actions (see :mod:`tapetruth.reconcile`),
#: tag their `source` with this constant and keep it first in your own
#: ``ChainConfig.source_priority``.
TAPE_CONFIRMED_SOURCE = "tape_confirmed"


@dataclass
class ChainConfig:
    """Every tolerance the collapse chain uses, in one place. Defaults are the values this
    engine was tuned and measured against; override per-field for your own data's
    characteristics -- nothing here is hardcoded into the functions themselves."""

    #: Rows for the SAME (symbol, ratio_from, ratio_to) within this many days of each other
    #: are treated as the SAME real-world event logged by multiple sources and collapsed to
    #: one. Rows further apart than this are treated as genuinely distinct recurring events
    #: (a company that splits the identical ratio again years later) and are never merged.
    #: Discriminator is consecutive-gap clustering within the group, NOT total group span --
    #: so a decade of repeat 2-for-1 splits, each isolated by a large gap, stays intact.
    collapse_window_days: int = 150

    #: Different-RATIO claims on the same symbol within this many days of each other are
    #: treated as one real event mis-logged twice (e.g. a text-parsed announcement disagreeing
    #: with a price-derived feed on the exact ratio), never as two independent splits --
    #: exchanges require advance notice, so two genuine splits this close together is not a
    #: real-world pattern.
    near_date_window_days: int = 10

    #: Best (most authoritative) source name first. A source not in this list sorts last.
    #: Leave as the default (just `tape_confirmed` ranked first) to disable source-rank for
    #: everything else -- ties then break on the measured price gap (if `gap_fn` is
    #: supplied) or on date/informativeness alone.
    source_priority: list[str] = field(default_factory=lambda: [TAPE_CONFIRMED_SOURCE])

    #: A same-date or near-date conflict is resolved by the MEASURED raw price gap only when
    #: the winning candidate's implied ratio is within this log-tolerance of the measurement.
    #: ``ln(1.5)`` absorbs ordinary daily noise while still discriminating a real ratio from
    #: a fabricated one; outside this tolerance the deterministic source-rank rule decides
    #: instead of guessing.
    gap_pick_tolerance_ln: float = math.log(1.5)

    #: :func:`snap_actions_to_tape` and :func:`refute_phantom_actions` only act on claims
    #: implying at least this large a move (log scale). Small ratios (e.g. a 3-for-2 split)
    #: can be confounded by an ordinary day's price move, so neither pass touches them --
    #: better to leave a small-ratio claim alone than mis-snap or wrongly refute it.
    large_ratio_threshold_ln: float = math.log(2.0)

    #: :func:`find_unique_boundary` search window (calendar days) around a claim's recorded
    #: date, and its match tolerance.
    snap_back_days: int = 15
    snap_fwd_days: int = 7
    snap_tolerance_ln: float = math.log(1.35)

    #: :func:`refute_phantom_actions`: the tape counts as "flat" (nothing happened) below
    #: this log-tolerance, and a flat tape only refutes a claim when it disagrees with the
    #: implied ratio by more than this SECOND, larger tolerance (protects small-ratio /
    #: ambiguous cases from being refuted on noise).
    phantom_flat_tolerance_ln: float = math.log(1.2)
    phantom_disagreement_tolerance_ln: float = math.log(2.0)

    #: A price gap is only MEASURABLE when the nearest bars on each side of a claim are within
    #: this many calendar days of it. Across a longer hole (a trading halt, a data outage) the
    #: closes on either side are separated by unknown drift, so the move cannot be attributed
    #: to the claim: the gap is reported as unmeasurable and the claim is KEPT, never refuted.
    #: (CRSP likewise sets adjusted values to missing across a trading gap rather than guess.)
    max_gap_calendar_days: int = 10

    #: ``(symbol, date-ISO-string, ratio_from, ratio_to) -> citation``. Publicly-recorded
    #: events (a spin-off, say) that a vendor mislabeled as a split -- the tape CANNOT
    #: distinguish these from a real split of the claimed ratio (the price genuinely moved
    #: by that amount), so exclusion requires a cited public record, never a tape
    #: measurement. Empty by default; supply your own researched findings.
    recorded_mislabels: dict[tuple[str, str, int, int], str] = field(default_factory=dict)


_CLUSTER_KEYS = ["symbol", "ratio_from", "ratio_to", "_cluster_id"]


def _rank_expr(config: ChainConfig, has_source: bool) -> pl.Expr:
    if not has_source:
        return pl.lit(len(config.source_priority)).cast(pl.Int32)
    rank = {s: i for i, s in enumerate(config.source_priority)}
    unknown = len(config.source_priority)
    return pl.col("source").replace_strict(rank, default=unknown, return_dtype=pl.Int32)


def _gap_pick(group_rows: list[dict], gap_fn, tolerance_ln: float) -> dict | None:
    """Pick the conflict-group row whose implied ratio best matches the MEASURED raw price
    gap. ``gap_fn(symbol, first_date, last_date) -> float | None`` returns measured
    close_after/close_before across the boundary, or None when unavailable.

    A reverse split N:1 multiplies price ~N x overnight (implied m = ratio_from/ratio_to =
    N); a forward 1:N divides it (m = 1/N); a no-op implies m = 1. Keeps the candidate
    minimizing |ln(measured) - ln(implied)|, trusting the pick only when the winner is
    within `tolerance_ln` of the measurement -- otherwise returns None (caller falls back to
    the deterministic source-rank rule rather than guess)."""
    if gap_fn is None or not group_rows:
        return None
    dates = sorted(r["date"] for r in group_rows)
    m = gap_fn(group_rows[0]["symbol"], dates[0], dates[-1])
    if m is None or m <= 0:
        return None
    lm = math.log(m)
    best, best_err = None, None
    for r in group_rows:
        rf, rt = float(r["ratio_from"]), float(r["ratio_to"])
        if rf <= 0 or rt <= 0:
            continue
        err = abs(lm - math.log(rf / rt))
        if best_err is None or err < best_err:
            best, best_err = r, err
    if best is not None and best_err is not None and best_err <= tolerance_ln:
        return best
    return None


def _tape_confirms(gap_fn, symbol, date, rf, rt, tolerance_ln) -> bool:
    """True when the raw tape shows a move matching this claim's implied ratio right at its
    own date (closes just before vs just after)."""
    if gap_fn is None or rf <= 0 or rt <= 0:
        return False
    m = gap_fn(symbol, date, date)
    if m is None or m <= 0:
        return False
    return abs(math.log(m) - math.log(rf / rt)) <= tolerance_ln


def _cluster_ids(df: pl.DataFrame, config: ChainConfig, gap_fn) -> list[int]:
    """Cluster rows (already sorted by symbol, ratio_from, ratio_to, date) into same-event
    groups. Anchor-based, NOT consecutive-gap chaining: a row joins the current cluster only if
    it is within ``collapse_window_days`` of the cluster's FIRST row, so 0/140/280-day rows
    can never chain into one event. With a `gap_fn`, a cluster is further split wherever two
    rows each carry their OWN tape-confirmed boundary more than ``near_date_window_days``
    apart -- two real splits of the same ratio (serial reverse splits in a micro-cap) are two
    events even when they are close together."""
    ids, cid = [], 0
    rows = df.select(["symbol", "ratio_from", "ratio_to", "date"]).to_dicts()
    i = 0
    while i < len(rows):
        key = (rows[i]["symbol"], rows[i]["ratio_from"], rows[i]["ratio_to"])
        start = rows[i]["date"]
        j = i
        while (j < len(rows) and (rows[j]["symbol"], rows[j]["ratio_from"], rows[j]["ratio_to"]) == key
               and (rows[j]["date"] - start).days <= config.collapse_window_days):
            j += 1
        block = rows[i:j]
        anchors = []  # dates of tape-confirmed distinct events inside this block
        for r in block:
            if _tape_confirms(gap_fn, r["symbol"], r["date"], float(r["ratio_from"]),
                              float(r["ratio_to"]), config.gap_pick_tolerance_ln):
                if not anchors or (r["date"] - anchors[-1]).days > config.near_date_window_days:
                    anchors.append(r["date"])
        if len(anchors) <= 1:
            ids.extend([cid] * len(block))
            cid += 1
        else:
            for r in block:
                nearest = min(range(len(anchors)), key=lambda k: abs((r["date"] - anchors[k]).days))
                ids.append(cid + nearest)
            cid += len(anchors)
        i = j
    return ids


def collapse_duplicate_actions(
    actions: pl.DataFrame, config: ChainConfig = ChainConfig(), gap_fn=None
) -> tuple[pl.DataFrame, dict]:
    """Collapse same-event multi-source duplicate rows within (symbol, ratio_from, ratio_to)
    groups, using consecutive-date-gap clustering (window=``config.collapse_window_days``) --
    NOT total group span -- so a genuine multi-decade recurring event (each real occurrence
    isolated by a large gap) is never merged across events.

    `actions` must have columns symbol/date/ratio_from/ratio_to. `source` is optional -- if
    absent, canonical-pick falls back to "latest date in the cluster" only. Any OTHER columns
    present (`type`, ...) are preserved from the canonical row.

    Deterministic and idempotent: re-running on an already-collapsed frame is a no-op.

    Returns ``(collapsed_df, stats)`` where stats has n_input_rows / n_output_rows /
    n_rows_collapsed / n_clusters_multi_row / n_clusters_total / window_days.
    """
    empty_stats = {
        "n_input_rows": 0, "n_output_rows": 0, "n_rows_collapsed": 0,
        "n_clusters_multi_row": 0, "n_clusters_total": 0, "window_days": config.collapse_window_days,
    }
    if actions.height == 0:
        return actions, empty_stats

    has_source = "source" in actions.columns
    df = actions.sort(["symbol", "ratio_from", "ratio_to", "date"])

    df = df.with_columns(pl.Series("_cluster_id", _cluster_ids(df, config, gap_fn), dtype=pl.Int64))
    df = df.with_columns(_rank_expr(config, has_source).alias("_source_rank"))

    n_input = df.height
    n_clusters_total = df.select(_CLUSTER_KEYS).unique().height
    n_clusters_multi_row = (
        df.group_by(_CLUSTER_KEYS).agg(pl.len().alias("n")).filter(pl.col("n") > 1).height
    )

    # Canonical row per cluster: lowest (best) source rank first, tie-broken by the LATEST
    # date in the cluster (closer to a true effective date than an early announcement).
    sort_cols = _CLUSTER_KEYS + ["_source_rank", "date"]
    sort_desc = [False, False, False, False, False, True]
    df = df.sort(sort_cols, descending=sort_desc)
    canonical = df.group_by(_CLUSTER_KEYS).first()
    canonical = canonical.drop(["_cluster_id", "_source_rank"])
    canonical = canonical.sort(["symbol", "date"])

    stats = {
        "n_input_rows": n_input, "n_output_rows": canonical.height,
        "n_rows_collapsed": n_input - canonical.height,
        "n_clusters_multi_row": n_clusters_multi_row, "n_clusters_total": n_clusters_total,
        "window_days": config.collapse_window_days,
    }
    return canonical, stats


def collapse_same_date_conflicts(
    actions: pl.DataFrame, config: ChainConfig = ChainConfig(), gap_fn=None
) -> tuple[pl.DataFrame, dict]:
    """Collapse MULTIPLE canonical actions on the SAME (symbol, date) down to ONE.

    :func:`collapse_duplicate_actions` groups by (symbol, ratio_from, ratio_to), so it
    cannot catch two contradictory canonical actions on the same date with DIFFERENT ratios
    -- e.g. a claimed ``split 1:4`` and a claimed ``reverse_split 4:1`` on the identical day.
    Those are INVERSES: their implied ratios multiply to ~1.0, so if left as two separate
    rows the real event cancels to a no-op and the split-day price jump goes unadjusted. Two
    sources reporting the SAME DATE is the SAME economic event seen twice, never two separate
    splits.

    Rule (deterministic, idempotent): within a (symbol, date) group with >1 row --

    1. drop no-op rows (``ratio_from == ratio_to``, i.e. factor 1.0) -- a source reporting
       "no split" on a date another source flags a real split carries no information;
    2. among the survivors, keep the single best source-rank row (price/tape-derived sources
       first -- they derive the ratio from the actual price gap, so they win a
       direction/ratio conflict against a text-parsed claim), tie-broken by the MORE
       informative ratio (larger |ln|);
    3. if ALL rows were no-ops, keep exactly one (a genuine factor-1.0 date).

    A supplied ``gap_fn`` OVERRIDES the default pick with the candidate whose ratio matches
    the MEASURED raw price gap (data beats heuristics) -- see :func:`_gap_pick`.
    """
    empty_stats = {"n_input_rows": actions.height, "n_output_rows": actions.height,
                   "n_rows_collapsed": 0, "n_conflict_dates": 0}
    if actions.height == 0:
        return actions, empty_stats

    has_source = "source" in actions.columns
    df = actions.with_row_index("_ri").with_columns([
        (pl.col("ratio_to").cast(pl.Float64) / pl.col("ratio_from").cast(pl.Float64)).alias("_sr"),
        (pl.col("ratio_from") == pl.col("ratio_to")).alias("_noop"),
    ])
    df = df.with_columns(pl.col("_sr").log().abs().alias("_info"))
    df = df.with_columns(_rank_expr(config, has_source).alias("_rank"))

    df = df.sort(["symbol", "date", "_noop", "_rank", "_info"],
                 descending=[False, False, False, False, True])
    keep = {int(r["_ri"]) for r in
            df.group_by(["symbol", "date"], maintain_order=True).first().iter_rows(named=True)}

    n_conflict = 0
    n_gap_resolved = 0
    if df.height != df.select(["symbol", "date"]).unique().height:
        multi = (df.group_by(["symbol", "date"]).len().filter(pl.col("len") > 1)
                   .select(["symbol", "date"]))
        n_conflict = multi.height
        for g in multi.iter_rows(named=True):
            rows = df.filter((pl.col("symbol") == g["symbol"]) &
                             (pl.col("date") == g["date"])).to_dicts()
            pick = _gap_pick(rows, gap_fn, config.gap_pick_tolerance_ln)
            if pick is not None:
                for r in rows:
                    keep.discard(int(r["_ri"]))
                keep.add(int(pick["_ri"]))
                n_gap_resolved += 1

    canonical = (actions.with_row_index("_ri").filter(pl.col("_ri").is_in(list(keep)))
                 .drop("_ri").sort(["symbol", "date"]))
    stats = {"n_input_rows": actions.height, "n_output_rows": canonical.height,
             "n_rows_collapsed": actions.height - canonical.height,
             "n_conflict_dates": n_conflict, "n_gap_resolved": n_gap_resolved}
    return canonical, stats


def collapse_near_date_conflicts(
    actions: pl.DataFrame, config: ChainConfig = ChainConfig(), gap_fn=None
) -> tuple[pl.DataFrame, dict]:
    """Collapse DIFFERENT-ratio canonical actions within ``config.near_date_window_days`` of
    each other on the same symbol down to the single best-source row.

    Neither prior pass catches this: :func:`collapse_duplicate_actions` groups by (symbol,
    ratio_from, ratio_to) so different ratios never cluster; :func:`collapse_same_date_conflicts`
    requires the EXACT same date. A real-world example of the shape: a text-parsed filing
    reports ``reverse_split 10:1`` a few days before a price-derived feed reports
    ``reverse_split 30:1`` for what is actually the same event -- the wrong ratio, logged a
    few days apart, compounds with the right one into a fictitious combined factor. Two
    different REAL splits cannot legitimately execute within days of each other (exchanges
    require advance notice), so a near-date, different-ratio pair on one symbol is one event
    seen twice, not two events.

    Rule (deterministic, idempotent): cluster consecutive actions per symbol where the gap
    to the previous action is <= ``near_date_window_days``; within a multi-row cluster keep
    the single best source-rank row (price/tape-derived sources first), tie-broken by the
    LATEST date (effective date beats an early announcement). A supplied ``gap_fn``
    overrides with the measured-price-gap match (see :func:`_gap_pick`).
    """
    stats0 = {"n_input_rows": actions.height, "n_output_rows": actions.height,
              "n_rows_collapsed": 0, "n_conflict_clusters": 0,
              "window_days": config.near_date_window_days}
    if actions.height == 0:
        return actions, stats0

    has_source = "source" in actions.columns
    df = actions.with_row_index("_ri").sort(["symbol", "date"]).with_columns(
        (pl.col("date").diff().dt.total_days().over("symbol")).alias("_gap"))
    df = df.with_columns(
        ((pl.col("_gap").is_null()) | (pl.col("_gap") > config.near_date_window_days))
        .cast(pl.Int32).alias("_new"))
    df = df.with_columns(pl.col("_new").cum_sum().over("symbol").alias("_cl"))
    df = df.with_columns((pl.col("ratio_from") == pl.col("ratio_to")).alias("_noop"))
    df = df.with_columns(_rank_expr(config, has_source).alias("_rank"))

    df = df.sort(["symbol", "_cl", "_noop", "_rank", "date"],
                 descending=[False, False, False, False, True])
    keep = {int(r["_ri"]) for r in
            df.group_by(["symbol", "_cl"], maintain_order=True).first().iter_rows(named=True)}

    multi = df.group_by(["symbol", "_cl"]).len().filter(pl.col("len") > 1).select(["symbol", "_cl"])
    n_conflict = multi.height
    n_gap_resolved = 0
    for g in multi.iter_rows(named=True):
        rows = df.filter((pl.col("symbol") == g["symbol"]) & (pl.col("_cl") == g["_cl"])).to_dicts()
        pick = _gap_pick(rows, gap_fn, config.gap_pick_tolerance_ln)
        if pick is not None:
            for r in rows:
                keep.discard(int(r["_ri"]))
            keep.add(int(pick["_ri"]))
            n_gap_resolved += 1

    out = (actions.with_row_index("_ri").filter(pl.col("_ri").is_in(list(keep)))
           .drop("_ri").sort(["symbol", "date"]))
    stats = {"n_input_rows": actions.height, "n_output_rows": out.height,
             "n_rows_collapsed": actions.height - out.height,
             "n_conflict_clusters": n_conflict, "n_gap_resolved": n_gap_resolved,
             "window_days": config.near_date_window_days}
    return out, stats


def make_close_series_fn(bars) -> Callable[[str], pl.DataFrame | None]:
    """Cached per-symbol close-series loader from a :class:`~tapetruth.providers.BarProvider`
    -- shared by :func:`make_gap_fn`, :func:`make_last_bar_date_fn`, and
    :func:`snap_actions_to_tape`'s boundary search so all three read the same tape through
    one cache. Construct once, pass the SAME `series_fn` into all three."""
    cache: dict[str, pl.DataFrame | None] = {}

    def series_fn(symbol: str) -> pl.DataFrame | None:
        if symbol not in cache:
            df = bars.get_bars(symbol)
            cache[symbol] = (df.select(["date", "close"]).sort("date")
                             if df is not None and df.height > 0 else None)
        return cache[symbol]

    return series_fn


def make_gap_fn(bars=None, series_fn=None, max_gap_calendar_days: int | None = None) -> Callable:
    """Builds a `gap_fn` measuring the raw price gap across a claim's date range:
    ``(median close of up to 3 trading days AFTER last_date) / (median close of up to 3
    trading days STRICTLY BEFORE first_date)``. Medians damp single-day noise. Pass
    `series_fn` (from :func:`make_close_series_fn`) to share a cache with
    :func:`make_last_bar_date_fn` and the snap pass; `bars` is a convenience if you don't
    already have one. Returns `None` (rather than guessing) for symbols/dates without
    coverage."""
    _series = series_fn if series_fn is not None else make_close_series_fn(bars)
    limit = max_gap_calendar_days if max_gap_calendar_days is not None else ChainConfig().max_gap_calendar_days

    def gap_fn(symbol: str, first_date, last_date):
        s = _series(symbol)
        if s is None or s.height == 0:
            return None
        bdf = s.filter(pl.col("date") < first_date).tail(3)
        adf = s.filter(pl.col("date") > last_date).head(3)
        if bdf.height == 0 or adf.height == 0:
            return None
        # Unmeasurable across a hole (halt / outage): drift on the far side of the hole would
        # be mistaken for (or would cancel) the event's own move.
        if ((first_date - bdf["date"][-1]).days > limit
                or (adf["date"][0] - last_date).days > limit):
            return None
        before, after = bdf["close"], adf["close"]
        b, a = float(before.median()), float(after.median())
        if b <= 0 or a <= 0:
            return None
        return a / b

    return gap_fn


def make_last_bar_date_fn(bars=None, series_fn=None) -> Callable:
    """``fn(symbol) -> the symbol's final bar date, or None`` -- shares
    :func:`make_close_series_fn`'s cache when the same `series_fn` is passed to both this and
    :func:`make_gap_fn`."""
    _series = series_fn if series_fn is not None else make_close_series_fn(bars)

    def last_bar_date_fn(symbol: str):
        s = _series(symbol)
        if s is None or s.height == 0:
            return None
        return s["date"].max()

    return last_bar_date_fn


def snap_actions_to_tape(
    actions: pl.DataFrame, config: ChainConfig = ChainConfig(), series_fn=None, gap_fn=None
) -> tuple[pl.DataFrame, dict]:
    """Snap wrong-dated LARGE actions to their unique tape boundary.

    Some feeds carry the right ratio but the wrong date (an announcement date instead of the
    market-effective date; a one-day offset from a timezone or "arrived pre-adjusted"
    quirk). When the recorded date carries no measurable gap, every downstream
    date-anchored consumer misreads the situation -- a phantom-refuter kills a real-but-
    misdated split, a vintage classifier concludes "already adjusted," a reconciler flags a
    false boundary mismatch. Three symptoms, one cause.

    For each action with ``|ln(ratio_from/ratio_to)| >= config.large_ratio_threshold_ln``
    (small ratios are left alone -- single-day noise can fake a small boundary), search
    :func:`~tapetruth.locator.find_unique_boundary` in
    ``[date - config.snap_back_days, date + config.snap_fwd_days]``; on a unique, persistent
    hit, REWRITE the action's date to the tape date. This is a load-time view correction --
    if you're reading from a database, nothing there is mutated; you get a corrected frame
    back. Runs BEFORE :func:`refute_phantom_actions` in :func:`collapse_all` so misdated real
    events get snapped rather than wrongly refuted.
    """
    stats = {"n_input_rows": actions.height, "n_snapped": 0, "snapped": []}
    if series_fn is None or gap_fn is None or actions.height == 0:
        return actions, stats
    new_dates, snapped = {}, []
    for i, r in enumerate(actions.iter_rows(named=True)):
        rf, rt = float(r["ratio_from"]), float(r["ratio_to"])
        if rf <= 0 or rt <= 0:
            continue
        implied = rf / rt
        if abs(math.log(implied)) < config.large_ratio_threshold_ln:
            continue
        hit = find_unique_boundary(
            series_fn(r["symbol"]), r["date"], implied,
            back_days=config.snap_back_days, fwd_days=config.snap_fwd_days,
            tol_ln=config.snap_tolerance_ln,
        )
        if hit is None or hit["tape_date"] == r["date"]:
            continue
        new_dates[i] = hit["tape_date"]
        snapped.append({k: r[k] for k in ("symbol", "date", "ratio_from", "ratio_to")}
                       | {"tape_date": str(hit["tape_date"]), "measured_1d": hit["measured_1d"]})
    if not new_dates:
        return actions, stats
    dates_series = actions["date"].to_list()
    for i, d in new_dates.items():
        dates_series[i] = d
    out = actions.with_columns(pl.Series("date", dates_series, dtype=pl.Date))
    stats.update({"n_snapped": len(snapped), "snapped": snapped})
    return out, stats


def refute_phantom_actions(
    actions: pl.DataFrame, config: ChainConfig = ChainConfig(), gap_fn=None, vintage_fn=None
) -> tuple[pl.DataFrame, dict]:
    """Drop LONE phantom actions the tape refutes.

    Rule (conservative, tape-is-truth): for each action where `gap_fn` yields a measured
    boundary move, refute IFF the action implies a LARGE move
    (``|ln(implied)| >= config.large_ratio_threshold_ln``) AND the tape shows essentially
    none (``|ln(measured)| < config.phantom_flat_tolerance_ln``) AND the disagreement is
    gross (``|ln(measured/implied)| > config.phantom_disagreement_tolerance_ln``). Real
    events keep themselves (a genuine 10:1 split measures ~10x, matching its own implied
    ratio). Actions with no bar coverage at all (pre-history, unlisted) are NEVER refuted --
    absence of evidence keeps the action.

    `vintage_fn`, if supplied: ``fn(symbol, date) -> str | None``. When it returns
    ``"ADJUSTED"`` for this (symbol, date), the action is kept unconditionally -- this
    protects a REAL action whose price series arrived already pre-adjusted at the source (no
    tape gap exists anywhere in your data, not because the event is fake, but because a prior
    vendor already applied it before you ever saw the series). tapetruth ships the mechanism
    for this guard but no bundled vintage classifier -- you supply `vintage_fn` from
    whatever detects pre-adjusted vintages in your own pipeline, or omit it and lose this one
    protection layer.

    Refuted rows are returned in ``stats["refuted"]`` for logging / rebuild-targeting.
    """
    stats = {"n_input_rows": actions.height, "n_refuted": 0, "refuted": [],
             "n_unmeasurable_kept": 0, "unmeasurable_kept": []}
    if gap_fn is None or actions.height == 0:
        return actions, stats
    keep_idx, refuted, unmeasurable = [], [], []
    for i, r in enumerate(actions.iter_rows(named=True)):
        rf, rt = float(r["ratio_from"]), float(r["ratio_to"])
        if rf <= 0 or rt <= 0:
            keep_idx.append(i); continue
        implied = math.log(rf / rt)
        if abs(implied) < config.large_ratio_threshold_ln:
            keep_idx.append(i); continue
        if vintage_fn is not None and vintage_fn(r["symbol"], r["date"]) == "ADJUSTED":
            keep_idx.append(i); continue
        m = gap_fn(r["symbol"], r["date"], r["date"])
        if m is None or m <= 0:
            # absence of evidence keeps the action -- logged so callers can flag it
            unmeasurable.append({k: r[k] for k in ("symbol", "date", "ratio_from", "ratio_to")})
            keep_idx.append(i); continue
        lm = math.log(m)
        if (abs(lm) < config.phantom_flat_tolerance_ln
                and abs(lm - implied) > config.phantom_disagreement_tolerance_ln):
            refuted.append({k: r[k] for k in ("symbol", "date", "ratio_from", "ratio_to")}
                           | {"measured_gap": round(m, 4)})
        else:
            keep_idx.append(i)
    out = actions.with_row_index("_ri").filter(pl.col("_ri").is_in(keep_idx)).drop("_ri")
    stats.update({"n_refuted": len(refuted), "refuted": refuted,
                  "n_unmeasurable_kept": len(unmeasurable), "unmeasurable_kept": unmeasurable})
    return out, stats


def drop_recorded_mislabels(
    actions: pl.DataFrame, config: ChainConfig = ChainConfig()
) -> tuple[pl.DataFrame, dict]:
    """Drop rows matching ``config.recorded_mislabels`` -- a cited public record excluding a
    claim the tape genuinely CONFIRMS but that isn't really the claimed event type (the
    canonical shape: a spin-off's value-separation price drop looks, on the tape alone,
    identical to a real split of the same ratio -- only a citation disambiguates, never a
    measurement). Key: ``(symbol, date.isoformat(), ratio_from, ratio_to)``."""
    stats = {"n_input_rows": actions.height, "n_dropped": 0, "dropped": []}
    if actions.height == 0 or not config.recorded_mislabels:
        return actions, stats
    keep_idx, dropped = [], []
    for i, r in enumerate(actions.iter_rows(named=True)):
        key = (r["symbol"], str(r["date"]), int(r["ratio_from"]), int(r["ratio_to"]))
        if key in config.recorded_mislabels:
            dropped.append({k: r[k] for k in ("symbol", "date", "ratio_from", "ratio_to")}
                           | {"citation": config.recorded_mislabels[key]})
        else:
            keep_idx.append(i)
    out = actions.with_row_index("_ri").filter(pl.col("_ri").is_in(keep_idx)).drop("_ri")
    stats.update({"n_dropped": len(dropped), "dropped": dropped})
    return out, stats


def drop_post_coverage_actions(actions: pl.DataFrame, last_date_fn=None) -> tuple[pl.DataFrame, dict]:
    """Drop actions dated AFTER the symbol's final bar.

    An adjustment factor exists to make a series continuous ACROSS a boundary. When the
    claimed date lies beyond the last bar you have, there is no boundary inside your data --
    applying the factor anyway rescales EVERY bar the symbol ever traded, not just the bars
    after a real boundary. This is wrong regardless of whether some later real-world event
    actually happened; you simply don't have the data to place it. The same rule is correct
    for a live, currently-trading symbol too: an announced-but-not-yet-effective future
    action must not touch history until the first post-event bar actually lands (the check
    self-heals on the next run once the boundary enters your data). No margin: drop iff
    ``action date > last_bar_date``; symbols with no coverage at all (``last_date_fn`` returns
    None) are never dropped.

    MUST run BEFORE :func:`collapse_duplicate_actions` in the chain -- a post-coverage row
    inside a duplicate cluster could otherwise win the latest-date tiebreak and evict the
    real in-coverage row, losing the event entirely.
    """
    stats = {"n_input_rows": actions.height, "n_dropped": 0, "dropped": []}
    if last_date_fn is None or actions.height == 0:
        return actions, stats
    keep_idx, dropped = [], []
    for i, r in enumerate(actions.iter_rows(named=True)):
        last_bar = last_date_fn(r["symbol"])
        if last_bar is not None and r["date"] > last_bar:
            dropped.append({k: r[k] for k in ("symbol", "date", "ratio_from", "ratio_to")}
                           | {"last_bar_date": str(last_bar)})
        else:
            keep_idx.append(i)
    out = actions.with_row_index("_ri").filter(pl.col("_ri").is_in(keep_idx)).drop("_ri")
    stats.update({"n_dropped": len(dropped), "dropped": dropped})
    return out, stats


def collapse_all(
    actions: pl.DataFrame,
    config: ChainConfig = ChainConfig(),
    gap_fn=None,
    last_date_fn=None,
    series_fn=None,
    vintage_fn=None,
) -> tuple[pl.DataFrame, dict]:
    """THE canonical dedup chain -- every consumer of your corporate-action claims should go
    through this one entry point (D3: a fix that lives in only one caller does not exist).

    0. :func:`drop_recorded_mislabels` -- publicly-cited exclusions (spin-off class)
    1. :func:`drop_post_coverage_actions` -- claims dated after the final bar; runs before
       clustering so a post-coverage row can never win a duplicate-cluster tiebreak
    2. :func:`collapse_duplicate_actions` -- same-event multi-source rows
    3. :func:`collapse_same_date_conflicts` -- contradictory same-date rows
    4. :func:`collapse_near_date_conflicts` -- different-ratio rows within days of each other
    5. :func:`snap_actions_to_tape` -- rewrite wrong-dated large actions onto the unique tape
       boundary; every downstream date-anchored measurement then reads the right day
    6. :func:`refute_phantom_actions` -- lone tape-refuted phantoms; vintage-aware (never
       refutes a real event on a pre-adjusted vintage)

    Pass ``gap_fn=make_gap_fn(series_fn=...)`` so conflicts resolve by the MEASURED price gap
    (data beats source-rank heuristics), ``last_date_fn=make_last_bar_date_fn(series_fn=...)``
    to arm the post-coverage pass, and construct all three `_fn` helpers from ONE
    :func:`make_close_series_fn` call to share its file/DataFrame cache. ``vintage_fn`` is
    optional (see :func:`refute_phantom_actions`).

    Returns ``(collapsed, {stage: stats})``.
    """
    c0, s0 = drop_recorded_mislabels(actions, config=config)
    c05, s05 = drop_post_coverage_actions(c0, last_date_fn=last_date_fn)
    c1, s1 = collapse_duplicate_actions(c05, config=config, gap_fn=gap_fn)
    c2, s2 = collapse_same_date_conflicts(c1, config=config, gap_fn=gap_fn)
    c3, s3 = collapse_near_date_conflicts(c2, config=config, gap_fn=gap_fn)
    c35, s35 = snap_actions_to_tape(c3, config=config, series_fn=series_fn, gap_fn=gap_fn)
    c4, s4 = refute_phantom_actions(c35, config=config, gap_fn=gap_fn, vintage_fn=vintage_fn)
    return c4, {
        "mislabel": s0, "post_coverage": s05, "duplicate": s1, "same_date": s2,
        "near_date": s3, "snap": s35, "phantom": s4,
    }
