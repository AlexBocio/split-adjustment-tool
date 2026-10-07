"""The synthetic defect gauntlet -- tapetruth's own benchmark.

Generates a universe of synthetic OHLCV symbols with KNOWN, PLANTED defects across the
twelve classes the engine is designed around, runs the full chain + guard over them, and
scores how many were caught, how many legitimate look-alikes were correctly left alone, and
where the engine's documented, honest limitations are. No real ticker, no real price ever
appears here or anywhere else in this package -- every symbol is synthetic
(``TEST<CLASS><NN>``), generated from a geometric random walk with realistic daily
volatility, gaps, wicks, and log-normal volume.

This doubles as the public benchmark: ``python -m tapetruth.demo`` runs it end-to-end and
prints a scorecard. Because the defects are planted with known ground truth, this is a
BETTER demonstration than any real dataset could be -- detection and repair are measurable,
not asserted.

Three categories, sixteen classes
---------------------------------
(Four hard real-world classes were added in v0.1.1: `serial_reverse_split`,
`halted_reverse_split`, `real_small_ratio_split` (legitimate) and `small_ratio_phantom`
(known gap). `spinoff_mislabel` and `mixed_vintage_ohl` are CONFIGURED classes: the gauntlet
hands the engine their answer key via `recorded_mislabels` / `vintage_fn`, so their 100% shows
the hook works, not that anything was detected unaided.)

**defect** (8 classes) -- something is genuinely wrong; the engine should catch and fix it.
`duplicate_actions`, `same_date_contradiction`, `near_date_fake`, `post_coverage_action`,
`spinoff_mislabel`, `misdated_action`, `phantom_action`, `in_band_phantom_dip`.

**legitimate** (3 classes) -- looks like a defect pattern-wise, but is actually real; the
engine should leave it alone. `extreme_ratio_real_split`, `mixed_vintage_ohl`,
`envelope_wick`.

**known_gap** (1 class) -- a real defect the current engine does NOT catch, by design, and
says so honestly (see ``docs/STANDARD.md`` "Known gaps"). `bad_close_vspike`. Reported
separately, never folded into the headline detection rate -- claiming credit for a class
you don't catch would defeat the entire honest-classes philosophy this engine is built on.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass, field
from typing import Callable

import polars as pl

from tapetruth.chain import (
    TAPE_CONFIRMED_SOURCE,
    ChainConfig,
    collapse_all,
    make_close_series_fn,
    make_gap_fn,
    make_last_bar_date_fn,
    refute_phantom_actions,
)
from tapetruth.guard import GuardConfig, apply_bad_print_guard
from tapetruth.providers import ACTION_SCHEMA, InMemoryBarProvider, InMemoryTruthProvider

__all__ = [
    "GauntletConfig",
    "PlantResult",
    "GauntletUniverse",
    "GauntletResult",
    "ClassScore",
    "Scorecard",
    "DEFECT_CLASSES",
    "build_gauntlet_universe",
    "run_gauntlet",
    "score_run",
    "demonstrate_vintage_awareness",
]

#: class name -> category ("defect" | "legitimate" | "known_gap"). Informational export;
#: `score_run` derives category from each planted instance directly, not from this table.
DEFECT_CLASSES: dict[str, str] = {
    "duplicate_actions": "defect",
    "same_date_contradiction": "defect",
    "near_date_fake": "defect",
    "post_coverage_action": "defect",
    "spinoff_mislabel": "defect",
    "misdated_action": "defect",
    "phantom_action": "defect",
    "in_band_phantom_dip": "defect",
    "extreme_ratio_real_split": "legitimate",
    "mixed_vintage_ohl": "legitimate",
    "envelope_wick": "legitimate",
    "bad_close_vspike": "known_gap",
    "serial_reverse_split": "legitimate",
    "halted_reverse_split": "legitimate",
    "real_small_ratio_split": "legitimate",
    "small_ratio_phantom": "known_gap",
}

#: Classes whose outcome is decided by side-data the gauntlet itself supplies (an answer key):
#: reported as CONFIGURED, not as independent detection.
CONFIGURED_CLASSES = {"spinoff_mislabel", "mixed_vintage_ohl"}

#: Classes that intentionally carry abnormal OHLC (as opposed to abnormal ACTIONS) --
#: excluded from the clean-symbol false-positive denominator (they're supposed to look odd).
_GUARD_DEFECT_CLASSES = {"envelope_wick", "in_band_phantom_dip", "bad_close_vspike"}

#: The realistic "one source derives its date from the price itself" pattern documented in
#: docs/STANDARD.md rule 1 -- in every real duplicate-action cluster this engine was tuned
#: against, exactly one source (a price-derived feed) carries the true market-effective
#: date, while the others (announcements, filings) can be off by days to months. Ranking
#: this source in `run_gauntlet`'s ChainConfig is what makes the duplicate-collapse
#: canonical-pick land on the correctly-dated row, matching the real-world pattern the
#: engine is designed around -- an all-unranked-sources cluster (no source ever carries a
#: trustworthy date) is a harder, less realistic scenario than what source_priority exists
#: to solve.
_PRICE_DERIVED_SOURCE = "price_feed"


@dataclass
class GauntletConfig:
    """Universe-generation knobs. Defaults produce a universe that runs end-to-end in a few
    seconds while still giving each of the 12 classes enough instances to be statistically
    meaningful (not a single coin-flip)."""

    n_instances_per_class: int = 20
    n_clean_symbols: int = 50
    n_trading_days: int = 504  # ~2 years
    start_date: dt.date = dt.date(2021, 1, 4)
    seed: int = 1337
    base_price_range: tuple[float, float] = (5.0, 400.0)
    daily_vol: float = 0.02  # ~2% daily stdev -- realistic equity volatility


@dataclass
class PlantResult:
    """One planted instance: its synthetic bars, any action claims to add to the universe,
    any truth/vintage/mislabel side-data it needs, and the ground-truth `check` that decides
    whether the engine handled it correctly."""

    symbol: str
    bars: pl.DataFrame
    actions: list[dict]
    truth: dict
    vintage: dict
    mislabels: dict
    defect_class: str
    category: str  # "defect" | "legitimate" | "known_gap"
    check: Callable[["GauntletResult"], bool]
    description: str


# ---------------------------------------------------------------------------------------
# Synthetic series generation
# ---------------------------------------------------------------------------------------


def _trading_dates(start: dt.date, n: int) -> list[dt.date]:
    """`n` weekday (Mon-Fri) dates starting at `start`, rolled forward to the next weekday."""
    dates = []
    d = start
    while d.weekday() > 4:
        d += dt.timedelta(days=1)
    while len(dates) < n:
        dates.append(d)
        d += dt.timedelta(days=1)
        while d.weekday() > 4:
            d += dt.timedelta(days=1)
    return dates


def _generate_clean_bars(
    n_days: int, start: dt.date, base_price: float, daily_vol: float, rng: random.Random
) -> pl.DataFrame:
    """Geometric random walk OHLCV: log-normal returns, a gap-driven open, wicks scaled off
    the candle body, log-normal volume. No corporate-action discontinuity -- callers inject
    those with :func:`_inject_split`."""
    dates = _trading_dates(start, n_days)
    closes = [max(0.01, base_price)]
    for _ in range(1, n_days):
        ret = rng.gauss(0.0002, daily_vol)
        closes.append(max(0.01, closes[-1] * math.exp(ret)))

    opens, highs, lows, vols = [], [], [], []
    prev_close = closes[0]
    for c in closes:
        o = max(0.01, prev_close * math.exp(rng.gauss(0.0, daily_vol * 0.3)))
        body_hi, body_lo = max(o, c), min(o, c)
        h = body_hi * (1.0 + abs(rng.gauss(0.0, daily_vol * 0.4)))
        l = max(0.01, body_lo * (1.0 - abs(rng.gauss(0.0, daily_vol * 0.4))))
        v = int(max(1_000, rng.lognormvariate(math.log(800_000), 0.6)))
        opens.append(o); highs.append(h); lows.append(l); vols.append(v)
        prev_close = c

    return pl.DataFrame({
        "date": dates, "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
    }).with_columns(pl.col("date").cast(pl.Date))


def _inject_split(bars: pl.DataFrame, split_date: dt.date, ratio_from: int, ratio_to: int) -> pl.DataFrame:
    """Mutates a clean synthetic RAW series to carry a REAL split discontinuity at
    `split_date` -- this is what an as-traded tape actually looks like at a genuine split
    (D1): price divides (forward) or multiplies (reverse) for every bar ON OR AFTER the
    split date; volume moves inversely."""
    price_mult = ratio_from / ratio_to
    vol_mult = ratio_to / ratio_from
    out = bars.with_columns([
        pl.when(pl.col("date") >= split_date).then(pl.col(c) * price_mult).otherwise(pl.col(c)).alias(c)
        for c in ("open", "high", "low", "close")
    ])
    return out.with_columns(
        pl.when(pl.col("date") >= split_date)
          .then((pl.col("volume").cast(pl.Float64) * vol_mult).round(0).cast(pl.Int64))
          .otherwise(pl.col("volume")).alias("volume")
    )


def _mid_index(cfg: GauntletConfig, rng: random.Random) -> int:
    n = cfg.n_trading_days
    return rng.randint(n // 3, 2 * n // 3)


# ---------------------------------------------------------------------------------------
# The twelve planters
# ---------------------------------------------------------------------------------------


def _plant_duplicate_actions(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTDUP{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    split_date = bars["date"][_mid_index(cfg, rng)]
    ratio_from, ratio_to = 1, rng.choice([2, 3, 4, 5, 10])
    bars = _inject_split(bars, split_date, ratio_from, ratio_to)

    # One source is price-derived and carries the TRUE effective date (the realistic
    # pattern -- see docs/STANDARD.md rule 1); the other two are announcement/filing-style
    # claims that can land days to weeks off the true date. All three claim the same ratio.
    other_offsets = rng.sample([o for o in range(-60, 61) if o != 0], 2)
    actions = [
        {"symbol": symbol, "date": split_date, "ratio_from": ratio_from, "ratio_to": ratio_to,
         "source": _PRICE_DERIVED_SOURCE, "type": "split"},
        {"symbol": symbol, "date": split_date + dt.timedelta(days=other_offsets[0]),
         "ratio_from": ratio_from, "ratio_to": ratio_to, "source": "filing_feed_a", "type": "split"},
        {"symbol": symbol, "date": split_date + dt.timedelta(days=other_offsets[1]),
         "ratio_from": ratio_from, "ratio_to": ratio_to, "source": "filing_feed_b", "type": "split"},
    ]

    def check(result: "GauntletResult") -> bool:
        return result.actions.filter(pl.col("symbol") == symbol).height == 1

    all_offsets = [0] + other_offsets
    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "duplicate_actions", "defect", check,
        f"3 sources logged the same {ratio_from}:{ratio_to} split within "
        f"{max(all_offsets) - min(all_offsets)}d of each other -- must collapse to 1 row",
    )


def _plant_same_date_contradiction(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTSDC{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    d = bars["date"][_mid_index(cfg, rng)]
    ratio_to = rng.choice([2, 3, 4])
    bars = _inject_split(bars, d, 1, ratio_to)

    actions = [
        {"symbol": symbol, "date": d, "ratio_from": 1, "ratio_to": ratio_to,
         "source": "price_feed", "type": "split"},
        {"symbol": symbol, "date": d, "ratio_from": ratio_to, "ratio_to": 1,
         "source": "text_feed", "type": "reverse_split"},  # fabricated inverse claim
    ]

    def check(result: "GauntletResult") -> bool:
        rows = result.actions.filter((pl.col("symbol") == symbol) & (pl.col("date") == d))
        if rows.height != 1:
            return False
        r = rows.row(0, named=True)
        return r["ratio_from"] == 1 and r["ratio_to"] == ratio_to

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "same_date_contradiction", "defect", check,
        f"same date carries BOTH a real 1:{ratio_to} split and a fabricated inverse claim "
        f"({ratio_to}:1) -- must collapse to 1 row AND pick the tape-confirmed one",
    )


def _plant_near_date_fake(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTNDF{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    d_true = bars["date"][_mid_index(cfg, rng)]
    true_ratio = rng.choice([10, 15, 20])
    bars = _inject_split(bars, d_true, true_ratio, 1)  # true: N:1 reverse split
    d_fake = d_true + dt.timedelta(days=rng.randint(2, 8))  # within near_date_window_days=10
    fake_ratio = true_ratio * 3

    actions = [
        {"symbol": symbol, "date": d_true, "ratio_from": true_ratio, "ratio_to": 1,
         "source": "text_feed_early", "type": "reverse_split"},
        {"symbol": symbol, "date": d_fake, "ratio_from": fake_ratio, "ratio_to": 1,
         "source": "misparsed_feed", "type": "reverse_split"},  # fabricated ratio, unranked
    ]

    def check(result: "GauntletResult") -> bool:
        rows = result.actions.filter(pl.col("symbol") == symbol)
        if rows.height != 1:
            return False
        r = rows.row(0, named=True)
        return r["ratio_from"] == true_ratio and r["ratio_to"] == 1

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "near_date_fake", "defect", check,
        f"two different-ratio reverse-split claims {(d_fake - d_true).days}d apart -- one "
        f"real ({true_ratio}:1), one fabricated ({fake_ratio}:1) -- must collapse to the "
        f"tape-confirmed one",
    )


def _plant_post_coverage_action(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTPCA{i:02d}"
    n_short = max(30, cfg.n_trading_days // 3)  # "delisted" early -- short series
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(n_short, cfg.start_date, base, cfg.daily_vol, rng)
    last_bar_date = bars["date"][-1]
    action_date = last_bar_date + dt.timedelta(days=rng.randint(200, 700))

    actions = [{"symbol": symbol, "date": action_date, "ratio_from": 1, "ratio_to": 10,
               "source": "stale_feed", "type": "split"}]

    def check(result: "GauntletResult") -> bool:
        return result.actions.filter(pl.col("symbol") == symbol).height == 0

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "post_coverage_action", "defect", check,
        f"action dated {(action_date - last_bar_date).days}d after the symbol's final bar -- "
        f"no boundary exists in the data, must be dropped",
    )


def _plant_spinoff_mislabel(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTSPM{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    d = bars["date"][_mid_index(cfg, rng)]
    # A spin-off drops the parent's price ~50% -- INDISTINGUISHABLE from a real 1:2 split by
    # tape alone. That's the point of this class: only a citation can exclude it.
    bars = _inject_split(bars, d, 1, 2)

    actions = [{"symbol": symbol, "date": d, "ratio_from": 1, "ratio_to": 2,
               "source": "vendor_feed", "type": "split"}]
    mislabels = {(symbol, d.isoformat(), 1, 2):
                 "synthetic spin-off value-separation event, mislabeled by the vendor as a "
                 "1:2 split (gauntlet class spinoff_mislabel)"}

    def check(result: "GauntletResult") -> bool:
        return result.actions.filter(pl.col("symbol") == symbol).height == 0

    return PlantResult(
        symbol, bars, actions, {}, {}, mislabels, "spinoff_mislabel", "defect", check,
        "tape genuinely confirms a 1:2 move (spin-off value separation, not a split) -- only "
        "a cited exclusion (ChainConfig.recorded_mislabels) can drop this; the tape alone "
        "cannot tell the difference",
    )


def _plant_misdated_action(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTMDA{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    d_true = bars["date"][_mid_index(cfg, rng)]
    ratio_to = rng.choice([4, 5, 8, 10])
    bars = _inject_split(bars, d_true, 1, ratio_to)
    wrong_offset = rng.choice([-3, -2, -1, 1, 2, 3])
    d_wrong = d_true + dt.timedelta(days=wrong_offset)

    actions = [{"symbol": symbol, "date": d_wrong, "ratio_from": 1, "ratio_to": ratio_to,
               "source": "announcement_feed", "type": "split"}]

    def check(result: "GauntletResult") -> bool:
        rows = result.actions.filter(pl.col("symbol") == symbol)
        if rows.height != 1:
            return False
        return rows.row(0, named=True)["date"] == d_true

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "misdated_action", "defect", check,
        f"action logged {wrong_offset:+d}d off the true tape boundary -- must snap to the "
        f"unique boundary date",
    )


def _plant_phantom_action(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTPHA{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)  # no split
    d = bars["date"][_mid_index(cfg, rng)]
    # NOTE: ratio_to=2 is deliberately excluded. |ln(1/2)| == ChainConfig's default
    # large_ratio_threshold_ln AND phantom_disagreement_tolerance_ln (both ln(2), matching
    # the extracted engine's own thresholds) -- a claim at EXACTLY that ratio sits on the
    # boundary the disagreement check uses `>` (strict) to decide, so a perfectly flat tape
    # can go either way on essentially a coin flip. That's a genuine, documented property of
    # the tolerance design (see docs/STANDARD.md "small ratios ... noise can confound"), not
    # something this gauntlet should paper over -- but it also isn't a fair test of "does the
    # engine refute an obvious phantom," so the planted ratio stays clear of that exact edge.
    ratio_to = rng.choice([3, 4, 5])

    actions = [{"symbol": symbol, "date": d, "ratio_from": 1, "ratio_to": ratio_to,
               "source": "misparsed_feed", "type": "split"}]

    def check(result: "GauntletResult") -> bool:
        return result.actions.filter(pl.col("symbol") == symbol).height == 0

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "phantom_action", "defect", check,
        f"a claimed 1:{ratio_to} split with NO tape support (flat close through the date) -- "
        f"must be refuted",
    )


def _plant_extreme_ratio_real_split(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTERS{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    d = bars["date"][_mid_index(cfg, rng)]
    ratio_from = rng.choice([50, 75, 100])  # nano-cap-reality extreme, correctly dated
    bars = _inject_split(bars, d, ratio_from, 1)

    actions = [{"symbol": symbol, "date": d, "ratio_from": ratio_from, "ratio_to": 1,
               "source": "price_feed", "type": "reverse_split"}]

    def check(result: "GauntletResult") -> bool:
        rows = result.actions.filter(pl.col("symbol") == symbol)
        if rows.height != 1:
            return False
        r = rows.row(0, named=True)
        return r["ratio_from"] == ratio_from and r["ratio_to"] == 1 and r["date"] == d

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "extreme_ratio_real_split", "legitimate", check,
        f"a genuine {ratio_from}:1 reverse split, correctly dated and tape-confirmed -- an "
        f"extreme ratio alone must never trigger refutation",
    )


def _plant_mixed_vintage_ohl(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTMVO{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    # NOT split -- this vendor's series arrived ALREADY pre-adjusted, so the raw tape shows
    # no discontinuity at all: identical shape to a true phantom. Only the vintage flag tells
    # them apart.
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    d = bars["date"][_mid_index(cfg, rng)]
    # ratio_to=2 excluded -- see the identical note in _plant_phantom_action: it sits exactly
    # on the disagreement-tolerance boundary, which would make the WITHOUT-vintage-awareness
    # side of demonstrate_vintage_awareness() a coin flip rather than a clean "yes, this is
    # what the vintage guard prevents" comparison.
    ratio_to = rng.choice([3, 4, 5])

    actions = [{"symbol": symbol, "date": d, "ratio_from": 1, "ratio_to": ratio_to,
               "source": "vendor_feed", "type": "split"}]
    vintage = {(symbol, d): "ADJUSTED"}

    def check(result: "GauntletResult") -> bool:
        # Meaningful when vintage_fn is engaged (run_gauntlet always wires it in) --
        # see demonstrate_vintage_awareness() for the explicit with/without comparison.
        return result.actions.filter(pl.col("symbol") == symbol).height == 1

    return PlantResult(
        symbol, bars, actions, {}, vintage, {}, "mixed_vintage_ohl", "legitimate", check,
        f"real 1:{ratio_to} split on a PRE-ADJUSTED vintage (tape shows no gap, identical "
        f"shape to a true phantom) -- must survive ONLY because vintage_fn marks the "
        f"boundary ADJUSTED",
    )


def _plant_envelope_wick(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTEWK{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    idx = _mid_index(cfg, rng)
    d = bars["date"][idx]
    c = float(bars["close"][idx])
    real_high = c * rng.uniform(1.9, 2.3)  # genuinely traded, exceeds the [0.55,1.8] envelope
    real_low = c * 0.97
    # fully control this day's candle (open == close) so geometry trivially holds regardless
    # of the randomly-generated open/high/low the base generator produced for this index.
    bars = bars.with_columns([
        pl.when(pl.col("date") == d).then(pl.lit(c)).otherwise(pl.col("open")).alias("open"),
        pl.when(pl.col("date") == d).then(pl.lit(real_high)).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("date") == d).then(pl.lit(real_low)).otherwise(pl.col("low")).alias("low"),
    ])
    truth = {(symbol, d): {"high": real_high, "low": real_low, "n_bars": 390}}

    def check(result: "GauntletResult") -> bool:
        row = result.bars.filter((pl.col("symbol") == symbol) & (pl.col("date") == d))
        if row.height != 1:
            return False
        r = row.row(0, named=True)
        # correct: the real, truth-confirmed wick is KEPT unchanged and flagged -- never
        # collapsed down to close (the envelope-only bug) nor overwritten with the band edge.
        return abs(r["high"] - real_high) < 1e-9 and r.get("confirmed_extreme", 0) == 1

    return PlantResult(
        symbol, bars, [], truth, {}, {}, "envelope_wick", "legitimate", check,
        f"a real {real_high / c:.2f}x wick, confirmed by dense sub-daily coverage -- must be "
        f"kept unchanged and flagged as a confirmed extreme, never deleted down to close",
    )


def _plant_in_band_phantom_dip(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTIPD{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    idx = _mid_index(cfg, rng)
    d = bars["date"][idx]
    c = float(bars["close"][idx])
    bad_low = c * 0.75    # in-band (> 0.55) -- never trips the OOB band on its own
    good_high = c * 1.03  # high/low ratio 1.373 -- never trips the 2.0 range-insanity ratio
    bars = bars.with_columns([
        pl.when(pl.col("date") == d).then(pl.lit(c)).otherwise(pl.col("open")).alias("open"),
        pl.when(pl.col("date") == d).then(pl.lit(good_high)).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("date") == d).then(pl.lit(bad_low)).otherwise(pl.col("low")).alias("low"),
    ])
    true_low = c * 0.985
    truth = {(symbol, d): {"high": c * 1.02, "low": true_low, "n_bars": 390}}  # dense coverage

    def check(result: "GauntletResult") -> bool:
        row = result.bars.filter((pl.col("symbol") == symbol) & (pl.col("date") == d))
        if row.height != 1:
            return False
        r = row.row(0, named=True)
        return r["repair_source"] == "truth" and abs(r["low"] - true_low) / true_low < 0.03

    return PlantResult(
        symbol, bars, [], truth, {}, {}, "in_band_phantom_dip", "defect", check,
        f"a bad low print ({bad_low / c:.2f}x close) too small to trip the OOB band or the "
        f"range-insanity ratio -- only caught because DENSE sub-daily coverage contradicts "
        f"it directly",
    )


def _plant_bad_close_vspike(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTBCV{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    idx = _mid_index(cfg, rng)
    d = bars["date"][idx]
    c_prev = float(bars["close"][idx - 1])
    bad_close = c_prev * rng.choice([6.0, 8.0, 0.15, 0.12])
    # Keep the WHOLE candle self-consistent around the bad close (a geometry check can't
    # catch a print that's simply wrong but internally coherent); next day is left untouched
    # so it reverts fully, back to the pre-spike trajectory.
    bars = bars.with_columns([
        pl.when(pl.col("date") == d).then(pl.lit(bad_close)).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("date") == d).then(pl.lit(bad_close)).otherwise(pl.col("open")).alias("open"),
        pl.when(pl.col("date") == d).then(pl.lit(bad_close * 1.01)).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("date") == d).then(pl.lit(bad_close * 0.99)).otherwise(pl.col("low")).alias("low"),
    ])

    def check(result: "GauntletResult") -> bool:
        row = result.bars.filter((pl.col("symbol") == symbol) & (pl.col("date") == d))
        if row.height != 1:
            return False
        # Ground truth: this is a KNOWN, DOCUMENTED gap (docs/STANDARD.md "Known gaps") --
        # the guard never validates `close` against anything. This asserts the (unfortunate
        # but currently correct) status quo; it starts FAILING the day someone ships a
        # close-validator, which is exactly the point of tracking it explicitly.
        return row.row(0, named=True)["bad_print_flag"] == 0

    return PlantResult(
        symbol, bars, [], {}, {}, {}, "bad_close_vspike", "known_gap", check,
        f"an isolated {bad_close / c_prev:.1f}x close V-spike with full next-day reversal, "
        f"geometry-consistent -- KNOWN GAP: apply_bad_print_guard never validates close "
        f"itself",
    )



# ---------------------------------------------------------------------------------------
# Hard real-world cases added in v0.1.1 (specialist review). These are the patterns that broke
# earlier versions on real micro-cap data.
# ---------------------------------------------------------------------------------------


def _plant_serial_reverse_split(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTSRS{i:02d}"
    base = rng.uniform(0.5, 3.0)  # micro-cap: repeat reverse splits to stay listed
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    n = cfg.n_trading_days
    i1 = n // 3
    i2 = i1 + rng.randint(max(5, n // 12), max(6, min(90, n // 3)))  # inside the collapse window
    d1, d2 = bars["date"][i1], bars["date"][i2]
    r = rng.choice([5, 10, 20])
    bars = _inject_split(_inject_split(bars, d1, r, 1), d2, r, 1)
    actions = [{"symbol": symbol, "date": d1, "ratio_from": r, "ratio_to": 1,
                "source": "price_feed", "type": "reverse_split"},
               {"symbol": symbol, "date": d2, "ratio_from": r, "ratio_to": 1,
                "source": "price_feed", "type": "reverse_split"}]

    def check(result: "GauntletResult") -> bool:
        rows = result.actions.filter(pl.col("symbol") == symbol).sort("date")
        return rows.height == 2 and rows["date"].to_list() == [d1, d2]

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "serial_reverse_split", "legitimate", check,
        f"two real {r}:1 reverse splits {i2 - i1} trading days apart, both on the tape -- must "
        f"stay TWO events, never collapsed into one",
    )


def _plant_halted_reverse_split(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTHLT{i:02d}"
    base = rng.uniform(0.5, 3.0)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    n = cfg.n_trading_days
    halt = rng.randint(max(8, min(30, n // 8)), max(9, min(60, n // 5)))  # days with no bars at all
    k = rng.randint(n // 4, max(n // 4, n - halt - max(5, n // 6)))
    resume = bars["date"][k + halt]
    r = rng.choice([5, 10, 20])
    # During the halt the company lost ~1/r of its value, exactly offsetting the reverse split:
    # the close before the halt and the close after it are about equal.
    bars = bars.filter((pl.col("date") < bars["date"][k]) | (pl.col("date") >= resume))
    bars = _inject_split(bars, resume, r, 1)
    bars = _inject_split(bars, resume, 1, r)  # the offsetting collapse (price only matters here)
    actions = [{"symbol": symbol, "date": resume, "ratio_from": r, "ratio_to": 1,
                "source": "price_feed", "type": "reverse_split"}]

    def check(result: "GauntletResult") -> bool:
        return result.actions.filter(pl.col("symbol") == symbol).height == 1

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "halted_reverse_split", "legitimate", check,
        f"a real {r}:1 reverse split effective when trading resumed after a {halt}-day halt, its "
        f"jump cancelled by the collapse during the halt -- unmeasurable, so it must be KEPT",
    )


def _plant_real_small_ratio_split(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTSMR{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)
    d = bars["date"][_mid_index(cfg, rng)]
    ratio_from, ratio_to = rng.choice([(2, 3), (4, 5)])  # 3-for-2, 5-for-4
    bars = _inject_split(bars, d, ratio_from, ratio_to)
    actions = [{"symbol": symbol, "date": d, "ratio_from": ratio_from, "ratio_to": ratio_to,
                "source": "price_feed", "type": "split"}]

    def check(result: "GauntletResult") -> bool:
        rows = result.actions.filter(pl.col("symbol") == symbol)
        return rows.height == 1 and rows["date"][0] == d

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "real_small_ratio_split", "legitimate", check,
        f"a genuine {ratio_to}-for-{ratio_from} split, correctly dated -- must be kept",
    )


def _plant_small_ratio_phantom(i: int, cfg: GauntletConfig, rng: random.Random) -> PlantResult:
    symbol = f"TESTSRP{i:02d}"
    base = rng.uniform(*cfg.base_price_range)
    bars = _generate_clean_bars(cfg.n_trading_days, cfg.start_date, base, cfg.daily_vol, rng)  # no split
    d = bars["date"][_mid_index(cfg, rng)]
    actions = [{"symbol": symbol, "date": d, "ratio_from": 2, "ratio_to": 3,
                "source": "misparsed_feed", "type": "split"}]

    def check(result: "GauntletResult") -> bool:
        # known gap: True means the phantom STILL SURVIVES (the documented limitation holds)
        return result.actions.filter(pl.col("symbol") == symbol).height == 1

    return PlantResult(
        symbol, bars, actions, {}, {}, {}, "small_ratio_phantom", "known_gap", check,
        "a phantom 3-for-2 split on a flat tape -- ratios under 2x are not tape-verified "
        "(docs/STANDARD.md Known gaps #3)",
    )

_PLANTERS: list[Callable[[int, GauntletConfig, random.Random], PlantResult]] = [
    _plant_duplicate_actions,
    _plant_same_date_contradiction,
    _plant_near_date_fake,
    _plant_post_coverage_action,
    _plant_spinoff_mislabel,
    _plant_misdated_action,
    _plant_phantom_action,
    _plant_extreme_ratio_real_split,
    _plant_mixed_vintage_ohl,
    _plant_envelope_wick,
    _plant_in_band_phantom_dip,
    _plant_bad_close_vspike,
    _plant_serial_reverse_split,
    _plant_halted_reverse_split,
    _plant_real_small_ratio_split,
    _plant_small_ratio_phantom,
]


# ---------------------------------------------------------------------------------------
# Universe assembly, run, and scoring
# ---------------------------------------------------------------------------------------


@dataclass
class GauntletUniverse:
    bars: InMemoryBarProvider
    actions_df: pl.DataFrame
    truth: InMemoryTruthProvider
    vintage_map: dict
    mislabels: dict
    plants: list[PlantResult]
    clean_symbols: list[str]
    config: GauntletConfig


def build_gauntlet_universe(config: GauntletConfig = GauntletConfig()) -> GauntletUniverse:
    """Builds the full synthetic universe: 12 classes x `n_instances_per_class` planted
    instances, plus `n_clean_symbols` control symbols with no defects at all."""
    rng = random.Random(config.seed)
    bars_map: dict[str, pl.DataFrame] = {}
    all_actions: list[dict] = []
    truth: dict = {}
    vintage: dict = {}
    mislabels: dict = {}
    plants: list[PlantResult] = []

    for planter in _PLANTERS:
        for i in range(1, config.n_instances_per_class + 1):
            plant = planter(i, config, rng)
            bars_map[plant.symbol] = plant.bars
            all_actions.extend(plant.actions)
            truth.update(plant.truth)
            vintage.update(plant.vintage)
            mislabels.update(plant.mislabels)
            plants.append(plant)

    clean_symbols = []
    for i in range(1, config.n_clean_symbols + 1):
        symbol = f"TESTCLN{i:02d}"
        base = rng.uniform(*config.base_price_range)
        bars_map[symbol] = _generate_clean_bars(
            config.n_trading_days, config.start_date, base, config.daily_vol, rng)
        clean_symbols.append(symbol)

    actions_df = (pl.DataFrame(all_actions, schema=ACTION_SCHEMA) if all_actions
                 else pl.DataFrame(schema=ACTION_SCHEMA))

    return GauntletUniverse(
        bars=InMemoryBarProvider(bars_map), actions_df=actions_df,
        truth=InMemoryTruthProvider(truth), vintage_map=vintage, mislabels=mislabels,
        plants=plants, clean_symbols=clean_symbols, config=config,
    )


@dataclass
class GauntletResult:
    actions: pl.DataFrame  # post-collapse_all
    bars: pl.DataFrame     # post-apply_bad_print_guard, all symbols concatenated
    chain_stats: dict


def run_gauntlet(
    universe: GauntletUniverse,
    chain_config: ChainConfig | None = None,
    guard_config: GuardConfig | None = None,
) -> GauntletResult:
    """Runs the full engine (collapse_all + apply_bad_print_guard) over a gauntlet universe,
    with the universe's own mislabels/vintage wired in. This is the "fully configured
    engine" pass that :func:`score_run` grades -- see :func:`demonstrate_vintage_awareness`
    for a side-by-side with the vintage hook OFF."""
    if chain_config is None:
        chain_config = ChainConfig(
            source_priority=[TAPE_CONFIRMED_SOURCE, _PRICE_DERIVED_SOURCE],
            recorded_mislabels=dict(universe.mislabels),
        )
    elif not chain_config.recorded_mislabels:
        chain_config.recorded_mislabels = dict(universe.mislabels)
    guard_config = guard_config or GuardConfig()

    series_fn = make_close_series_fn(universe.bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    last_date_fn = make_last_bar_date_fn(series_fn=series_fn)
    vintage_fn = lambda symbol, date: universe.vintage_map.get((symbol, date))  # noqa: E731

    collapsed, stats = collapse_all(
        universe.actions_df, config=chain_config, gap_fn=gap_fn, last_date_fn=last_date_fn,
        series_fn=series_fn, vintage_fn=vintage_fn,
    )

    all_bars = pl.concat([
        universe.bars.get_bars(s).with_columns(pl.lit(s).alias("symbol"))
        for s in universe.bars.symbols()
    ])
    guarded = apply_bad_print_guard(all_bars, truth=universe.truth, config=guard_config)

    return GauntletResult(actions=collapsed, bars=guarded, chain_stats=stats)


@dataclass
class ClassScore:
    defect_class: str
    category: str
    n_instances: int
    n_correct: int
    rate: float
    description: str


@dataclass
class Scorecard:
    class_scores: list[ClassScore]
    overall_detection_rate: float           # mean rate, category == "defect"
    overall_legitimate_preserve_rate: float  # mean rate, category == "legitimate"
    known_gap_scores: list[ClassScore]       # category == "known_gap" -- reported, not counted
    clean_false_positive_rate: float         # fraction of non-abnormal-OHLC bars flagged
    n_clean_bars_checked: int


def score_run(universe: GauntletUniverse, result: GauntletResult) -> Scorecard:
    """Grades a :func:`run_gauntlet` result against the universe's planted ground truth."""
    by_class: dict[str, list[PlantResult]] = {}
    for p in universe.plants:
        by_class.setdefault(p.defect_class, []).append(p)

    class_scores = []
    for cls, plants in by_class.items():
        n = len(plants)
        n_correct = sum(1 for p in plants if p.check(result))
        class_scores.append(ClassScore(
            defect_class=cls, category=plants[0].category, n_instances=n, n_correct=n_correct,
            rate=(n_correct / n) if n else 0.0, description=plants[0].description,
        ))
    class_scores.sort(key=lambda s: s.defect_class)

    defect_scores = [s for s in class_scores if s.category == "defect"]
    legit_scores = [s for s in class_scores if s.category == "legitimate"]
    gap_scores = [s for s in class_scores if s.category == "known_gap"]

    overall_detection = (sum(s.rate for s in defect_scores) / len(defect_scores)) if defect_scores else 0.0
    overall_legit = (sum(s.rate for s in legit_scores) / len(legit_scores)) if legit_scores else 0.0

    # Clean false-positive rate: every symbol EXCEPT the 3 guard-defect classes (which
    # legitimately carry abnormal OHLC on purpose) -- clean control symbols plus all 9
    # chain-only / action-only classes, whose bars should never be touched by the guard.
    fp_symbols = set(universe.clean_symbols) | {
        p.symbol for p in universe.plants if p.defect_class not in _GUARD_DEFECT_CLASSES
    }
    fp_bars = result.bars.filter(pl.col("symbol").is_in(list(fp_symbols)))
    n_checked = fp_bars.height
    n_flagged = fp_bars.filter(pl.col("bad_print_flag") == 1).height
    clean_fp_rate = (n_flagged / n_checked) if n_checked else 0.0

    return Scorecard(
        class_scores=class_scores, overall_detection_rate=overall_detection,
        overall_legitimate_preserve_rate=overall_legit, known_gap_scores=gap_scores,
        clean_false_positive_rate=clean_fp_rate, n_clean_bars_checked=n_checked,
    )


def demonstrate_vintage_awareness(universe: GauntletUniverse) -> dict:
    """Runs :func:`~tapetruth.chain.refute_phantom_actions` on the `mixed_vintage_ohl`
    instances TWICE -- once with the `vintage_fn` hook engaged (correct), once without
    (naive) -- to make concrete what "vintage-aware" actually buys you. The main
    :func:`run_gauntlet` / :func:`score_run` pass always runs WITH the hook engaged (the
    fully-configured, recommended posture); this helper exists purely to show the
    counterfactual.

    Returns ``{"n_instances", "survived_without_vintage_awareness", "survived_with_vintage_awareness"}``.
    """
    mvo_symbols = [p.symbol for p in universe.plants if p.defect_class == "mixed_vintage_ohl"]
    mvo_actions = universe.actions_df.filter(pl.col("symbol").is_in(mvo_symbols))
    series_fn = make_close_series_fn(universe.bars)
    gap_fn = make_gap_fn(series_fn=series_fn)

    without, _ = refute_phantom_actions(mvo_actions, gap_fn=gap_fn, vintage_fn=None)
    vintage_fn = lambda symbol, date: universe.vintage_map.get((symbol, date))  # noqa: E731
    with_vintage, _ = refute_phantom_actions(mvo_actions, gap_fn=gap_fn, vintage_fn=vintage_fn)

    return {
        "n_instances": len(mvo_symbols),
        "survived_without_vintage_awareness": without.height,
        "survived_with_vintage_awareness": with_vintage.height,
    }
