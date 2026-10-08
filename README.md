# split-adjustment-tool

**Verify, repair and apply stock-split adjustments against your own price tape.**

Split records are claims, and claims are often wrong: a spin-off logged as a split, the same
split logged twice by two sources, a split dated three days off, a reverse split that never
happened. Any backtest built on them is quietly wrong. This small, vendor-neutral Python library
checks every claim against the price series you actually have, keeps the real ones, fixes the
dates it can, rejects the fakes, and then produces split-adjusted prices and volume.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

---

## Why this exists

In September 2025 a user of a well-known retail brokerage's market-data API posted a list of
more than 20 wrong split records across 19 tickers on the vendor's own forum, most of them
spin-offs whose share distribution had been recorded as a split ratio:

> "I found that there is a systemic bug mixing up the split ratio whenever it was converted
> from a spin-off."
> — [forum.alpaca.markets/t/17839](https://forum.alpaca.markets/t/17839)

That is the normal state of corporate-action data, not an exotic edge case. This library's
answer: **your own as-traded price series is the evidence.** A real split leaves a permanent jump
in the raw prices on its ex-date. A claim is kept where that jump is there, moved to where the
jump actually is when it was misdated, and rejected when the prices show nothing happened.

## What it does

| Module | What it does |
|---|---|
| `split_adjustment_tool.chain` | **Cleans split claims.** Collapses duplicates from several sources (but never merges two real splits that each show up in the prices), resolves same-day and near-day contradictions by the measured price jump, drops claims after your data ends, moves misdated claims to the real jump, and rejects lone phantoms on a flat tape. Claims across a trading halt are kept and flagged as unmeasurable, never rejected. One source's same-day reverse + forward legs (an odd-lot cash-out) are combined into one event; ratios are exact numbers (1:1.0526 is not rounded). |
| `split_adjustment_tool.factors` | **Applies them.** `apply_split_adjustment(bars, actions)` adds a cumulative price factor and volume factor (CRSP-style: anchored at the latest bar, effective from each ex-date) and `adj_open/high/low/close` + `adj_volume`; dollar volume is unchanged by construction. `build_factor_table` gives the sparse one-row-per-event form. |
| `split_adjustment_tool.reconcile` | **Compares two adjustment histories event by event** (yours vs a vendor's): each split is matched, `DATE_DIFFERS`, `RATIO_DIFFERS`, or only on one side. Sparse vendor tables are aligned as-of, so they compare against a daily series. |
| `split_adjustment_tool.breaks` | **Catches the price feed itself going wrong.** Finds lasting jumps no claim explains (judged against each stock's own normal moves) and asks an independent *witness* series (a second vendor, or daily closes built from your own intraday tape) what happened: `PROVIDER_BREAK` (the feed is wrong from that day, with the day it was fixed if it was), `REAL_MOVE` (real: a split your claims missed, or a genuine crash), or `UNEXPLAINED` when there is no witness. It also rejects a bogus claim that only the broken feed "confirms". |
| `split_adjustment_tool.identity` | **Keeps companies apart when tickers change hands.** Describe which ticker each security used and when; claims are relabelled to the security and bars are served per security, so a renamed company is one continuous history and a reused ticker never attaches one company's splits to another's prices. |
| `split_adjustment_tool.feedcheck` | **Catches two quieter feed failures** against a witness series: prices frozen for longer than the stock's own normal (`STALE_FEED`, or `ILLIQUID` when the witness was flat too) and bars stamped one session late or early (`DATE_SHIFT` spans with direction). |
| `split_adjustment_tool.guard` | **Repairs bad price prints** in daily bars (a low far below the day's body, a high below its own open). With intraday data it repairs from the intraday extreme; a wild move the intraday data confirms is kept and flagged as `confirmed_extreme`, never overwritten. |

**Timeframes and time zones.** Splits are checked on daily bars (a split is a once-a-day event); the
adjustment then applies to bars of any size — daily, 1-hour, 1-minute, ticks. A split takes effect at the
start of its ex-date in the exchange's local time, so each row gets the factor for its exchange-local
trading date, which the library works out for you from whatever timestamps you have:

```python
adjusted = apply_split_adjustment(minute_bars, clean, timestamp_col="ts")          # UTC, any zone, or naive
adjusted = apply_split_adjustment(minute_bars, clean, timestamp_col="ts",
                                  tz_by_symbol={"7203.T": "Asia/Tokyo"})            # per-exchange zones
```

Time-zone-aware timestamps are converted to the exchange zone (New York by default); naive timestamps
are read as UTC unless you say otherwise (`naive_timestamps_tz`). An 8 PM New York after-hours bar —
already the next day in UTC — stays on the right side of a split. Verified on real hourly and 1-minute
bars across a 10-for-1 split, pre-market and after-hours included, with timestamps supplied as UTC, naive
UTC and Tokyo time: identical to an independently built adjusted series in all six runs. The bad-print
guard repairs daily bars only (intraday data is its witness).

Everything is IO-agnostic: you hand it a small provider object for your bars and claims (CSV,
Parquet and in-memory implementations included). It never opens a database or calls an API.

## Quickstart

```bash
git clone https://github.com/AlexBocio/split-adjustment-tool && cd split-adjustment-tool
pip install -e .
python -m split_adjustment_tool.demo      # the benchmark, end to end, in a few seconds
```

```python
import polars as pl
from split_adjustment_tool import (collapse_all, make_close_series_fn, make_gap_fn,
                                   make_last_bar_date_fn, apply_split_adjustment)
from split_adjustment_tool.providers import CSVBarProvider, CSVActionSource

bars = CSVBarProvider("my_bars_dir/")                       # one CSV per symbol: date,open,high,low,close,volume (as traded)
claims = CSVActionSource("my_splits.csv").get_actions()     # symbol,date,ratio_from,ratio_to[,source]
# ratio convention: a 1-for-10 forward split is 1,10; a 1-for-5 reverse split is 5,1

series_fn = make_close_series_fn(bars)
clean, stats = collapse_all(claims, gap_fn=make_gap_fn(series_fn=series_fn),
                            last_date_fn=make_last_bar_date_fn(series_fn=series_fn),
                            series_fn=series_fn)
print(stats["phantom"]["refuted"], stats["phantom"]["unmeasurable_kept"], stats["snap"]["snapped"])

raw = pl.read_csv("my_bars_dir/ABC.csv", try_parse_dates=True).with_columns(pl.lit("ABC").alias("symbol"))
adjusted = apply_split_adjustment(raw, clean)               # adds price_factor, volume_factor, adj_* columns
```

## Real-data validation

Checked on real US equity data (8 stocks: five large forward splits in 2024 and three micro-cap
reverse splits in 2026, about 3 years of daily bars each):

- **Claims:** 8 real splits plus 6 planted bad claims (a duplicate, a 4-day misdate, two phantoms,
  a wrong ratio, a same-day contradiction, a claim after the data ends): **14 of 14 handled
  correctly** — the misdated split was moved to its real date from the price jump alone.
- **Adjustment:** adjusted closes from this library matched an independently built adjusted series
  on **5,530 of 5,530 trading days** (worst difference 7e-16, i.e. identical), including micro-caps
  with back-to-back reverse splits.
- **Comparison:** against a hand-made reference list, the reconciler flagged a planted misdate
  (`DATE_DIFFERS`), a planted wrong ratio (`RATIO_DIFFERS`), a planted phantom, and two real earlier
  reverse splits the reference list had missed.
- **Bad prints:** on 6,369 real daily bars with hourly data, the guard changed exactly one bar —
  a real decimal-slip low (69.005 on a ~$690 day), repaired to 681.94 — and caught two documented
  historical bad prints; three real crash days were kept and flagged, not flattened.

## When the provider itself is wrong

Everything above treats your price tape as the evidence. If a vendor delivers good prices until day X and
then, from the next day, prices on the wrong basis (10x off, a split applied early or never applied),
nothing in the claims can catch it — and a bogus claim on that day would even look confirmed. Give the
library a second, independent price series and it will tell you:

```python
from split_adjustment_tool import scan_unexplained_jumps, classify_jumps, check_claims_against_witness
jumps = classify_jumps(scan_unexplained_jumps(primary_series_fn, symbols, clean), witness_series_fn)
clean, wstats = check_claims_against_witness(clean, witness_series_fn)   # drops claims only the bad feed shows
```

Checked against a real incident: a free data feed delivered 220 daily bars across 94 stocks on the wrong
split basis (August–September 2026), and the bad copies were kept. With daily closes built from the
same stocks' 1-minute tape as the witness, **91 of the 104 bad windows were flagged `PROVIDER_BREAK` on
their exact first day, with zero provider-break flags on good days**; on clean data (9 stocks, 6,369
bars) it raised no provider breaks at all and listed 9 `REAL_MOVE`s for review (genuine crashes, and one
reverse split missing from the claim list). Without a witness a provider break and a real one-day crash
look identical — both are reported `UNEXPLAINED`, never acted on.

## The benchmark

`python -m split_adjustment_tool.demo` builds a synthetic universe (no real tickers, no licensed
data) with 16 classes of planted problems and real-looking events, runs everything, and prints a
scorecard. The verdict requires defects caught **and** every class of real events kept (at least
95% per class), so a tool that deletes real splits cannot pass. Two classes (`spinoff_mislabel`,
`mixed_vintage_ohl`) are handed their answer key and are labelled *configured*. Treat it as a
regression suite you can also point at your own pipeline (`split_adjustment_tool.gauntlet`), not as
proof on its own — the real-data numbers above are the evidence.

## Limits (read before relying on it)

- **Splits and reverse splits only.** Cash dividends, spin-offs and rights need a distribution
  factor this library does not compute. A spin-off looks like a split in the prices; the only
  exclusion is a cited record you supply (`ChainConfig.recorded_mislabels`).
- **Ratios under 2x (3-for-2, 5-for-4, stock dividends) are de-duplicated but not price-verified** —
  an ordinary day's move can mimic them, so a phantom 3-for-2 passes through.
- **Ticker identity is opt-in.** Without a symbol history (`split_adjustment_tool.identity`) everything is
  keyed on the ticker string, and a reused ticker can attach one company's splits to another's prices.
- **Without intraday data, the bad-print guard cannot tell a real one-day crash from a bad print**
  on a volatile small stock. Give it intraday data for small caps.
- **An isolated bad close that snaps back the next day** is caught only with a witness series (the
  `breaks` module reports it as a one-day window); without one it is not.
- **The feed checks need a witness** (a second vendor, or daily closes built from your own intraday data)
  to say what a frozen run or a shifted span was; without one they report `UNVERIFIED`.
- Full method, rules and known gaps: [`docs/STANDARD.md`](docs/STANDARD.md).

## Status

**v0.4.0 (alpha).** For educational and research use — see [`DISCLAIMER.md`](DISCLAIMER.md) (AS-IS,
no warranty, not investment advice). [Apache 2.0](LICENSE). It ships code and a synthetic benchmark;
it does not ship, sell or redistribute any market data — you bring your own bars and claims.
Changes: [`CHANGELOG.md`](CHANGELOG.md).

## Roadmap

- Point-in-time factors (`as_of`): adjust as it was known on a date, for honest backtests.
- Event types beyond splits (stock dividends, spin-offs, rights) with distribution factors.
- A command-line tool; a PyPI release; a public real-data benchmark of hard cases.

## Contributing

Issues and pull requests are welcome. Keep test fixtures synthetic (no real tickers, no licensed
price data) so the repository stays simple to contribute to and fork.
