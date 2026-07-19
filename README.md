# tapetruth

**Records are hypotheses; the tape is truth.**

A small, vendor-neutral, pluggable Python engine that verifies and repairs corporate-action
claims (splits, reverse splits) and OHLCV bar-data prints against your own price tape —
instead of trusting any single feed at face value.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

---

## Why this exists

In 2024–2025, users of a well-known retail brokerage's market-data API discovered that
several of its historical split adjustments were simply wrong. The vendor's own forum
confirmed it plainly:

> "There is a systemic bug mixing up the split ratio whenever it was converted from a
> spin-off."
> — [forum.alpaca.markets/t/17839](https://forum.alpaca.markets/t/17839)

At least 13 tickers were affected. The fix was reactive: patched one ticker at a time, after
users noticed and reported it. That's the failure mode this project exists to catch *before*
it reaches your backtest: a corporate-action record is a claim from some upstream system —
scraped, filed, or vendor-computed — and claims can simply be wrong. A spin-off's ~50% value
separation looks, in the price series alone, exactly like a 1-for-2 split. A duplicate row
from a second data source compounds a real split into a fictitious 1000x factor. An action
logged three days off from its true effective date breaks every downstream calculation that
anchors to a date. None of these are exotic edge cases — they are the *normal* failure modes
of corporate-action data, and they silently corrupt any analysis built on top.

tapetruth's answer: **your own as-traded price series is truth.** It carries every real
split as a literal, permanent price discontinuity, immune to any adjustment-factor bug by
construction. A claimed corporate action is applied only where that tape confirms it — the
right ratio, at the right date, with next-bar persistence. Everything else is either snapped
to where the tape says it actually happened, or refuted outright.

## What it does

tapetruth ships three independent mechanisms, all vendor-neutral and pluggable:

| Module | What it catches |
|---|---|
| `tapetruth.chain` | **Corporate-action de-duplication and conflict resolution.** Multi-source duplicate rows for the same event, same-date contradictions (a "split" and a "reverse split" logged for the same day), near-date fabrications, actions dated after a symbol's last bar, misdated actions (snapped to their true tape boundary), and lone phantom claims the tape flatly contradicts. |
| `tapetruth.guard` | **OHLC bad-print repair.** Out-of-band fields, geometry violations (a high below its own candle body), range-insane days an independent sub-daily source contradicts, and in-band bad prints only dense sub-daily coverage can catch — repaired using truth data when available, never silently deleted. |
| `tapetruth.reconcile` | **Honest-class reconciliation.** Compares two independent adjustment-factor series and classifies the result as EXACT / MINOR / EXACT_ADJUSTED_EQUIV / EXPLAINED / NOT_COMPARABLE / MISMATCH — never collapsing "no data to compare" and "genuinely wrong" into the same bucket. |

Twelve defect classes, each with a synthetic, ground-truth-labeled test case in the included
benchmark (see below): duplicate actions, same-date contradictions, near-date fakes,
post-coverage actions, spin-off mislabels, misdated actions, phantom actions, extreme-ratio
real splits (a false-positive-avoidance check), mixed-vintage OHL, envelope-deleted wicks,
in-band phantom dips, and — reported honestly, not hidden — an isolated bad-close V-spike
that the current engine does **not** catch (see `docs/STANDARD.md` "Known gaps").

Everything is IO-agnostic. tapetruth never opens a database connection or calls an API —
you hand it a small object (a `BarProvider`, an `ActionSource`, optionally a
`TruthProvider`) that reads from wherever your data actually lives. Reference
implementations for CSV, Parquet, and plain in-memory DataFrames ship in the box.

## Quickstart

```bash
pip install -e .          # from a clone; PyPI package planned (see Roadmap)
```

```python
from tapetruth import ChainConfig, collapse_all, make_close_series_fn, make_gap_fn, make_last_bar_date_fn
from tapetruth.providers import CSVBarProvider, CSVActionSource

bars = CSVBarProvider("my_bars_dir/")           # one CSV per symbol: date,open,high,low,close,volume
actions = CSVActionSource("my_actions.csv").get_actions()   # symbol,date,ratio_from,ratio_to[,source][,type]

series_fn = make_close_series_fn(bars)
collapsed, stats = collapse_all(
    actions,
    gap_fn=make_gap_fn(series_fn=series_fn),
    last_date_fn=make_last_bar_date_fn(series_fn=series_fn),
    series_fn=series_fn,
)
print(stats)  # per-stage counts: mislabel, post_coverage, duplicate, same_date, near_date, snap, phantom
```

Then see it work end-to-end, with zero setup:

```bash
python -m tapetruth.demo
```

This generates a synthetic universe of planted defects with known ground truth, runs the
full engine over it, and prints a scorecard — detection rate per defect class,
false-positive rate on clean data, and the engine's honestly-reported known gaps.

## The gauntlet — score your own pipeline

`tapetruth.gauntlet` is both the test fixture for this repository AND a standalone
benchmark you can point at your own corporate-action / bar-repair logic. It generates
realistic synthetic OHLCV data (geometric random walk, realistic daily volatility, gaps,
wicks, log-normal volume) with the twelve defect classes planted at known dates with known
ground truth — no real ticker, no licensed price data, nothing that can't ship in a public
repository, and (because the ground truth is known) a *better* demonstration than any real
dataset could be, since detection and repair are measurable rather than asserted.

```python
from tapetruth.gauntlet import GauntletConfig, build_gauntlet_universe, run_gauntlet, score_run

universe = build_gauntlet_universe(GauntletConfig())
result = run_gauntlet(universe)                 # swap this line for YOUR engine's output
scorecard = score_run(universe, result)
print(scorecard.overall_detection_rate, scorecard.clean_false_positive_rate)
```

## Method

The full doctrine — twelve rules, each tied to the failure mode that forged it, plus the
acceptance metrics and the honestly-tracked known gaps — is documented in
[`docs/STANDARD.md`](docs/STANDARD.md). The general shape (discipline one data stream
against an independent one, tolerance-banded, never assumed) has real academic lineage: the
CRSP price/share adjustment-factor methodology is the closest ancestor (and has its own
documented gaps in exactly this problem space — see the citations in `docs/STANDARD.md`
§9), and the TAQ tick-cleaning literature (Barndorff-Nielsen et al., 2009) is the closest
academic kin to the bad-print guard's design.

## Status

**v0.1.0.** Published for educational and research purposes — see
[`DISCLAIMER.md`](DISCLAIMER.md) for the full terms (AS-IS, no warranty, not investment
advice). Licensed [Apache 2.0](LICENSE): free to use, modify, and redistribute, including
commercially.

tapetruth ships the engine and the benchmark. It does not ship, sell, or redistribute any
market data — you bring your own bars and your own action claims.

## Roadmap (planned, not yet shipped)

- **Vendor adapter pack** — pre-built `BarProvider`/`ActionSource` implementations for
  common data providers.
- **Certification reports** — a standardized "validated against gauntlet vX" report format
  for sharing reconciliation results.
- **PyPI release** — `pip install tapetruth` without a clone.
- **Additional defect classes** as they're discovered and measured.

## Contributing

Issues and pull requests are welcome. Please keep any test fixtures synthetic (no real
ticker symbols, no licensed price data) — this keeps the repository legally simple to
contribute to and to fork.
