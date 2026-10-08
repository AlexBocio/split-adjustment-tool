# Changelog

## 0.4.0 — 2026-10-08

### Added
- `split_adjustment_tool.identity`: permanent security IDs. `symbol_history` (which ticker a security traded
  under, and when; overlapping ownership of one ticker is refused), `rekey_by_security` (relabel claims or
  any dated rows from ticker to security ID by date), `SecurityBarProvider` (bars by security: a renamed
  company is one continuous history, a reused ticker is cut at the hand-over). Every other module works on
  security IDs unchanged.
- `split_adjustment_tool.feedcheck`: `scan_stale_runs` (identical closes for longer than the symbol's own
  normal; against a witness: `STALE_FEED` / `ILLIQUID` / `UNVERIFIED`) and `scan_date_shifts` (bars stamped
  one session late or early, found as spans with direction), `FeedCheckConfig`.
- `compose_same_day_legs` (also step 1b of `collapse_all`): one source's same-day reverse + forward legs
  (an odd-lot cash-out) become one composite event, so 1-for-1000 then 1000-for-1 nets to no change
  instead of leaving a 1000x factor. Same-direction ratios are never legs, and a measured price gap that
  matches one leg but not the composite leaves the rows as a conflict.

### Changed
- Ratios are exact numbers: `ratio_from` / `ratio_to` are Float64 in `ACTION_SCHEMA` and the CSV reader
  (integer input still works); `recorded_mislabels` keys match 1, 1.0 and 1.00 alike instead of
  truncating, so a 1:1.0526 event round-trips exactly.
- `collapse_all` stats gain a `composite` stage.

### Checked on real data (privately; no data in this repository)
- Feed checks on 200 stocks over 21 months against a witness built from 1-minute bars: no flags on the
  most liquid names, no date-shift spans; frozen-close runs were separated into genuinely illiquid
  stretches (both feeds flat) and stale-feed stretches (the witness moved, up to 19 sessions long).
- Identity on 150 reused tickers: keyed on the ticker, a later company's split rescaled 29,879 of the
  earlier company's daily bars; keyed on the security, none.
- Same-day legs in a 14,213-row split-claims table: one reverse+forward pair composed to a no-op; one
  same-direction pair (an approved range quoted next to the final ratio) correctly left as a conflict —
  the case that made same-direction ratios ineligible.

## 0.3.0 — 2026-10-07

### Added
- `split_adjustment_tool.breaks`: when the price feed itself goes wrong.
  `scan_unexplained_jumps` (persistent jumps no claim explains, volatility-scaled; records the day a
  break reverted; one-day windows included), `classify_jumps` (independent witness series ->
  `PROVIDER_BREAK` / `REAL_MOVE` / `UNEXPLAINED`), `check_claims_against_witness` (drops a claim only
  the broken feed "confirms"), `BreakConfig`.
- Demo: provider-break benchmark section (break from day X, window X..Y, one-day break, bogus claim on a
  break, unclaimed real split, clean false-break control); every class must reach 95%.
- Real-data check documented: 91 of 104 real bad windows (220 bars, 94 stocks) flagged on their first
  day, zero provider-break flags on good days or on clean data.

## 0.2.1 — 2026-10-07

### Added
- `apply_split_adjustment` is time-zone agnostic: `timestamp_col=` accepts UTC, any zone or naive
  timestamps and derives each row's exchange-local trading date (`exchange_tz`, default New York;
  `tz_by_symbol` per-symbol overrides; `naive_timestamps_tz`, default UTC). A Datetime-typed `date`
  column is handled the same way; a Date-typed `date` is used as is. Your own columns are untouched.
- Intraday use documented and tested (1-hour and 1-minute bars, after-hours and pre-market).

## 0.2.0 — 2026-10-07

Renamed from `tapetruth` to **split-adjustment-tool** (package `split_adjustment_tool`).

### Added
- `split_adjustment_tool.factors`: `apply_split_adjustment` (price and volume factors, adjusted
  OHLC and volume, CRSP-style anchoring) and `build_factor_table` (sparse one-row-per-event form).
- Reconciler compares event by event: new `DATE_DIFFERS` class and an `event_issues` list
  (`RATIO_DIFFERS`, `DATE_DIFFERS`, `ONLY_OURS`, `ONLY_REFERENCE`); sparse tables are aligned as-of.
- Guard output column `confirmed_extreme` and `repair_source="confirmed"`.
- `ChainConfig.max_gap_calendar_days`; phantom-refuter stats `unmeasurable_kept`.
- Benchmark classes `serial_reverse_split`, `halted_reverse_split`, `real_small_ratio_split`,
  `small_ratio_phantom` (known gap); per-class floor for real events in the demo verdict.

### Fixed
- Two real same-ratio splits close together (serial micro-cap reverse splits) were merged into one.
  Clustering is now anchored on each cluster's first row and split wherever the price tape shows
  separate jumps.
- A real reverse split after a trading halt could be rejected because drift during the halt cancelled
  its jump. Gaps across holes longer than 10 calendar days are now unmeasurable: kept and reported.
- Reconciler's median-based classification hid wrong splits covering under half the history and
  5-day misdates; levels are now judged on the worst date and events are matched individually.
- Bad-print guard flattened real one-day crashes (every limit was measured against the close, and a
  confirming intraday source was discarded). Wicks are now judged against the day's body; an open
  inside the intraday range is kept; extremes the intraday data confirms are flagged, not overwritten.

### Changed
- Docs state the real scope of the price check (ratios of at least 2x) and list known gaps.
- Development status set to Alpha.

## 0.1.0 — 2026-07-19
First public release as `tapetruth`.
