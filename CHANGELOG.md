# Changelog

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
