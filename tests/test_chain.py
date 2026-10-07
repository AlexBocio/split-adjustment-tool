"""Unit tests for tapetruth.chain -- synthetic fixtures only (TEST1..TESTn symbols)."""
from __future__ import annotations

import datetime as dt

import polars as pl

from tapetruth.chain import (
    ChainConfig,
    collapse_all,
    collapse_duplicate_actions,
    collapse_near_date_conflicts,
    collapse_same_date_conflicts,
    drop_post_coverage_actions,
    drop_recorded_mislabels,
    make_close_series_fn,
    make_gap_fn,
    make_last_bar_date_fn,
    refute_phantom_actions,
    snap_actions_to_tape,
)
from tapetruth.providers import InMemoryBarProvider


def _make_bars(symbol_prices: dict[str, list[float]], start: dt.date = dt.date(2022, 1, 3)) -> InMemoryBarProvider:
    bars = {}
    for sym, prices in symbol_prices.items():
        dates = [start + dt.timedelta(days=i) for i in range(len(prices))]
        bars[sym] = pl.DataFrame({
            "date": dates,
            "open": prices, "high": [p * 1.01 for p in prices], "low": [p * 0.99 for p in prices],
            "close": prices, "volume": [1000] * len(prices),
        }).with_columns(pl.col("date").cast(pl.Date))
    return InMemoryBarProvider(bars)


def _actions(rows: list[dict]) -> pl.DataFrame:
    schema = {"symbol": pl.String, "date": pl.Date, "ratio_from": pl.Int64,
              "ratio_to": pl.Int64, "source": pl.String}
    return pl.DataFrame(rows, schema=schema)


# --------------------------------------------------------------------------------------
# collapse_duplicate_actions
# --------------------------------------------------------------------------------------


def test_collapse_duplicate_actions_merges_clustered_same_ratio_rows():
    actions = _actions([
        {"symbol": "TEST1", "date": dt.date(2022, 3, 1), "ratio_from": 1, "ratio_to": 4, "source": "feed_a"},
        {"symbol": "TEST1", "date": dt.date(2022, 3, 5), "ratio_from": 1, "ratio_to": 4, "source": "feed_b"},
        {"symbol": "TEST1", "date": dt.date(2022, 3, 10), "ratio_from": 1, "ratio_to": 4, "source": "feed_c"},
    ])
    collapsed, stats = collapse_duplicate_actions(actions)
    assert collapsed.height == 1
    assert stats["n_rows_collapsed"] == 2
    assert stats["n_clusters_multi_row"] == 1


def test_collapse_duplicate_actions_keeps_distinct_events_far_apart():
    actions = _actions([
        {"symbol": "TEST2", "date": dt.date(2018, 1, 1), "ratio_from": 1, "ratio_to": 2, "source": "feed"},
        {"symbol": "TEST2", "date": dt.date(2022, 1, 1), "ratio_from": 1, "ratio_to": 2, "source": "feed"},
    ])
    collapsed, stats = collapse_duplicate_actions(actions)
    assert collapsed.height == 2
    assert stats["n_rows_collapsed"] == 0


def test_collapse_duplicate_actions_prefers_configured_source():
    actions = _actions([
        {"symbol": "TEST3", "date": dt.date(2022, 1, 1), "ratio_from": 1, "ratio_to": 4, "source": "low_pri"},
        {"symbol": "TEST3", "date": dt.date(2022, 1, 3), "ratio_from": 1, "ratio_to": 4, "source": "tape_confirmed"},
    ])
    config = ChainConfig(source_priority=["tape_confirmed", "low_pri"])
    collapsed, _ = collapse_duplicate_actions(actions, config=config)
    assert collapsed.height == 1
    assert collapsed["source"][0] == "tape_confirmed"


def test_collapse_duplicate_actions_idempotent():
    actions = _actions([
        {"symbol": "TEST4", "date": dt.date(2022, 1, 1), "ratio_from": 1, "ratio_to": 4, "source": "a"},
        {"symbol": "TEST4", "date": dt.date(2022, 1, 5), "ratio_from": 1, "ratio_to": 4, "source": "b"},
    ])
    once, _ = collapse_duplicate_actions(actions)
    twice, _ = collapse_duplicate_actions(once)
    assert once.equals(twice)


def test_collapse_duplicate_actions_empty_input():
    empty = _actions([])
    out, stats = collapse_duplicate_actions(empty)
    assert out.height == 0
    assert stats["n_input_rows"] == 0


# --------------------------------------------------------------------------------------
# collapse_same_date_conflicts / collapse_near_date_conflicts
# --------------------------------------------------------------------------------------


def test_collapse_same_date_conflicts_resolved_by_measured_gap():
    prices = [100.0] * 10 + [25.0] * 10  # true 1:4 split at index 10
    bars = _make_bars({"TEST5": prices})
    series_fn = make_close_series_fn(bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    d = bars.get_bars("TEST5")["date"][10]
    actions = _actions([
        {"symbol": "TEST5", "date": d, "ratio_from": 1, "ratio_to": 4, "source": "price_feed"},
        {"symbol": "TEST5", "date": d, "ratio_from": 4, "ratio_to": 1, "source": "text_feed"},  # fabricated inverse
    ])
    collapsed, stats = collapse_same_date_conflicts(actions, gap_fn=gap_fn)
    assert collapsed.height == 1
    row = collapsed.row(0, named=True)
    assert (row["ratio_from"], row["ratio_to"]) == (1, 4)
    assert stats["n_gap_resolved"] == 1


def test_collapse_same_date_conflicts_falls_back_to_source_rank_without_gap_fn():
    d = dt.date(2022, 5, 1)
    actions = _actions([
        {"symbol": "TEST6", "date": d, "ratio_from": 1, "ratio_to": 4, "source": "tape_confirmed"},
        {"symbol": "TEST6", "date": d, "ratio_from": 4, "ratio_to": 1, "source": "unranked"},
    ])
    config = ChainConfig(source_priority=["tape_confirmed"])
    collapsed, _ = collapse_same_date_conflicts(actions, config=config, gap_fn=None)
    assert collapsed.height == 1
    assert collapsed["source"][0] == "tape_confirmed"


def test_collapse_same_date_conflicts_drops_noop_rows():
    d = dt.date(2022, 5, 1)
    actions = _actions([
        {"symbol": "TEST7", "date": d, "ratio_from": 1, "ratio_to": 1, "source": "quiet_feed"},
        {"symbol": "TEST7", "date": d, "ratio_from": 1, "ratio_to": 4, "source": "real_feed"},
    ])
    collapsed, _ = collapse_same_date_conflicts(actions, gap_fn=None)
    assert collapsed.height == 1
    row = collapsed.row(0, named=True)
    assert (row["ratio_from"], row["ratio_to"]) == (1, 4)


def test_collapse_near_date_conflicts_resolved_by_measured_gap():
    prices = [10.0] * 10 + [100.0] * 10  # true 10:1 reverse split at index 10 (price x10)
    bars = _make_bars({"TEST8": prices})
    series_fn = make_close_series_fn(bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    d_true = bars.get_bars("TEST8")["date"][10]
    d_fake = d_true + dt.timedelta(days=4)
    actions = _actions([
        {"symbol": "TEST8", "date": d_true, "ratio_from": 10, "ratio_to": 1, "source": "text_feed"},
        {"symbol": "TEST8", "date": d_fake, "ratio_from": 30, "ratio_to": 1, "source": "price_feed"},
    ])
    collapsed, stats = collapse_near_date_conflicts(actions, gap_fn=gap_fn)
    assert collapsed.height == 1
    row = collapsed.row(0, named=True)
    assert (row["ratio_from"], row["ratio_to"]) == (10, 1)


def test_collapse_near_date_conflicts_leaves_far_apart_splits_alone():
    # genuinely serial reverse splits (a real-world pattern in some nano-cap names), gaps
    # larger than the window -> untouched
    actions = _actions([
        {"symbol": "TEST9", "date": dt.date(2022, 5, 11), "ratio_from": 20, "ratio_to": 1, "source": "a"},
        {"symbol": "TEST9", "date": dt.date(2022, 6, 23), "ratio_from": 15, "ratio_to": 1, "source": "a"},
    ])
    collapsed, stats = collapse_near_date_conflicts(actions, config=ChainConfig(near_date_window_days=10))
    assert collapsed.height == 2
    assert stats["n_conflict_clusters"] == 0


# --------------------------------------------------------------------------------------
# drop_post_coverage_actions
# --------------------------------------------------------------------------------------


def test_drop_post_coverage_actions_drops_late_claims():
    bars = _make_bars({"TEST10": [100.0] * 10})
    last_date_fn = make_last_bar_date_fn(series_fn=make_close_series_fn(bars))
    last = bars.get_bars("TEST10")["date"][-1]
    actions = _actions([
        {"symbol": "TEST10", "date": last - dt.timedelta(days=1), "ratio_from": 1, "ratio_to": 2, "source": "a"},
        {"symbol": "TEST10", "date": last + dt.timedelta(days=400), "ratio_from": 1, "ratio_to": 10, "source": "a"},
    ])
    out, stats = drop_post_coverage_actions(actions, last_date_fn=last_date_fn)
    assert out.height == 1
    assert stats["n_dropped"] == 1


def test_drop_post_coverage_actions_keeps_when_no_coverage():
    bars = _make_bars({})
    last_date_fn = make_last_bar_date_fn(series_fn=make_close_series_fn(bars))
    actions = _actions([
        {"symbol": "UNKNOWN", "date": dt.date(2022, 1, 1), "ratio_from": 1, "ratio_to": 2, "source": "a"},
    ])
    out, stats = drop_post_coverage_actions(actions, last_date_fn=last_date_fn)
    assert out.height == 1
    assert stats["n_dropped"] == 0


# --------------------------------------------------------------------------------------
# drop_recorded_mislabels
# --------------------------------------------------------------------------------------


def test_drop_recorded_mislabels_drops_only_matching_key():
    actions = _actions([
        {"symbol": "TEST11", "date": dt.date(2022, 5, 1), "ratio_from": 1, "ratio_to": 2, "source": "vendor"},
        {"symbol": "TEST11", "date": dt.date(2022, 6, 1), "ratio_from": 1, "ratio_to": 3, "source": "vendor"},
    ])
    config = ChainConfig(recorded_mislabels={("TEST11", "2022-05-01", 1, 2): "cited spin-off"})
    out, stats = drop_recorded_mislabels(actions, config=config)
    assert out.height == 1
    assert stats["n_dropped"] == 1
    assert out["date"][0] == dt.date(2022, 6, 1)


def test_recorded_mislabel_is_the_only_thing_that_catches_it():
    """The tape genuinely confirms the claimed ratio for a mislabeled spin-off -- proves
    that WITHOUT the citation, nothing in the tape-based chain drops it."""
    prices = [100.0] * 10 + [50.0] * 10  # tape genuinely shows a 1:2-shaped move
    bars = _make_bars({"TEST12": prices})
    series_fn = make_close_series_fn(bars)
    d = bars.get_bars("TEST12")["date"][10]
    actions = _actions([{"symbol": "TEST12", "date": d, "ratio_from": 1, "ratio_to": 2, "source": "vendor"}])

    no_citation, _ = collapse_all(
        actions, gap_fn=make_gap_fn(series_fn=series_fn),
        last_date_fn=make_last_bar_date_fn(series_fn=series_fn), series_fn=series_fn,
    )
    assert no_citation.height == 1  # tape can't tell -- survives without a citation

    config = ChainConfig(recorded_mislabels={("TEST12", d.isoformat(), 1, 2): "cited spin-off"})
    with_citation, _ = collapse_all(
        actions, config=config, gap_fn=make_gap_fn(series_fn=series_fn),
        last_date_fn=make_last_bar_date_fn(series_fn=series_fn), series_fn=series_fn,
    )
    assert with_citation.height == 0  # citation is the only thing that can drop it


# --------------------------------------------------------------------------------------
# snap_actions_to_tape
# --------------------------------------------------------------------------------------


def test_snap_actions_to_tape_corrects_misdated_action():
    prices = [100.0] * 20 + [25.0] * 20
    bars = _make_bars({"TEST13": prices})
    series_fn = make_close_series_fn(bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    true_date = bars.get_bars("TEST13")["date"][20]
    wrong_date = true_date + dt.timedelta(days=2)
    actions = _actions([{"symbol": "TEST13", "date": wrong_date, "ratio_from": 1, "ratio_to": 4, "source": "a"}])
    out, stats = snap_actions_to_tape(actions, series_fn=series_fn, gap_fn=gap_fn)
    assert stats["n_snapped"] == 1
    assert out["date"][0] == true_date


def test_snap_actions_to_tape_leaves_small_ratios_alone():
    prices = [100.0] * 20 + [90.0] * 20  # small ~10% move -- not a "large" ratio
    bars = _make_bars({"TEST14": prices})
    series_fn = make_close_series_fn(bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    true_date = bars.get_bars("TEST14")["date"][20]
    wrong_date = true_date + dt.timedelta(days=2)
    actions = _actions([{"symbol": "TEST14", "date": wrong_date, "ratio_from": 10, "ratio_to": 9, "source": "a"}])
    out, stats = snap_actions_to_tape(actions, series_fn=series_fn, gap_fn=gap_fn)
    assert stats["n_snapped"] == 0
    assert out["date"][0] == wrong_date


# --------------------------------------------------------------------------------------
# refute_phantom_actions
# --------------------------------------------------------------------------------------


def test_refute_phantom_actions_drops_lone_phantom():
    bars = _make_bars({"TEST15": [100.0] * 20})  # flat, no event
    series_fn = make_close_series_fn(bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    d = bars.get_bars("TEST15")["date"][10]
    actions = _actions([{"symbol": "TEST15", "date": d, "ratio_from": 1, "ratio_to": 3, "source": "misparsed"}])
    out, stats = refute_phantom_actions(actions, gap_fn=gap_fn)
    assert out.height == 0
    assert stats["n_refuted"] == 1


def test_refute_phantom_actions_keeps_real_split():
    prices = [10.0] * 10 + [100.0] * 10  # real 10:1 reverse split (price x10)
    bars = _make_bars({"TEST16": prices})
    series_fn = make_close_series_fn(bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    d = bars.get_bars("TEST16")["date"][10]
    actions = _actions([{"symbol": "TEST16", "date": d, "ratio_from": 10, "ratio_to": 1, "source": "price_feed"}])
    out, stats = refute_phantom_actions(actions, gap_fn=gap_fn)
    assert out.height == 1
    assert stats["n_refuted"] == 0


def test_refute_phantom_actions_vintage_guard_protects_pre_adjusted_split():
    bars = _make_bars({"TEST17": [100.0] * 20})  # flat -- already pre-adjusted upstream
    series_fn = make_close_series_fn(bars)
    gap_fn = make_gap_fn(series_fn=series_fn)
    d = bars.get_bars("TEST17")["date"][10]
    actions = _actions([{"symbol": "TEST17", "date": d, "ratio_from": 1, "ratio_to": 3, "source": "vendor"}])

    without, _ = refute_phantom_actions(actions, gap_fn=gap_fn, vintage_fn=None)
    assert without.height == 0  # wrongly refuted -- indistinguishable from a true phantom

    with_vintage, _ = refute_phantom_actions(
        actions, gap_fn=gap_fn, vintage_fn=lambda symbol, date: "ADJUSTED")
    assert with_vintage.height == 1  # correctly protected


def test_refute_phantom_actions_no_coverage_keeps_action():
    actions = _actions([{"symbol": "UNKNOWN", "date": dt.date(2022, 1, 1), "ratio_from": 1,
                         "ratio_to": 3, "source": "a"}])
    out, stats = refute_phantom_actions(actions, gap_fn=make_gap_fn(series_fn=make_close_series_fn(_make_bars({}))))
    assert out.height == 1
    assert stats["n_refuted"] == 0


# --------------------------------------------------------------------------------------
# collapse_all end-to-end
# --------------------------------------------------------------------------------------


def test_collapse_all_end_to_end():
    prices = [100.0] * 20 + [25.0] * 20
    bars = _make_bars({"TEST18": prices})
    series_fn = make_close_series_fn(bars)
    d = bars.get_bars("TEST18")["date"][20]
    actions = _actions([
        {"symbol": "TEST18", "date": d, "ratio_from": 1, "ratio_to": 4, "source": "a"},
        {"symbol": "TEST18", "date": d + dt.timedelta(days=3), "ratio_from": 1, "ratio_to": 4, "source": "b"},
        {"symbol": "TEST18", "date": d - dt.timedelta(days=2), "ratio_from": 1, "ratio_to": 4, "source": "c"},
    ])
    collapsed, stats = collapse_all(
        actions, gap_fn=make_gap_fn(series_fn=series_fn),
        last_date_fn=make_last_bar_date_fn(series_fn=series_fn), series_fn=series_fn,
    )
    assert collapsed.height == 1
    assert set(stats.keys()) == {"mislabel", "post_coverage", "duplicate", "same_date",
                                 "near_date", "snap", "phantom"}


def test_collapse_all_empty_input():
    empty = _actions([])
    collapsed, stats = collapse_all(empty)
    assert collapsed.height == 0


# --- M3 regression: serial reverse splits must not be merged (specialist finding C3) ---
def _serial_reverse_tape():
    import datetime as _dt
    days = [_dt.date(2024, 1, 1) + _dt.timedelta(days=i) for i in range(400)]
    px, p = [], 1.0
    for i in range(400):
        if i == 100:
            p *= 10
        if 100 < i < 190:
            p *= 0.975
        if i == 190:
            p *= 10
        px.append(p)
    bars = pl.DataFrame({"symbol": ["TESTSRS"] * 400, "date": days, "open": px, "high": px,
                         "low": px, "close": px, "volume": [1000] * 400})
    return days, bars


def test_two_tape_confirmed_serial_reverse_splits_are_not_merged():
    from tapetruth import collapse_all, make_close_series_fn, make_gap_fn, make_last_bar_date_fn
    from tapetruth.providers import InMemoryBarProvider
    days, bars = _serial_reverse_tape()
    acts = pl.DataFrame({"symbol": ["TESTSRS", "TESTSRS"], "date": [days[100], days[190]],
                         "ratio_from": [10, 10], "ratio_to": [1, 1]}).with_columns(pl.col("date").cast(pl.Date))
    sf = make_close_series_fn(InMemoryBarProvider({"TESTSRS": bars}))
    out, _ = collapse_all(acts, gap_fn=make_gap_fn(series_fn=sf),
                          last_date_fn=make_last_bar_date_fn(series_fn=sf), series_fn=sf)
    assert out.height == 2
    assert sorted(out["date"].to_list()) == [days[100], days[190]]


def test_duplicate_rows_of_one_event_still_collapse_with_tape():
    from tapetruth import collapse_all, make_close_series_fn, make_gap_fn, make_last_bar_date_fn
    from tapetruth.providers import InMemoryBarProvider
    days, bars = _serial_reverse_tape()
    acts = pl.DataFrame({"symbol": ["TESTSRS"] * 3, "date": [days[100], days[100], days[98]],
                         "ratio_from": [10, 10, 10], "ratio_to": [1, 1, 1],
                         "source": ["a", "b", "c"]}).with_columns(pl.col("date").cast(pl.Date))
    sf = make_close_series_fn(InMemoryBarProvider({"TESTSRS": bars}))
    out, _ = collapse_all(acts, gap_fn=make_gap_fn(series_fn=sf),
                          last_date_fn=make_last_bar_date_fn(series_fn=sf), series_fn=sf)
    assert out.height == 1
    assert out["date"][0] == days[100]


def test_anchor_clustering_does_not_chain():
    # rows at day 0, 140, 280 (window 150): 0 and 140 may merge, 280 must stay separate
    import datetime as _dt
    d0 = _dt.date(2020, 1, 1)
    acts = pl.DataFrame({"symbol": ["TESTCH"] * 3,
                         "date": [d0, d0 + _dt.timedelta(days=140), d0 + _dt.timedelta(days=280)],
                         "ratio_from": [1, 1, 1], "ratio_to": [2, 2, 2]}).with_columns(pl.col("date").cast(pl.Date))
    out, _ = collapse_duplicate_actions(acts)
    assert out.height == 2


# --- M4 regression: a real reverse split after a long halt must not be refuted (C4) ---
def test_reverse_split_after_halt_with_offsetting_drift_is_kept_and_flagged():
    import datetime as _dt
    from tapetruth import collapse_all, make_close_series_fn, make_gap_fn, make_last_bar_date_fn
    from tapetruth.providers import InMemoryBarProvider
    start = _dt.date(2024, 1, 1)
    days = [start + _dt.timedelta(days=i) for i in range(100)]          # trades at 1.00
    resume = start + _dt.timedelta(days=160)                            # 60-day halt
    days += [resume + _dt.timedelta(days=i) for i in range(100)]        # trades at 1.00 again
    px = [1.0] * 200   # 1:10 reverse split during the halt, offset exactly by a 90% collapse
    bars = pl.DataFrame({"symbol": ["TESTHALT"] * 200, "date": days, "open": px, "high": px,
                         "low": px, "close": px, "volume": [1000] * 200})
    acts = pl.DataFrame({"symbol": ["TESTHALT"], "date": [resume], "ratio_from": [10],
                         "ratio_to": [1]}).with_columns(pl.col("date").cast(pl.Date))
    sf = make_close_series_fn(InMemoryBarProvider({"TESTHALT": bars}))
    out, st = collapse_all(acts, gap_fn=make_gap_fn(series_fn=sf),
                           last_date_fn=make_last_bar_date_fn(series_fn=sf), series_fn=sf)
    assert out.height == 1
    assert st["phantom"]["n_refuted"] == 0
    assert st["phantom"]["n_unmeasurable_kept"] == 1


def test_phantom_on_continuous_tape_still_refuted():
    import datetime as _dt
    from tapetruth import collapse_all, make_close_series_fn, make_gap_fn, make_last_bar_date_fn
    from tapetruth.providers import InMemoryBarProvider
    days = [_dt.date(2024, 1, 1) + _dt.timedelta(days=i) for i in range(200)]
    px = [50.0] * 200
    bars = pl.DataFrame({"symbol": ["TESTPH"] * 200, "date": days, "open": px, "high": px,
                         "low": px, "close": px, "volume": [1000] * 200})
    acts = pl.DataFrame({"symbol": ["TESTPH"], "date": [days[120]], "ratio_from": [1],
                         "ratio_to": [4]}).with_columns(pl.col("date").cast(pl.Date))
    sf = make_close_series_fn(InMemoryBarProvider({"TESTPH": bars}))
    out, st = collapse_all(acts, gap_fn=make_gap_fn(series_fn=sf),
                           last_date_fn=make_last_bar_date_fn(series_fn=sf), series_fn=sf)
    assert out.height == 0 and st["phantom"]["n_refuted"] == 1
