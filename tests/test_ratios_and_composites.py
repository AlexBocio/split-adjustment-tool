"""v0.4 C3 (exact ratios) + C4 (same-day composite legs). Synthetic fixtures only."""
from __future__ import annotations

import datetime as dt

import polars as pl

from split_adjustment_tool import ChainConfig, build_factor_table, collapse_all, compose_same_day_legs
from split_adjustment_tool.providers import ACTION_SCHEMA, CSVActionSource

D = dt.date(2024, 6, 3)


def _acts(rows):
    return pl.DataFrame(rows, schema=ACTION_SCHEMA, orient="row")


def test_fractional_ratio_round_trips_exactly_through_clean_and_factors():
    acts = _acts([("TSTF", D, 1.0, 1.0526, "s", "split")])
    out, _ = collapse_all(acts)
    assert out["ratio_to"].to_list() == [1.0526]
    ft = build_factor_table(out, start_date=D - dt.timedelta(days=30))
    assert ft.filter(pl.col("date") < D)["price_factor"].to_list() == [1.0 / 1.0526]


def test_csv_ratios_are_read_as_exact_numbers(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("symbol,date,ratio_from,ratio_to\nTSTC,2024-06-03,1,1.0526\nTSTD,2024-06-03,3,2\n", encoding="utf-8")
    df = CSVActionSource(p).get_actions()
    assert df.schema["ratio_to"] == pl.Float64 and df["ratio_to"].to_list() == [1.0526, 2.0]


def test_recorded_mislabel_matches_integer_and_float_spellings():
    cfg = ChainConfig(recorded_mislabels={("TSTM", "2024-06-03", 1, 2): "public record: spin-off"})
    acts = _acts([("TSTM", D, 1.0, 2.0, "s", "split")])
    out, st = collapse_all(acts, config=cfg)
    assert out.height == 0 and st["mislabel"]["n_dropped"] == 1


def test_odd_lot_cash_out_legs_net_to_a_no_op_not_a_1000x_factor():
    acts = _acts([("TSTO", D, 1000.0, 1.0, "v", "reverse_split"),
                  ("TSTO", D, 1.0, 1000.0, "v", "forward_split")])
    out, st = compose_same_day_legs(acts)
    assert out.height == 1 and st["n_composed_groups"] == 1
    assert out["ratio_from"][0] == out["ratio_to"][0] == 1000.0 and out["type"][0] == "composite"
    ft = build_factor_table(out)
    assert ft["price_factor"].max() == 1.0          # no 1000x factor survives


def test_reverse_then_forward_composite_is_the_product():
    acts = _acts([("TSTP", D, 10.0, 1.0, "v", "reverse_split"), ("TSTP", D, 1.0, 2.0, "v", "forward_split")])
    out, _ = compose_same_day_legs(acts)
    assert (out["ratio_from"][0], out["ratio_to"][0]) == (10.0, 2.0)   # net 5:1 = price x5


def test_different_sources_on_one_day_are_competing_claims_not_legs():
    acts = _acts([("TSTQ", D, 1.0, 4.0, "a", "split"), ("TSTQ", D, 4.0, 1.0, "b", "reverse_split")])
    out, st = compose_same_day_legs(acts)
    assert out.height == 2 and st["n_composed_groups"] == 0


def test_tape_matching_one_leg_means_a_double_logged_event_not_a_composite():
    acts = _acts([("TSTR", D, 1.0, 4.0, "v", "split"), ("TSTR", D, 4.0, 1.0, "v", "reverse_split")])
    def gap_fn(symbol, first, last):
        return 0.25                                  # the tape fell 4x: the 1:4 forward split is real

    out, st = compose_same_day_legs(acts, gap_fn=gap_fn)
    assert out.height == 2 and st["n_left_as_conflict"] == 1


def test_same_ratio_twice_from_one_source_is_a_duplicate_not_legs():
    acts = _acts([("TSTS", D, 1.0, 2.0, "v", "split"), ("TSTS", D, 1.0, 2.0, "v", "split")])
    out, st = compose_same_day_legs(acts)
    assert st["n_composed_groups"] == 0
    final, _ = collapse_all(acts)
    assert final.height == 1


def test_composing_can_be_switched_off():
    acts = _acts([("TSTT", D, 10.0, 1.0, "v", "reverse_split"), ("TSTT", D, 1.0, 2.0, "v", "forward_split")])
    out, st = compose_same_day_legs(acts, config=ChainConfig(compose_same_source_legs=False))
    assert out.height == 2 and st["n_composed_groups"] == 0


def test_two_same_direction_ratios_from_one_source_are_not_legs():
    """Real-data shape: one SEC filing quoting an approved range (1-for-10) beside the final ratio (1-for-750).
    Multiplying them would fabricate a 7,500:1 reverse split; they are competing claims instead."""
    acts = _acts([("TSTL", D, 10.0, 1.0, "f", "reverse_split"), ("TSTL", D, 750.0, 1.0, "f", "reverse_split")])
    out, st = compose_same_day_legs(acts)
    assert out.height == 2 and st["n_composed_groups"] == 0 and st["n_left_as_conflict"] == 1
