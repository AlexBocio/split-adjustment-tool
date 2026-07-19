"""Gauntlet round-trip test -- asserts detection/false-positive rates stay above the floors
the demo (and README) advertise. Uses a smaller-than-default universe so the full test suite
stays fast; the floors are the same ones `python -m tapetruth.demo` checks."""
from __future__ import annotations

from tapetruth.gauntlet import (
    DEFECT_CLASSES,
    GauntletConfig,
    build_gauntlet_universe,
    demonstrate_vintage_awareness,
    run_gauntlet,
    score_run,
)

DETECTION_FLOOR = 0.80
FALSE_POSITIVE_CEILING = 0.02

_SMALL_CONFIG = GauntletConfig(n_instances_per_class=8, n_clean_symbols=20, n_trading_days=252, seed=7)


def test_universe_has_expected_shape():
    universe = build_gauntlet_universe(_SMALL_CONFIG)
    n_expected_plant_symbols = 12 * _SMALL_CONFIG.n_instances_per_class
    assert len(universe.plants) == n_expected_plant_symbols
    assert len(universe.clean_symbols) == _SMALL_CONFIG.n_clean_symbols
    assert len(universe.bars.symbols()) == n_expected_plant_symbols + _SMALL_CONFIG.n_clean_symbols
    # every symbol is synthetic -- starts with TEST, never a real ticker
    assert all(s.startswith("TEST") for s in universe.bars.symbols())
    assert set(DEFECT_CLASSES) == {p.defect_class for p in universe.plants}


def test_gauntlet_meets_acceptance_floors():
    universe = build_gauntlet_universe(_SMALL_CONFIG)
    result = run_gauntlet(universe)
    scorecard = score_run(universe, result)

    assert scorecard.overall_detection_rate >= DETECTION_FLOOR, (
        f"detection {scorecard.overall_detection_rate:.3f} below floor {DETECTION_FLOOR} -- "
        f"per-class: {[(s.defect_class, s.rate) for s in scorecard.class_scores if s.category == 'defect']}"
    )
    assert scorecard.clean_false_positive_rate <= FALSE_POSITIVE_CEILING, (
        f"false-positive rate {scorecard.clean_false_positive_rate:.4f} above ceiling "
        f"{FALSE_POSITIVE_CEILING}"
    )
    # legitimate-preservation classes should also be near-perfect (not gated by the
    # acceptance criteria directly, but this is the whole point of those 3 classes existing)
    assert scorecard.overall_legitimate_preserve_rate >= 0.90


def test_known_gap_class_is_reported_and_not_counted():
    universe = build_gauntlet_universe(_SMALL_CONFIG)
    result = run_gauntlet(universe)
    scorecard = score_run(universe, result)
    gap_classes = {s.defect_class for s in scorecard.known_gap_scores}
    assert gap_classes == {"bad_close_vspike"}
    defect_classes = {s.defect_class for s in scorecard.class_scores if s.category == "defect"}
    assert "bad_close_vspike" not in defect_classes


def test_demonstrate_vintage_awareness_shows_the_difference():
    universe = build_gauntlet_universe(_SMALL_CONFIG)
    demo = demonstrate_vintage_awareness(universe)
    assert demo["n_instances"] == _SMALL_CONFIG.n_instances_per_class
    assert demo["survived_without_vintage_awareness"] == 0
    assert demo["survived_with_vintage_awareness"] == demo["n_instances"]


def test_gauntlet_is_deterministic_given_a_seed():
    u1 = build_gauntlet_universe(GauntletConfig(n_instances_per_class=3, n_clean_symbols=2, n_trading_days=60, seed=42))
    u2 = build_gauntlet_universe(GauntletConfig(n_instances_per_class=3, n_clean_symbols=2, n_trading_days=60, seed=42))
    assert u1.actions_df.equals(u2.actions_df)
    assert u1.bars.symbols() == u2.bars.symbols()
    for sym in u1.bars.symbols():
        assert u1.bars.get_bars(sym).equals(u2.bars.get_bars(sym))
