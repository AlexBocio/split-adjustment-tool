"""python -m tapetruth.demo -- the 60-second gauntlet.

Builds a synthetic universe of corporate-action and OHLCV defects (planted, ground-truth
labeled), runs it through the full tapetruth engine, and prints a scorecard: how many of
each defect class did the engine catch, how often did it correctly leave legitimate data
alone, and where are its known, documented gaps.

No real ticker, no real price, no network access -- everything here is generated in-process.
"""
from __future__ import annotations

import sys
import time

from tapetruth import __version__
from tapetruth.gauntlet import (
    GauntletConfig,
    build_gauntlet_universe,
    demonstrate_vintage_awareness,
    run_gauntlet,
    score_run,
)

_DETECTION_FLOOR = 0.80
_FALSE_POSITIVE_CEILING = 0.02
_BAR_WIDTH = 24


def _bar(rate: float) -> str:
    filled = max(0, min(_BAR_WIDTH, round(rate * _BAR_WIDTH)))
    return "#" * filled + "-" * (_BAR_WIDTH - filled)


def _print_class_rows(scores) -> None:
    for s in sorted(scores, key=lambda s: -s.rate):
        print(f"  {s.defect_class:<26} [{_bar(s.rate)}] {s.rate * 100:5.1f}%  "
              f"({s.n_correct}/{s.n_instances})")


def main() -> int:
    print(f"tapetruth v{__version__} -- the 60-second gauntlet")
    print("records are hypotheses; the tape is truth.\n")

    t0 = time.time()
    config = GauntletConfig()
    print(f"Building a synthetic universe (seed={config.seed}, "
          f"{config.n_instances_per_class} instances/class, "
          f"{config.n_clean_symbols} clean control symbols, "
          f"{config.n_trading_days} trading days each)...")
    universe = build_gauntlet_universe(config)
    n_symbols = len(universe.bars.symbols())
    n_actions = universe.actions_df.height
    print(f"  {n_symbols:,} symbols, {n_actions:,} planted action claims, "
          f"{len(universe.plants):,} defect/legitimate instances across 12 classes\n")

    print("Running the collapse chain + bad-print guard...")
    result = run_gauntlet(universe)
    print(f"  done in {time.time() - t0:.1f}s\n")

    scorecard = score_run(universe, result)

    print("=" * 72)
    print("DEFECT DETECTION  (the engine should catch these)")
    print("=" * 72)
    _print_class_rows([s for s in scorecard.class_scores if s.category == "defect"])
    print(f"\n  OVERALL DETECTION: {scorecard.overall_detection_rate * 100:.1f}%  "
          f"(floor: {_DETECTION_FLOOR * 100:.0f}%)\n")

    print("=" * 72)
    print("FALSE-POSITIVE AVOIDANCE  (the engine should leave these alone)")
    print("=" * 72)
    _print_class_rows([s for s in scorecard.class_scores if s.category == "legitimate"])
    print(f"\n  CLEAN-SYMBOL FALSE-POSITIVE RATE: {scorecard.clean_false_positive_rate * 100:.2f}%  "
          f"(ceiling: {_FALSE_POSITIVE_CEILING * 100:.0f}%, "
          f"n={scorecard.n_clean_bars_checked:,} bars checked)\n")

    print("=" * 72)
    print("KNOWN GAPS  (documented, honestly reported, NOT counted above)")
    print("=" * 72)
    print("  Ground-truth check here is INVERTED from the sections above: `rate` is the")
    print("  fraction that correctly reproduce the DOCUMENTED, still-open gap (i.e. still go")
    print("  UNDETECTED, as recorded in docs/STANDARD.md). A high number is not a success --")
    print("  it confirms the limitation is real and still open; it would start FALLING the")
    print("  day someone ships a fix, which is the point of tracking it explicitly.\n")
    for s in scorecard.known_gap_scores:
        print(f"  {s.defect_class:<26} [{_bar(s.rate)}] {s.rate * 100:5.1f}% still undetected "
              f"({s.n_correct}/{s.n_instances})  -- see docs/STANDARD.md \"Known gaps\"")

    print()
    print("=" * 72)
    print("VINTAGE-AWARENESS  (what the hook actually buys you)")
    print("=" * 72)
    vdemo = demonstrate_vintage_awareness(universe)
    print(f"  {vdemo['n_instances']} real splits on a pre-adjusted vintage (tape shows no gap):")
    print(f"    without vintage_fn: {vdemo['survived_without_vintage_awareness']}/"
          f"{vdemo['n_instances']} survive  (WRONG -- indistinguishable from a phantom)")
    print(f"    with vintage_fn:    {vdemo['survived_with_vintage_awareness']}/"
          f"{vdemo['n_instances']} survive  (correct)")

    passed = (scorecard.overall_detection_rate >= _DETECTION_FLOOR
             and scorecard.clean_false_positive_rate <= _FALSE_POSITIVE_CEILING)
    print()
    print("=" * 72)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}  "
          f"(detection {scorecard.overall_detection_rate * 100:.1f}% "
          f"{'>=' if scorecard.overall_detection_rate >= _DETECTION_FLOOR else '<'} "
          f"{_DETECTION_FLOOR * 100:.0f}%, "
          f"false-positive {scorecard.clean_false_positive_rate * 100:.2f}% "
          f"{'<=' if scorecard.clean_false_positive_rate <= _FALSE_POSITIVE_CEILING else '>'} "
          f"{_FALSE_POSITIVE_CEILING * 100:.0f}%)")
    print("=" * 72)
    print(f"\nTotal time: {time.time() - t0:.1f}s. Full method: docs/STANDARD.md")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
