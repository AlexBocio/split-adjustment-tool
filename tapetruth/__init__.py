"""tapetruth -- records are hypotheses; the tape is truth.

A vendor-neutral, pluggable engine for verifying and repairing corporate-action claims and
OHLCV bar-data prints against your own price tape, instead of trusting any single feed at
face value.

Quickstart
----------
>>> from tapetruth import ChainConfig, collapse_all, make_close_series_fn, make_gap_fn
>>> from tapetruth.providers import CSVBarProvider, CSVActionSource
>>> bars = CSVBarProvider("my_bars_dir/")
>>> actions = CSVActionSource("my_actions.csv").get_actions()
>>> series_fn = make_close_series_fn(bars)
>>> collapsed, stats = collapse_all(actions, gap_fn=make_gap_fn(series_fn=series_fn))

Or see the benchmark: ``python -m tapetruth.demo``.

Full method: ``docs/STANDARD.md``. Full API: see each module's docstring
(:mod:`tapetruth.chain`, :mod:`tapetruth.guard`, :mod:`tapetruth.locator`,
:mod:`tapetruth.reconcile`, :mod:`tapetruth.providers`, :mod:`tapetruth.gauntlet`).
"""
from __future__ import annotations

__version__ = "0.1.0"

from tapetruth.chain import (
    ChainConfig,
    TAPE_CONFIRMED_SOURCE,
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
from tapetruth.gauntlet import (
    DEFECT_CLASSES,
    GauntletConfig,
    build_gauntlet_universe,
    run_gauntlet,
    score_run,
)
from tapetruth.guard import GuardConfig, apply_bad_print_guard
from tapetruth.locator import find_unique_boundary
from tapetruth.providers import (
    ActionSource,
    BarProvider,
    CSVActionSource,
    CSVBarProvider,
    InMemoryActionSource,
    InMemoryBarProvider,
    InMemoryTruthProvider,
    ParquetActionSource,
    ParquetBarProvider,
    TruthProvider,
)
from tapetruth.reconcile import (
    ReconcileConfig,
    ReconciliationClass,
    reconcile_all,
    reconcile_symbol,
    reconciliation_gate,
)

__all__ = [
    "__version__",
    # chain
    "ChainConfig", "TAPE_CONFIRMED_SOURCE", "collapse_all", "collapse_duplicate_actions",
    "collapse_same_date_conflicts", "collapse_near_date_conflicts", "snap_actions_to_tape",
    "refute_phantom_actions", "drop_post_coverage_actions", "drop_recorded_mislabels",
    "make_close_series_fn", "make_gap_fn", "make_last_bar_date_fn",
    # locator
    "find_unique_boundary",
    # guard
    "GuardConfig", "apply_bad_print_guard",
    # reconcile
    "ReconcileConfig", "ReconciliationClass", "reconcile_symbol", "reconcile_all",
    "reconciliation_gate",
    # providers
    "BarProvider", "ActionSource", "TruthProvider",
    "InMemoryBarProvider", "InMemoryActionSource", "InMemoryTruthProvider",
    "CSVBarProvider", "CSVActionSource", "ParquetBarProvider", "ParquetActionSource",
    # gauntlet
    "GauntletConfig", "build_gauntlet_universe", "run_gauntlet", "score_run", "DEFECT_CLASSES",
]
