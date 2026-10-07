"""split-adjustment-tool -- records are hypotheses; the tape is truth.

A vendor-neutral, pluggable engine for verifying and repairing corporate-action claims and
OHLCV bar-data prints against your own price tape, instead of trusting any single feed at
face value.

Quickstart
----------
>>> from split_adjustment_tool import ChainConfig, collapse_all, make_close_series_fn, make_gap_fn
>>> from split_adjustment_tool.providers import CSVBarProvider, CSVActionSource
>>> bars = CSVBarProvider("my_bars_dir/")
>>> actions = CSVActionSource("my_actions.csv").get_actions()
>>> series_fn = make_close_series_fn(bars)
>>> collapsed, stats = collapse_all(actions, gap_fn=make_gap_fn(series_fn=series_fn))

Or see the benchmark: ``python -m split_adjustment_tool.demo``.

Full method: ``docs/STANDARD.md``. Full API: see each module's docstring
(:mod:`split_adjustment_tool.chain`, :mod:`split_adjustment_tool.guard`, :mod:`split_adjustment_tool.locator`,
:mod:`split_adjustment_tool.reconcile`, :mod:`split_adjustment_tool.providers`, :mod:`split_adjustment_tool.gauntlet`).
"""
from __future__ import annotations

__version__ = "0.2.0"

from split_adjustment_tool.chain import (
    TAPE_CONFIRMED_SOURCE,
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
from split_adjustment_tool.factors import apply_split_adjustment, build_factor_table
from split_adjustment_tool.gauntlet import (
    DEFECT_CLASSES,
    GauntletConfig,
    build_gauntlet_universe,
    run_gauntlet,
    score_run,
)
from split_adjustment_tool.guard import GuardConfig, apply_bad_print_guard
from split_adjustment_tool.locator import find_unique_boundary
from split_adjustment_tool.providers import (
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
from split_adjustment_tool.reconcile import (
    ReconcileConfig,
    ReconciliationClass,
    reconcile_all,
    reconcile_symbol,
    reconciliation_gate,
)

__all__ = [
    "apply_split_adjustment", "build_factor_table",
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
