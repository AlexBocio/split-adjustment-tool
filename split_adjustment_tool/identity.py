"""Permanent security IDs: ticker changes and reused tickers.

A ticker is not an identity. Companies rename (FB -> META), and a ticker freed by a delisting is
reused by an unrelated company years later. Keyed on the ticker alone, a split of the OLD company
gets applied to the NEW company's prices (and a renamed company's history is cut in two).

This module adds an identity layer WITHOUT changing any other module: you describe which ticker each
security traded under and when (a *symbol history*), then

* :func:`rekey_by_security` relabels your split claims (or any dated rows) from ticker to security ID,
  using each row's date; and
* :class:`SecurityBarProvider` serves bars by security ID, stitching a renamed security's tickers
  together and cutting a reused ticker at the hand-over.

Every other function in the package then works on security IDs exactly as it worked on tickers.

Symbol history schema (:data:`SYMBOL_HISTORY_SCHEMA`): ``security_id, symbol, valid_from, valid_to``.
``valid_to`` may be null (still trading under that ticker). Intervals are inclusive. A security can
have several rows (one per ticker it used); a ticker can have several rows (one per security that
used it), which must not overlap in time.
"""
from __future__ import annotations

import datetime as _dt

import polars as pl

from split_adjustment_tool.providers import BAR_SCHEMA

__all__ = [
    "SYMBOL_HISTORY_SCHEMA",
    "symbol_history",
    "validate_symbol_history",
    "rekey_by_security",
    "SecurityBarProvider",
]

SYMBOL_HISTORY_SCHEMA = {"security_id": pl.String, "symbol": pl.String, "valid_from": pl.Date, "valid_to": pl.Date}


def symbol_history(rows: list[tuple]) -> pl.DataFrame:
    """Build a symbol history from ``(security_id, symbol, valid_from, valid_to)`` tuples (``valid_to``
    may be ``None``). Validates it (see :func:`validate_symbol_history`) before returning."""
    df = pl.DataFrame(rows, schema=SYMBOL_HISTORY_SCHEMA, orient="row")
    validate_symbol_history(df)
    return df


def validate_symbol_history(history: pl.DataFrame) -> None:
    """Raise ``ValueError`` when an interval is inverted or one ticker is claimed by two securities
    at the same time (the history must say unambiguously who owned a ticker on any date)."""
    for r in history.iter_rows(named=True):
        if r["valid_to"] is not None and r["valid_to"] < r["valid_from"]:
            raise ValueError(f"inverted interval for {r['security_id']}/{r['symbol']}: "
                             f"{r['valid_from']} > {r['valid_to']}")
    far = _dt.date(9999, 12, 31)
    for sym in history["symbol"].unique().to_list():
        rows = sorted(history.filter(pl.col("symbol") == sym).iter_rows(named=True), key=lambda r: r["valid_from"])
        for a, b in zip(rows, rows[1:], strict=False):
            if (a["valid_to"] or far) >= b["valid_from"]:
                raise ValueError(f"ticker {sym} is claimed by {a['security_id']} and {b['security_id']} "
                                 f"at the same time ({b['valid_from']})")


def rekey_by_security(df: pl.DataFrame, history: pl.DataFrame, date_col: str = "date",
                      symbol_col: str = "symbol", keep_ticker_as: str | None = "ticker",
                      unmatched: str = "keep") -> pl.DataFrame:
    """Replace ``symbol_col`` with the security ID that owned that ticker on each row's date.

    ``keep_ticker_as`` keeps the original ticker in a new column (``None`` to drop it). Rows whose
    (ticker, date) no history interval covers are handled by ``unmatched``: ``"keep"`` leaves the
    ticker as the key (so data you have not mapped behaves exactly as before), ``"drop"`` removes
    them, ``"raise"`` raises ``ValueError``.
    """
    if unmatched not in ("keep", "drop", "raise"):
        raise ValueError("unmatched must be 'keep', 'drop' or 'raise'")
    far = _dt.date(9999, 12, 31)
    h = history.with_columns(pl.col("valid_to").fill_null(far))
    base = df.with_row_index("_ri")
    hits = (base.select("_ri", symbol_col, date_col)
                .join(h.rename({"symbol": symbol_col}), on=symbol_col, how="inner")
                .filter((pl.col(date_col) >= pl.col("valid_from")) & (pl.col(date_col) <= pl.col("valid_to")))
                .select("_ri", "security_id"))
    out = base.join(hits, on="_ri", how="left")
    n_unmatched = out["security_id"].null_count()
    if n_unmatched and unmatched == "raise":
        raise ValueError(f"{n_unmatched} row(s) have no symbol-history interval for their (ticker, date)")
    if n_unmatched and unmatched == "drop":
        out = out.filter(pl.col("security_id").is_not_null())
    if keep_ticker_as:
        out = out.with_columns(pl.col(symbol_col).alias(keep_ticker_as))
    out = out.with_columns(pl.coalesce(pl.col("security_id"), pl.col(symbol_col)).alias(symbol_col))
    return out.drop("_ri", "security_id").select(
        [c for c in df.columns] + ([keep_ticker_as] if keep_ticker_as and keep_ticker_as not in df.columns else []))


class SecurityBarProvider:
    """A :class:`~split_adjustment_tool.providers.BarProvider` keyed by SECURITY ID over a ticker-keyed one.

    ``get_bars(security_id)`` returns the security's bars across every ticker it traded under, each
    ticker's bars cut to the interval the security owned it -- so a renamed company is one
    continuous history and a reused ticker never mixes two companies. Pass it anywhere a
    ``BarProvider`` is accepted (e.g. :func:`~split_adjustment_tool.chain.make_close_series_fn`).
    Unknown IDs fall through to the underlying provider unchanged (an unmapped ticker still works).
    """

    def __init__(self, bars, history: pl.DataFrame):
        validate_symbol_history(history)
        self._bars = bars
        self._h = history

    def get_bars(self, security_id: str) -> pl.DataFrame:
        rows = self._h.filter(pl.col("security_id") == security_id).sort("valid_from")
        if rows.height == 0:
            return self._bars.get_bars(security_id)
        parts = []
        for r in rows.iter_rows(named=True):
            b = self._bars.get_bars(r["symbol"])
            if b is None or b.height == 0:
                continue
            b = b.filter(pl.col("date") >= r["valid_from"])
            if r["valid_to"] is not None:
                b = b.filter(pl.col("date") <= r["valid_to"])
            parts.append(b)
        if not parts:
            return pl.DataFrame(schema=BAR_SCHEMA)
        return pl.concat(parts, how="vertical_relaxed").unique("date", keep="first").sort("date")

    def symbols(self) -> list[str]:
        mapped = set(self._h["symbol"].to_list())
        return sorted(set(self._h["security_id"].to_list())
                      | {s for s in self._bars.symbols() if s not in mapped})
