"""Utility helpers for parsing numeric inputs that arrive as free text."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Precompile regexes so they can be reused across calls without extra overhead.
THIN_SPACE_PATTERN = re.compile(r"\s+")
THOUSANDS_PATTERN = re.compile(r"[.,\u00A0\u202F](?=\d{3}(?:\D|$))")
APOSTROPHE_PATTERN = re.compile(r"[\'\u2018\u2019]")


def normalize_animal_counts(
    series: pd.Series | Any, *, index: pd.Index | None = None
) -> pd.Series:
    """Return a rounded integer series from arbitrary user inputs.

    The head-count fields often arrive from Excel/Google Sheets with thousand
    separators (``1.200`` or ``1,200``) and, in some cases, with a leading
    apostrophe (``'1.200``) to force text formatting. These characters cause
    ``pd.to_numeric`` to fall back to ``NaN`` and the totals end up at cero.
    We strip whitespace, apostrophes and thin spaces, normalise the decimal
    separator and coerce the values into integers so that the sums are accurate.
    """

    if series is None:
        if index is not None:
            return pd.Series(0, index=index, dtype=int)
        return pd.Series(dtype=int)

    if isinstance(series, pd.Series):
        if index is None:
            index = series.index
        work = series.astype(str).str.strip()
    else:
        if index is None and hasattr(series, "index"):
            index = series.index  # type: ignore[attr-defined]
        series = pd.Series(series, index=index)
        work = series.astype(str).str.strip()

    if work.empty:
        if index is not None:
            return pd.Series(0, index=index, dtype=int)
        return pd.Series(dtype=int)

    work = work.str.replace(THIN_SPACE_PATTERN, "", regex=True)
    work = work.str.replace(APOSTROPHE_PATTERN, "", regex=True)
    work = work.str.replace(THOUSANDS_PATTERN, "", regex=True)
    work = work.str.replace(",", ".", regex=False)

    numeric = pd.to_numeric(work, errors="coerce").fillna(0.0)
    return numeric.round().astype(int)


__all__ = ["normalize_animal_counts"]

