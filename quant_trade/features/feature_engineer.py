"""Utilities for merging feature blocks safely."""

from __future__ import annotations

import pandas as pd


def safe_merge(
    base_k: pd.DataFrame,
    *ext_blocks: pd.DataFrame | None,
    core_cols: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    """Safely align and merge external feature blocks onto a base kline.

    Parameters
    ----------
    base_k
        Base kline DataFrame indexed by ``open_time``.
    *ext_blocks
        Additional feature blocks to merge. ``None`` or empty DataFrames are
        ignored. Each block is reindexed to ``base_k``'s index, shifted by one
        step to avoid lookahead bias and cast to ``float64`` before merging.
    core_cols
        Core OHLCV columns used to identify valid rows.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with aligned features where rows with all core
        columns missing are removed.
    """

    if not isinstance(base_k, pd.DataFrame):
        raise TypeError("base_k must be a pandas DataFrame")

    base = base_k.dropna(how="all", subset=list(core_cols))
    blocks: list[pd.DataFrame] = [base]

    for blk in ext_blocks:
        if blk is None:
            continue
        if hasattr(blk, "empty") and blk.empty:
            continue
        aligned = blk.reindex(base.index).shift(1).astype("float64")
        blocks.append(aligned)

    merged = pd.concat(blocks, axis=1)
    return merged.dropna(subset=list(core_cols))
