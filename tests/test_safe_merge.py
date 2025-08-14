import warnings

import numpy as np
import pandas as pd

from quant_trade.features import safe_merge


def test_safe_merge_basic():
    idx = pd.RangeIndex(3)
    base = pd.DataFrame(
        {
            "open": [1.0, 2.0, np.nan],
            "high": [1.0, 2.0, np.nan],
            "low": [1.0, 2.0, np.nan],
            "close": [1.0, 2.0, np.nan],
            "volume": [1.0, 2.0, np.nan],
        },
        index=idx,
    )
    ext = pd.DataFrame({"feat": [10, 20, 30]}, index=idx)

    merged = safe_merge(base, ext)

    assert len(merged) == 2
    assert pd.isna(merged["feat"].iloc[0])
    assert merged["feat"].iloc[1] == 10

    naive = pd.concat([base, ext], axis=1)
    assert merged.isna().mean().mean() < naive.isna().mean().mean()

    with warnings.catch_warnings(record=True) as w:
        merged.isna().mean()
        assert not any("All-NaN" in str(warn.message) for warn in w)
