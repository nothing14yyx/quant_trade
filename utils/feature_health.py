import pandas as pd
import numpy as np

def health_check(features: dict, abs_clip: dict = None, zero_cols: list[str] | None = None):
    """
    features: 特征字典或pandas.Series
    abs_clip: 可选，给需要clip的特征传递clip范围的dict，比如{'atr_pct_1h': (0,0.2)}
    """
    zero_cols = zero_cols or []
    for k, v in features.items():
        if pd.isna(v) or not np.isfinite(v):
            if k in zero_cols:
                features[k] = 0.0
            else:
                features[k] = np.nan
        elif isinstance(v, (float, int)):
            features[k] = min(max(v, -1e4), 1e4)
    if abs_clip:
        for ck, (vmin, vmax) in abs_clip.items():
            features[ck] = abs(features.get(ck, 0))
            features[ck] = min(max(features[ck], vmin), vmax)
    return features


def apply_health_check_df(df: pd.DataFrame, abs_clip: dict = None, zero_cols: list[str] | None = None) -> pd.DataFrame:
    """向量化执行 ``health_check``，并生成 ``_isnan`` 标志列。"""

    zero_cols = zero_cols or []
    processed = df.clip(-1e4, 1e4)
    flags_df = processed.isna().astype(int)
    flags_df.columns = [f"{c}_isnan" for c in processed.columns]
    if zero_cols:
        for c in zero_cols:
            if c in processed:
                processed[c] = processed[c].fillna(0.0)

    if abs_clip:
        for ck, (vmin, vmax) in abs_clip.items():
            if ck in processed:
                processed[ck] = processed[ck].abs().clip(vmin, vmax)

    return pd.concat([processed, flags_df], axis=1)
