import pandas as pd
import numpy as np

def health_check(features: dict, abs_clip: dict = None):
    """
    features: 特征字典或pandas.Series
    abs_clip: 可选，给需要clip的特征传递clip范围的dict，比如{'atr_pct_1h': (0,0.2)}
    """
    for k, v in features.items():
        if pd.isna(v) or not np.isfinite(v):
            features[k] = 0.0
        elif isinstance(v, (float, int)):
            features[k] = min(max(v, -1e4), 1e4)
    if abs_clip:
        for ck, (vmin, vmax) in abs_clip.items():
            features[ck] = abs(features.get(ck, 0))
            features[ck] = min(max(features[ck], vmin), vmax)
    return features


def apply_health_check_df(df: pd.DataFrame, abs_clip: dict = None) -> pd.DataFrame:
    def _row_func(row):
        return pd.Series(health_check(row.to_dict(), abs_clip=abs_clip))
    return df.apply(_row_func, axis=1)