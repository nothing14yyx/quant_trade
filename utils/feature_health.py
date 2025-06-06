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
    """对 DataFrame 每行执行 ``health_check``，并生成 ``_isnan`` 标志列。

    该函数主要用于推理阶段的特征处理，既确保所有数值经过 ``health_check``
    规整，也会像训练阶段一样附加 ``_isnan`` 列，以便模型输入维度保持一致。
    ``abs_clip`` 参数会原样传递给 ``health_check``。
    """

    processed = df.apply(lambda row: pd.Series(health_check(row.to_dict(), abs_clip=abs_clip)), axis=1)

    flags_df = df.isna().astype(int)
    flags_df.columns = [f"{c}_isnan" for c in df.columns]

    processed = processed.fillna(0.0)

    return pd.concat([processed, flags_df], axis=1)
