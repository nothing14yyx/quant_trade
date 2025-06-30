class FeatureProcessor:
    """特征处理逻辑"""

    def __init__(self, feature_cols_1h, feature_cols_4h, feature_cols_d1):
        self.feature_cols = {
            "1h": feature_cols_1h,
            "4h": feature_cols_4h,
            "d1": feature_cols_d1,
        }

    def get_columns(self, period: str):
        return self.feature_cols.get(period, [])
