from quant_trade.feature_engineering import FeatureEngineer, BASE_COLS


def test_feature_cols_filtered(tmp_path):
    feature_file = tmp_path / 'cols.txt'
    feature_file.write_text('\n'.join([
        'open',
        'high',
        'low',
        'close',
        'my_feat',
        'future_volatility',
        'other_feat',
    ]))
    cfg = tmp_path / 'cfg.yaml'
    cfg.write_text(f"""
feature_engineering:
  feature_cols_path: "{feature_file}"
  merged_table_path: "{tmp_path / 'merged.csv'}"
  scaler_path: "{tmp_path / 'scaler.json'}"
""")
    fe = FeatureEngineer(str(cfg))
    assert not set(BASE_COLS) & set(fe.feature_cols_all)
    assert set(fe.feature_cols_all) == {'my_feat', 'other_feat'}
