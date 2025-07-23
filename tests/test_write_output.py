import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import inspect
from quant_trade.feature_engineering import FeatureEngineer

def test_write_output_add_missing_cols(tmp_path):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql('CREATE TABLE features (symbol TEXT, open_time TEXT)')
    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe.engine = engine
    fe.merged_table_path = tmp_path / 'dummy.csv'
    df = pd.DataFrame({
        'symbol': ['BTCUSDT'],
        'open_time': [pd.Timestamp('2020-01-01 00:00:00')],
        'extra_col': [1.23]
    })
    fe._write_output(df, save_to_db=True, append=True)
    cols = [c['name'] for c in inspect(engine).get_columns('features')]
    assert 'extra_col' in cols
    out = pd.read_sql('features', engine)
    assert out['extra_col'].iloc[0] == pytest.approx(1.23)


def test_write_output_skip_duplicates(tmp_path):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            'CREATE TABLE features (symbol TEXT, open_time TEXT, val REAL, PRIMARY KEY (symbol, open_time))'
        )
        conn.exec_driver_sql(
            "INSERT INTO features (symbol, open_time, val) VALUES ('BTCUSDT', '2020-01-01 00:00:00', 1.0)"
        )

    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe.engine = engine
    fe.merged_table_path = tmp_path / 'dummy.csv'
    df = pd.DataFrame(
        {
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'open_time': [pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-01 01:00:00')],
            'val': [2.0, 3.0],
        }
    )

    fe._write_output(df, save_to_db=True, append=True)

    out = pd.read_sql('features', engine).sort_values('open_time')
    assert len(out) == 2
    assert out['val'].tolist() == [1.0, 3.0]
