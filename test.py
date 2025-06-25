import pandas as pd

from quant_trade import FeatureEngineer
from quant_trade.generate_signal_from_db import connect_mysql, load_config

engine = connect_mysql(load_config())

# 取出有完整数据的币种
fe = FeatureEngineer()
symbols = fe.get_symbols(("1h", "4h", "d1"))

bad_syms = []
for sym in symbols:
    df = pd.read_sql(
        "SELECT open_time FROM klines WHERE symbol=%s AND `interval`='1h' "
        "ORDER BY open_time",
        engine,
        params=(sym,),
        parse_dates=['open_time']
    )

    # 检查时间是否单调递增
    ordered = df.open_time.is_monotonic_increasing
    no_dup = not df.open_time.duplicated().any()
    gaps = df.open_time.diff().dropna()
    expected = pd.Timedelta(hours=1)
    if not ordered or not no_dup or (gaps != expected).any():
        bad_syms.append(sym)

print("存在问题的币种：", bad_syms)
