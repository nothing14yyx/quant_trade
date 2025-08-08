from quant_trade.data_loader import DataLoader

# 1. 初始化 DataLoader（会读取 utils/config.yaml）
dl = DataLoader()

# 2. 获取要更新的币种列表（也可手动指定，如 ['BTCUSDT']）
symbols = dl.get_top_symbols()

# 3. 调用更新链上指标
dl.update_cm_metrics(symbols)
