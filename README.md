# Quant Trade 项目

该仓库包含一个量化交易的数据处理与信号生成框架，主要组件包括：

- **DataLoader**：从币安接口同步行情、资金费率及情绪指数，并可按日拉取 CoinGecko 的市值与板块数据。
- **FeatureEngineer**：生成多周期特征并进行标准化处理，新增影线比例、长期成交量突破等衍生指标，并提供跨周期的 RSI、MACD 背离特征。现已利用 CoinGecko 市值数据计算价格差、市值/成交量涨跌率等额外因子；同时加入 HV_7d/14d/30d、KC 宽度变化率、Ichimoku 基准线等新指标，并支持买卖比、资金流量比、成交量密度、价差百分比及 BTC/ETH 短期相关性。
  另外新增 `sma_5_*`、`sma_20_*` 均线及其交叉比值 `ma_ratio_5_20`，用于衡量短中期趋势变化。
-   `merge_features` 新增 `batch_size` 参数，可在内存有限时按币种分批写入：

-```python
-fe.merge_features(save_to_db=True, batch_size=1)
-```
- **ModelTrainer**：使用 LightGBM 训练多周期预测模型。
- **标签系统**：根据历史波动动态设定阈值，并额外提供未来波动率等辅助目标。
- **RobustSignalGenerator**：融合 AI 与多因子得分，生成交易信号。
-   新增 `ma_cross_logic`，会检查 `sma_5_1h` 与 `sma_20_1h` 的形态，
    在多空信号一致时适度放大得分，方向相反时则削弱或保持观望。
-   新增 `th_window` 与 `th_decay` 参数，用于控制动态阈值参考的历史得分
    数量及衰减程度，默认 `th_window=150`、`th_decay=1.0`，
    可根据策略需求适当调小窗口或衰减系数。
- **Backtester**：依据生成的信号回测策略表现。
- **FeatureSelector**：综合 AUC、SHAP 与 Permutation Importance 评分，筛选去冗余的核心特征。

`RobustSignalGenerator` 提供 `update_ic_scores(df, window=None, group_by=None)`
接口，可在启动回测或模拟时传入近期历史数据，按时间窗口或币种分组滚动计算
因子 IC，并据此自动更新权重。
`run_scheduler.py` 在启动和每天零点都会从 `features` 表读取最近 1000 条记录，
自动调用该接口更新因子权重，无需手动操作。

运行各组件前，请在 `utils/config.yaml` 中填写数据库与 API 配置，
其中 `api_key`、`api_secret`、`COINGECKO_API_KEY` 与 MySQL `password` 均支持通过环境变量传入。

默认情况下，项目会使用 CoinGecko 提供的公共 API，额度为每月 1 万次，
并限制每分钟最多 30 次调用。如有需要可在 `coingecko.api_key` 中配置你的公开密钥。

为减少搜索次数，`DataLoader` 会在初始化时读取 `coingecko_ids.json` 缓存，并在获得新的币种 id 后写回该文件。
`update_cg_market_data` 会先查询 `cg_market_data` 表，找出各币种的最后时间点，
若表为空则自动回补过去一年的记录，否则从最后时间的次日开始拉取缺失区间，
每天使用 `market_chart/range` 接口更新。
从 v2.5 起，`incremental_update_klines` 会在写入 K 线时同步并合并这些 CoinGecko 指标，
生成字段 `cg_price`、`cg_market_cap` 与 `cg_total_volume`，供后续特征工程使用。`FeatureEngineer` 会进一步基于这些列计算币安价格与 CoinGecko 价格差、市值和成交量的日涨跌率等因子。

## 安装与测试

在开始之前，请执行 `pip install -r requirements.txt` 安装依赖。

完成安装后，可运行 `pytest -q tests` 执行自带的单元测试。

```bash
pip install -r requirements.txt
pytest -q tests
```

内存不足或特征过多时，可以先在 `utils/config.yaml` 将 `feature_engineering.topn`
调小（如 20），或在执行 `feature_engineering.py` 时传入
`merge_features(topn=20)`。此外，`data_loader` 区段的 `start` 与 `end`
参数也可限定日期范围，以减少同步的历史数据行数。

## 数据库初始化

执行 `mysql < scripts/init_db.sql` 即可创建所需表格。
自 v2.4 起已移除 `depth_snapshot` 表，旧用户可直接删除该表后再运行脚本。

通过 `python param_search.py --rows 10000`(可选) 调整信号权重。
若希望同时优化 Δ-boost 参数，可加入 `--tune-delta`，例如：

```bash
python param_search.py --method optuna --tune-delta --trials 50
```

回测脚本 `backtester.py` 支持 `--recent-days N` 参数，可只回测最近 N 天的数据，例如：

```bash
python backtester.py --recent-days 7
```

回测中每笔交易的收益率计算为：

```
(exit_price - entry_price) * direction * position_size
----------------------------------------------------- - 2 * fee_rate
          entry_price * position_size
```

若 `position_size` 为 0，则该笔收益记为 0。

从 v2.1 起，`feature_selector.py` 会在计算特征覆盖率和训练模型前，
按 1h、4h、1d 等周期对数据下采样，只评估对应时间点的特征表现。
同样地，`model_trainer.py` 在训练各周期模型时也会先过滤相应的
时间行，确保 4h 与 1d 模型仅使用自身周期的数据。

自 v3.2 起，可在 `utils/config.yaml` 的 `train_settings` 下新增
`periods` 与 `tags` 两个列表，用于筛选想要训练的周期与标签，
例如：

```yaml
train_settings:
  periods: ["4h", "d1"]
  tags: ["up", "down"]
```
如果留空则会默认训练全部周期和标签。

自 v2.11 起，`feature_selector.py` 将相关性阈值从 0.95 下调至 0.90，并在此基
础上计算 VIF，若某列的 VIF 超过 10 会被迭代剔除。

## live_full_data 表

`signal_live_simulator.py` 会持续将实时信号写入 `live_full_data`。自 v2.2
起，该表新增 `indicators` 字段（建议 `TEXT` 或 `JSON` 类型），内容为
JSON 字符串，包含：

- `feat_1h`、`feat_4h`、`feat_d1`：标准化后的特征
- `raw_feat_1h`、`raw_feat_4h`、`raw_feat_d1`：原始特征
- `details`：信号生成时返回的调试信息

## cg_market_data 与 cg_global_metrics 表

这两个表分别由 `update_cg_market_data` 和 `update_cg_global_metrics` 写入
CoinGecko 的行情与全球指标数据，完整表结构见仓库根目录的 `schema.sql`。

自 v2.6 起，`cg_global_metrics` 额外保存 `eth_dominance`（ETH 市占率）字段。

自 v2.7 起，新增 `update_cg_coin_categories`，按日获取币种所属板块并写入
`cg_coin_categories` 表。方法会检查 `last_updated` 日期，仅在距离上次更新超过
一天时才重新调用 CoinGecko API，以避免过度请求。

自 v2.8 起，`update_cg_category_stats` 可一次性获取 `/coins/categories` 接口
返回的各板块市值、24 小时成交量等数据，并写入 `cg_category_stats` 表。


自 v2.10 起，`update_cg_global_metrics` 默认按日刷新（UTC 0 点），以减少 API 调用次数。


自 v2.9 起，`get_latest_cg_global_metrics` 将同时返回基于 `cg_category_stats`
 计算的热门板块信息，字段为 `hot_sector` 与 `hot_sector_strength`。




