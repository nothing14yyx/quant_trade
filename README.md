# Quant Trade 项目

该仓库包含一个量化交易的数据处理与信号生成框架，主要组件包括：

- **DataLoader**：从币安接口同步行情、资金费率及情绪指数，并可按日拉取 CoinGecko 的市值数据。
- **FeatureEngineer**：生成多周期特征并进行标准化处理，新增影线比例、长期成交量突破等衍生指标，并提供跨周期的 RSI、MACD 背离特征。
- **ModelTrainer**：使用 LightGBM 训练多周期预测模型。
- **标签系统**：根据历史波动动态设定阈值，并额外提供未来波动率等辅助目标。
- **RobustSignalGenerator**：融合 AI 与多因子得分，生成交易信号。
- **Backtester**：依据生成的信号回测策略表现。
- **FeatureSelector**：综合 AUC、SHAP 与 Permutation Importance 评分，筛选去冗余的核心特征。

`RobustSignalGenerator` 提供 `update_ic_scores(df, window=None, group_by=None)`
接口，可在启动回测或模拟时传入近期历史数据，按时间窗口或币种分组滚动计算
因子 IC，并据此自动更新权重。

运行各组件前，请在 `utils/config.yaml` 中填写数据库与 API 配置，
其中 `api_key`、`api_secret`、`COINGECKO_API_KEY` 与 MySQL `password` 均支持通过环境变量传入。

默认情况下，项目会使用 CoinGecko 提供的公共 API，额度为每月 1 万次，
并限制每分钟最多 30 次调用。如有需要可在 `coingecko.api_key` 中配置你的公开密钥。

## 安装与测试

在开始之前，请执行 `pip install -r requirements.txt` 安装依赖。

完成安装后，可运行 `pytest -q tests` 执行自带的单元测试。

通过 `python param_search.py --rows 10000`(可选) 调整信号权重。

回测脚本 `backtester.py` 支持 `--recent-days N` 参数，可只回测最近 N 天的数据，例如：

```bash
python backtester.py --recent-days 7
```

从 v2.1 起，`feature_selector.py` 会在计算特征覆盖率和训练模型前，
按 1h、4h、1d 等周期对数据下采样，只评估对应时间点的特征表现。
同样地，`model_trainer.py` 在训练各周期模型时也会先过滤相应的
时间行，确保 4h 与 1d 模型仅使用自身周期的数据。

## live_full_data 表

`signal_live_simulator.py` 会持续将实时信号写入 `live_full_data`。自 v2.2
起，该表新增 `indicators` 字段（建议 `TEXT` 或 `JSON` 类型），内容为
JSON 字符串，包含：

- `feat_1h`、`feat_4h`、`feat_d1`：标准化后的特征
- `raw_feat_1h`、`raw_feat_4h`、`raw_feat_d1`：原始特征
- `details`：信号生成时返回的调试信息




