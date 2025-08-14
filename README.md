# Quant Trade 项目

该仓库包含一个量化交易的数据处理与信号生成框架。

更多核心参数的默认值与调参建议，请参阅 [docs/params.md](docs/params.md)。

## 示例使用

训练完成后会在 `models/` 目录生成 `report.json`，汇总交叉验证区间、embargo、模型参数与评估指标。
可运行示例脚本查看推理与简单回测流程：

```bash
python backtests/demo_backtest.py
```

主要组件包括：

- **DataLoader**：从币安接口同步行情、资金费率及情绪指数，并可按日拉取 CoinGecko 的市值与板块数据。
- **CoinMetricsLoader**：批量获取更多链上指标。
  新增 `community_metrics(asset)` 用于查询社区版可用指标，
  `update_cm_metrics(community_only=True)` 可仅抓取这些指标。
- **Market Phase Detection**：根据活跃地址与市值判断牛市、熊市或震荡阶段，可在 `utils/config.yaml` 的 `market_phase.symbols` 列表（或单个 `symbol`）指定参与判断的交易对，默认仅使用 BTCUSDT。若填写多个币种，函数会分别返回各链的得分 `S` 与阶段，并按市值加权给出整体结果，例如：

```python
{
    "BTCUSDT": {"phase": "bull", "S": 1.8},
    "ETHUSDT": {"phase": "bear", "S": -0.9},
    "TOTAL": {"phase": "range", "S": 0.3}
}
```
`RobustSignalGenerator.update_market_phase()` 会读取 `market_phase.phase_th_mult` 与 `phase_dir_mult` 调整阈值和多空倾向，例如：

```yaml
market_phase:
  phase_th_mult:
    bull: 0.8
    bear: 1.2
    range: 1.0
  phase_dir_mult:
    bull:
      long: 1.2
      short: 0.8
    bear:
      long: 0.8
      short: 1.2
    range:
      long: 1.0
      short: 1.0
```
- **FeatureEngineer**：生成多周期特征并进行标准化处理，新增影线比例、长期成交量突破等衍生指标，并提供跨周期的 RSI、MACD 背离特征。现已利用 CoinGecko 市值数据计算价格差、市值/成交量涨跌率等额外因子；同时加入 HV_7d/14d/30d、KC 宽度变化率、Ichimoku 基准线、VWAP、随机指标等新指标，并支持买卖比、资金流量比、成交量密度、价差百分比及 BTC/ETH 短期相关性。
  另外新增 `sma_5_*`、`sma_20_*` 均线及其交叉比值 `ma_ratio_5_20`，用于衡量短中期趋势变化。
-   `merge_features` 新增 `batch_size` 参数，可在内存有限时按币种分批写入：

-```python
-fe.merge_features(save_to_db=True, batch_size=1)
-```
-   现在还可通过 `use_polars=True` 启用 [Polars](https://pola.rs/) 加速批量拼接，
    与 `n_jobs` 结合能更好利用多核性能。
    并行模式会占用较多 CPU/内存，请在资源允许的环境下使用。
-   新增 `period_cfg.d1.smooth_window` 参数，可滚动平滑
    `future_max_drawdown_d1`，默认窗口为 3。
- **ModelTrainer**：使用 LightGBM 训练多周期预测模型。
- **标签系统**：根据历史波动动态设定阈值，并额外提供未来波动率等辅助目标。
- **RobustSignalGenerator**：融合 AI 与多因子得分，生成交易信号。
-   新增 `ma_cross_logic`，会检查 `sma_5_1h` 与 `sma_20_1h` 的形态，
    在多空信号一致时适度放大得分，方向相反时则削弱或保持观望。
-   新增 `th_window` 与 `th_decay` 参数，用于控制动态阈值参考的历史得分
    数量及衰减程度，默认 `th_window=150`、`th_decay=1.0`，
    可根据策略需求适当调小窗口或衰减系数。
-   `signal_threshold.quantile` 指定历史得分分位数，默认 `0.78`，数值越高代表触发门槛越严格。
-   `signal_threshold.window` 与 `signal_threshold.dynamic_quantile`
    控制动态阈值计算所用的窗口与分位数，可针对不同资产或市场阶段自定义。
-   `compute_dynamic_threshold` 会依据最近 `history_scores` 计算分位数，并结合 `atr_4h`、`adx_4h`、`atr_d1`、`adx_d1` 与 `pred_vol`、`vix_proxy` 等指标自适应调整门槛，`regime` 与 `reversal` 还能微调阈值和 `rev_boost`，参数统一封装在 `DynamicThresholdInput` 中。
-   阈值相关配置已整合为 `SignalThresholdParams`，方便统一管理。
-   新增 `dynamic_threshold` 配置项，可自定义 ATR、ADX 与 funding 对阈值的影响系数及上限。
-   新增 `smooth_window`、`smooth_alpha` 与 `smooth_limit` 参数，用于平滑最近得分，减少噪声影响。
-   `vote_system.prob_th` 为基础概率阈值（默认 0.5），`prob_margin` 用于弱票判定（如 0.08 表示 `0.5±0.08`），`strong_prob_th` 与 `_compute_vote` 结合使用以判定强票。
-   新增 `risk_budget_threshold` 函数，可依据历史波动率或换手率分布计算风险阈值：

```python
from quant_trade.robust_signal_generator import risk_budget_threshold
vol_hist = [0.01, 0.02, 0.05, 0.03]
th = risk_budget_threshold(vol_hist, quantile=0.9)
-   早期兼容函数 `robust_signal_generator()` 已移除，请直接调用
    `RobustSignalGenerator.generate_signal()`。
-   因子评分新增对 Ichimoku 云层厚度、VWAP 偏离率及跨周期 RSI 差值的考量，
    帮助更准确地衡量趋势和动量强度。
    其中 `rsi_1h_mul_vol_ma_ratio_4h` 仅归入 `volume` 因子，不再在 `momentum` 中重复计分。
- **Backtester**：依据生成的信号回测策略表现。
- **FeatureSelector**：综合 AUC、SHAP 与 Permutation Importance 评分，筛选去冗余的核心特征。生成的 YAML 文件现统一存放在 `quant_trade/selected_features/` 目录。
  可通过 `feature_selector.rows` 设置读取样本的最大行数，以避免内存不足。

`RobustSignalGenerator` 提供 `update_ic_scores(df, window=None, group_by=None)`
接口，可在启动回测或模拟时传入近期历史数据，按时间窗口或币种分组滚动计算
因子 IC，并据此自动更新权重。
`run_scheduler.py` 在启动并根据 `ic_update_interval_hours` 配置定期
从 `features` 表读取最近 `ic_update_rows` 条记录，自动调用该接口更新因子权重。
默认间隔为 24 小时，仍会在午夜执行，与旧版保持兼容。可根据数据量和市场波动
适当调整更新频率。
若训练样本分布不均，可在 `config.yaml` 中通过 `ic_scores` 字段手动设置各周期权重，
例如:

```yaml
ic_scores:
  1h: 1.0
  4h: 0.2
  d1: 0.1
这样 `RobustSignalGenerator` 在融合多周期分数时将更侧重 1h 模型。

运行各组件前，请在 `utils/config.yaml` 中填写数据库与 API 配置，
其中 `api_key`、`api_secret`、`COINGECKO_API_KEY` 与 MySQL `password` 均可通过环境变量设置。例如：


### CoinMetrics 社区 API 限制

CoinMetrics 提供的社区版接口无需 API key，但只能访问部分公开链上指标，且
对同一 IP 每 6 秒最多接受 10 次请求。项目中 `CoinMetricsLoader` 默认
`rate_limit=10`、`period=6.0`，以免触发限速。若想调整抓取指标，请在
`utils/config.yaml` 的 `coinmetrics.metrics` 中填写 `community_metrics()`
返回的名称，避免使用未开放的字段。

```python
from quant_trade.utils import community_metrics

print(community_metrics()[:5])

该函数会实时查询官方目录，方便检查当前可用指标。



为减少搜索次数，`DataLoader` 会在初始化时读取 `coingecko_ids.json` 缓存，并在获得新的币种 id 后写回该文件。
`update_cg_market_data` 会先查询 `cg_market_data` 表，找出各币种的最后时间点，
若表为空则自动回补过去一年的记录，否则从最后时间的次日开始拉取缺失区间，
每天使用 `market_chart/range` 接口更新。
从 v2.5 起，`incremental_update_klines` 会在写入 K 线时同步并合并这些 CoinGecko 指标，
生成字段 `cg_price`、`cg_market_cap` 与 `cg_total_volume`，供后续特征工程使用。`FeatureEngineer` 会进一步基于这些列计算币安价格与 CoinGecko 价格差、市值和成交量的日涨跌率等因子。

## 安装与测试

在开始之前，请执行 `pip install -r requirements.txt` 安装依赖。

若需在加载配置时启用自动校验，可额外安装可选依赖 `pydantic`：

```bash
pip install pydantic
```

未安装时程序会跳过校验，继续使用原始配置。

完成安装后，可运行 `pytest -q tests` 执行自带的单元测试。

```bash
pip install -r requirements.txt
pytest -q tests

无论从哪个目录运行，`RobustSignalGenerator` 都会自动解析相对模型路径，无需手动调整工作目录。

## 信号生成流程

```mermaid
flowchart TD
    A[准备特征] --> B[阶段一: 计算多周期得分]
    B --> C[阶段二: 风险与拥挤度检查]
    C --> D[阶段三: 计算仓位与止盈止损]

内存不足或特征过多时，可以先在 `utils/config.yaml` 将 `feature_engineering.topn`
调小（如 20），或在执行 `feature_engineering.py` 时传入
`merge_features(topn=20)`。此外，`data_loader` 区段的 `start` 与 `end`
参数也可限定日期范围，以减少同步的历史数据行数。

若只需处理近期样本，可在 `feature_selector` 区段设置 `rows` 或 `start_time`
限制加载的数据量，例如：

```yaml
feature_selector:
  rows: 200000        # 仅读取最近 20 万行，可根据实际内存调整
  # 或
  # start_time: "2024-01-01"
```

## 风险参数调整

默认 `risk_adjust.factor` 为 0.15，`risk_adjust_threshold` 默认为 `null`，会根据历史得分自动计算阈值。若发现信号过少，可在 `utils/config.yaml` 放宽以下参数：

```yaml
risk_adjust:
  factor: 0.15        # 风险惩罚系数，值越低得分扣减越轻
risk_adjust_threshold: null     # 若为 null，将根据历史得分计算阈值
risk_th_quantile: 0.6           # 自适应阈值的分位数
veto_conflict_count: 2         # funding 冲突达到此数目直接放弃信号
min_trend_align: 2             # 趋势方向至少在 N 个周期保持一致
protection_limits:
  risk_score: 1.0    # 允许的风险得分上限
crowding_limit: 1.05     # 允许的拥挤度上限
risk_scale: 1.0         # risk_score 每增加 1，仓位乘以 e^{-risk_scale}
risk_filters_enabled: true
max_stop_loss_pct: 0.05     # 单笔最大止损比例
trailing_stop_pct: 0.03     # 移动止损触发比例
risk_budget_per_trade: 0.01 # 每笔占用的风险预算
crowding_protection:
  enabled: true
  same_side_limit: 5
  cool_down_minutes: 45

- `max_stop_loss_pct` 控制单笔交易最大的允许亏损比例。
- `trailing_stop_pct` 在获利回撤超过该比例时触发移动止损。
- `risk_budget_per_trade` 定义每笔交易可占用的风险预算上限。
- `crowding_protection` 用于监控市场同向拥挤度并在过热时暂停开仓。
- `risk_adjust.factor` 控制风险值对 `fused_score` 的削减力度，公式为
  `fused_score *= 1 - factor * risk_score`，一般建议取值在 `0.1`～`0.3` 之间。

- 风险过滤开启时，若计算出的仓位低于动态下限，将保留最小仓位并记为 `min_pos`，不再直接归零。

自 v2.6 起，`risk_score` 仅在計算倉位時生效，不再在得分階段二次扣減。
自 v2.7 起，引入 `RiskManager.calc_risk`，根据环境得分、预测波动率与 OI 变化率
综合计算风险值，`apply_risk_filters` 会直接调用该方法并受 `protection_limits.risk_score`
约束。

修改后重启调度器即可生效。

若希望暂时停用风险与拥挤度过滤，可在 `utils/config.yaml` 顶层设置：

```yaml
risk_filters_enabled: false
dynamic_threshold_enabled: true
direction_filters_enabled: true
filter_penalty_mode: true     # true 时仅惩罚得分，不直接弃用
penalty_factor: 0.5           # 惩罚系数，越低扣减越多
```
默认启用 `filter_penalty_mode`，资金费率冲突或风险值超限时不会直接丢弃信号，而是按 `penalty_factor` 缩减得分和仓位。
将其设为 `false` 后，`apply_risk_filters` 会直接返回得分，`compute_position_size` 也不会再根据风险值提高仓位下限。默认情况下仍会执行 `compute_dynamic_threshold` 更新 `base_th`；若希望保持固定阈值，可同时将 `dynamic_threshold_enabled` 设为 `false`。
`direction_filters_enabled` 则控制是否执行方向与仓位过滤，设为 `false` 时 `_determine_direction` 和 `_apply_position_filters` 将直接返回原始结果。

### 启用或禁用 AI 模型

`RobustSignalGenerator` 默认会根据配置加载 AI 模型，可在 `utils/config.yaml` 的
`enable_ai` 字段中控制，示例：

```yaml
enable_ai: true  # 设为 false 时跳过模型加载
```

也可通过环境变量 `ENABLE_AI=0` 临时关闭。

### 因子贡献度分解开关

`RobustSignalGenerator` 会在生成信号时计算各因子对最终得分的贡献度，可在
`utils/config.yaml` 中配置：

```yaml
enable_factor_breakdown: true  # 设为 false 可提升回测速度
```

### 动态阈值调节
`compute_dynamic_threshold` 会根据近期得分历史计算出门槛基准。其中
`th_window` 决定统计多少条 `history_scores`，窗口越短反应越灵敏；
`th_decay` 为衰减系数，设定后越新的分数权重越高；
`signal_threshold.quantile` 指定所取的分位数，数值越低则门槛越低，
更易触发信号。

`signal_threshold.window` 与 `signal_threshold.dynamic_quantile`
进一步允许针对不同资产或阶段调整窗口与分位数。

示例，若在高频或小仓位策略中需要更积极的入场，可在 `utils/config.yaml`
中调整（`signal_threshold` 位于配置根目录）：

```yaml
th_window: 80
th_decay: 0.5
signal_threshold:
  quantile: 0.65
  window: 80
  dynamic_quantile: 0.7
这会使阈值更快反应最新波动，从而在行情活跃时给出更多交易机会。
`rev_boost` 则决定在检测到潜在反转时额外加成的得分，数值越大越易触发交易。
`dynamic_threshold` 区块则控制 ATR、ADX 与 funding 对阈值的加成比例和上限，
如需放宽限制可在其中调整:

```yaml
dynamic_threshold:
  atr_mult: 3.0
  atr_cap: 0.15

`dynamic_threshold_enabled` 控制上述逻辑是否生效，默认值为 `true`。

### AI 评分后无信号解决方案

若在启用 AI 得分后发现始终没有交易信号，可能是阈值过高或风险过滤被关闭。
可在 `utils/config.yaml` 调低 `signal_threshold.base_th`，或保持风险过滤开启：

```yaml
signal_threshold:
  base_th: 0.06
risk_filters_enabled: true
```
这样能降低触发门槛，并利用风险过滤动态调整得分。

## 信号生成重构说明

信号生成流程重构为以下顺序：

features → AI → 因子 → (融合) → 阈值 → (风控倍率) → 两阶段 sizing

首先由特征模块提取 `features`，随后 AI 模型给出预测得分；这些得分与传统因子评分融合后，与设定的阈值比较并乘以风控倍率，最终通过两阶段 sizing 计算仓位。

更多调参请参见[参数表](docs/params.md)。

## 数据库初始化

执行 `mysql < scripts/init_db.sql` 即可创建所需表格。
自 v2.4 起已移除 `depth_snapshot` 表，旧用户可直接删除该表后再运行脚本。
自 v2.8 起 `merge_features(save_to_db=True)` 写入时将保持 `features` 表结构，
如需在首次运行前创建或补充字段，可执行
`mysql < scripts/migrate_add_feature_columns.sql`。

通过 `python -m quant_trade.param_search --rows 10000` (可选) 调整信号权重。
参数搜索默认按时间拆分为训练集和验证集，可通过 `--test-ratio` 调整验证集比例。
脚本默认使用 Optuna 搜索，并同时优化 Δ-boost 参数，例如：

```bash
python -m quant_trade.param_search --trials 50
```
以上示例需在项目根目录执行，或确保该目录已加入 `PYTHONPATH`。

若脚本抛出 `ValueError("no trades found during parameter search")`，请检查
`features` 表是否含有数据，并确认 `generate_signal` 是否能正常返回信号。

回测脚本 `backtester.py` 支持 `--recent-days N` 参数，可只回测最近 N 天的数据，例如：

```bash
python -m quant_trade.backtester --recent-days 7
```
同样地，其它脚本也推荐通过 `python -m quant_trade.<name>` 方式运行。
## 生成离线价位表与缩放参数

同步行情数据后，可运行：

```bash
python -m quant_trade.offline_price_table

脚本会在 `data/offline_prices/` 下为每个币种导出多周期价位表，同时生成 `price_scaler.json`，后续标准化特征时直接加载即可。

回测中每笔交易的收益率计算为：

(exit_price - entry_price) * direction * position_size
----------------------------------------------------- - 2 * fee_rate
          entry_price * position_size

若 `position_size` 为 0，则该笔收益记为 0。

从 v2.1 起，`feature_selector.py` 会在计算特征覆盖率和训练模型前，
按 1h、4h、d1 等周期对数据下采样，只评估对应时间点的特征表现。
同样地，`model_trainer.py` 在训练各周期模型时也会先过滤相应的
时间行，确保 4h 与 d1 模型仅使用自身周期的数据。

自 v3.2 起，可在 `utils/config.yaml` 的 `train_settings` 下新增
`periods` 与 `tags` 两个列表，用于筛选想要训练的周期与标签，
例如：

```yaml
train_settings:
  periods: ["4h", "d1"]
  tags: ["up", "down"]
如果留空则会默认训练全部周期和标签。

自 v3.3 起，可在 `train_settings` 中设置 `min_samples`（默认为 2000），
`model_trainer.py` 会按特征重要性依次加入特征，并在每次尝试后
统计当前特征组合 `dropna()` 后的有效样本量，若低于该阈值则停止
新增特征并给出提示。

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




