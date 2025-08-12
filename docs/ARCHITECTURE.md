# 系统架构概览

本文件描述量化交易系统各核心组件、数据流与缓存策略。本次重构遵循“**无行为变更**”合同：只整理结构与文档，不影响任何现有输入输出。

## 组件与数据流

```
+-----------------------+
|  data_loader          |
|  coinmetrics_loader   |
+-----------+-----------+
            |
            v
+-----------+-----------+
| feature_engineering   |
| feature_processor     |
+-----------+-----------+
            |
            v
+-----------+-----------+
| model_trainer         |
| ai_model_predictor    |
+-----------+-----------+
            |
            v
+-----------+-----------+
| RobustSignalGenerator |
|  _ai_score_cache      |
|  _factor_cache        |
+-----+-------------+---+
      |             |
      |     start_weight_update_thread
      |             |
      |     stop_weight_update_thread
      v             v
+-----+-------------+---+
|   risk_manager        |
|   backtester          |
+-----------------------+
```

## 核心组件职责与不变量

### data_loader / coinmetrics_loader
- **职责**：实时抓取行情与链上数据。
- **不变量**：时间戳单调递增，缺失数据自动补齐，行情与链上数据对齐。

### feature_engineering / feature_processor
- **职责**：生成、清洗并标准化特征。
- **不变量**：输出中无 `NaN`；同一输入产生确定性特征；标准化范围稳定。

### model_trainer / ai_model_predictor
- **职责**：训练模型并进行推理。
- **不变量**：训练参数可复现；推理使用最新特征与模型快照。

### robust_signal_generator
- **职责**：融合因子、AI 得分与风险控制，产生交易信号。
- **不变量**：
  - 缓存命中时返回与重新计算一致的结果；
  - 权重更新线程始终安全启动与停止，不会泄露资源；
  - 信号计算对外接口保持幂等。

### risk_manager / backtester
- **职责**：
  - `risk_manager` 负责仓位、风险敞口与风控规则；
  - `backtester` 在历史数据上复现完整交易流程。
- **不变量**：风险约束始终满足；回测逻辑与实时逻辑一致。

## 缓存策略与线程安全
- **缓存类型**：`_ai_score_cache` 与 `_factor_cache` 等使用 LRU 策略，默认最大条目数 300。
- **命中条件**：以 `(symbol, timestamp, features)` 作为键，命中则直接返回缓存结果。
- **淘汰策略**：超过上限时淘汰最久未使用的数据。
- **线程安全**：
  - 缓存读写在内部锁保护下执行，防止并发污染；
  - `start_weight_update_thread` 启动后台线程定期更新因子权重，`stop_weight_update_thread` 确保线程安全退出；
  - 多线程环境下，所有共享状态必须通过原子操作或锁维护一致性。

## 无行为变更
本次重构仅调整架构描述与文档，不改动已有接口、输入输出或业务逻辑。

