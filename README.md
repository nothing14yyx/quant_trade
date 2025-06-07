# Quant Trade 项目

该仓库包含一个量化交易的数据处理与信号生成框架，主要组件包括：

- **DataLoader**：从币安接口同步行情、资金费率以及情绪指数。
- **FeatureEngineer**：生成多周期特征并进行标准化处理。
- **ModelTrainer**：使用 LightGBM 训练多周期预测模型。
- **RobustSignalGenerator**：融合 AI 与多因子得分，生成交易信号。
- **Backtester**：依据生成的信号回测策略表现。
- **FeatureSelector**：根据实际周期下采样后的数据挑选最重要的特征。

`RobustSignalGenerator` 提供 `update_ic_scores(df)` 方法，可在启动回测或模拟时
传入近期历史数据，自动计算因子 IC 用于调整权重。

运行各组件前，请在 `utils/config.yaml` 中填写数据库与 API 配置。

## 安装与测试

在开始之前，请执行 `pip install -r requirements.txt` 安装依赖。

完成安装后，可运行 `pytest` 执行自带的单元测试。

从 v2.1 起，`feature_selector.py` 会在计算特征覆盖率和训练模型前，
按 1h、4h、1d 等周期对数据下采样，只评估对应时间点的特征表现。
同样地，`model_trainer.py` 在训练各周期模型时也会先过滤相应的
时间行，确保 4h 与 1d 模型仅使用自身周期的数据。
