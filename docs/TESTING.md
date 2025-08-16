# 测试指南

## 黄金测试

本项目提供了黄金测试用于验证信号生成逻辑的稳定性。
`tests/signal/test_golden_single.py` 与 `tests/signal/test_golden_batch.py`
会读取 `tests/fixtures/` 目录下的 `golden*.json` 文件，
并对比其中的 `signal`、`score`、`position_size` 等字段，
确保生成结果与基准一致。

## 新增基准

如需添加新的黄金基准，请在 `tests/fixtures/` 内创建
`golden_xxx.json` 文件。文件需包含以下字段：

- `features`：各周期的特征数据
- `raw`：对应的原始特征
- `expected`：包含期望的 `signal`、`score`、`position_size` 等信息

## 执行测试

- **仅运行黄金测试**：执行 `make test-quick`。
- **运行完整测试**：执行 `make test` 或 `pytest -q tests`，
  在提交前确保本地全部通过。CI 也将运行完整测试套件。

## 性能测试

可运行 `python scripts/benchmark_ai_batch.py` 评估批量推理性能。
脚本会显示 CPU 核心数（即 `batch.max_workers` 默认值）及推理耗时。

