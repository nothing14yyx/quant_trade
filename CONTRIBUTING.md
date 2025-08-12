# 贡献指南

欢迎参与本项目！提交代码前请遵循以下步骤：

## 基本流程
- 使用 `ruff` 做风格检查：

  ```bash
  make lint
  # 等价于 ruff check --fix . + mypy .
  ```

- 类型检查：

  ```bash
  mypy .
  ```

- 测试：

  ```bash
  pytest -q tests
  ```

  请确保所有测试通过，遵循根目录 `AGENTS.md` 的测试要求。

## 代码风格与提交
- 遵循 PEP8，使用四个空格缩进。
- 提交信息简洁，建议英文或简体中文，标题不超过 50 字。

## Makefile 目标
新贡献者可参考 Makefile 中的常用目标：

- `make lint`：运行 `ruff` 与 `mypy`。
- `make test`：先执行 `make lint`，再运行完整测试。
- `make test-quick`：运行部分快速测试。
