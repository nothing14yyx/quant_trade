from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ConfigManager:
    """加载并存储配置信息"""

    config_path: str | Path

    def load(self) -> dict:
        path = Path(self.config_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
