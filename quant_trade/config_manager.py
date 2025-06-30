import yaml
from pathlib import Path


class ConfigManager:
    """配置参数加载与管理"""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = self._load()

    def _load(self) -> dict:
        path = self.config_path
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get(self, key: str, default=None):
        return self.cfg.get(key, default)
