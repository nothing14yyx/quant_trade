from pathlib import Path
import os
import yaml
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
DEFAULT_CONFIG_PATH = BASE_DIR / "default_config.yaml"


def load_config(path: Path = CONFIG_PATH):
    """从 YAML 文件加载配置，若不存在则回退到默认配置"""
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file() and path == CONFIG_PATH:
        path = DEFAULT_CONFIG_PATH
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def connect_mysql(cfg):
    """根据配置创建并返回 SQLAlchemy engine"""
    mysql = cfg["mysql"]
    url = (
        f"mysql+pymysql://{mysql['user']}:{os.getenv('MYSQL_PASSWORD', mysql['password'])}"
        f"@{mysql['host']}:{mysql.get('port', 3306)}/{mysql['database']}?charset=utf8mb4"
    )
    return create_engine(url)


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "connect_mysql",
]
