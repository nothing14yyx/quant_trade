from pathlib import Path
import os
import yaml
from sqlalchemy import create_engine

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH):
    """从 YAML 文件加载配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_mysql(cfg):
    """根据配置创建并返回 SQLAlchemy engine"""
    mysql = cfg["mysql"]
    url = (
        f"mysql+pymysql://{mysql['user']}:{os.getenv('MYSQL_PASSWORD', mysql['password'])}"
        f"@{mysql['host']}:{mysql.get('port', 3306)}/{mysql['database']}?charset=utf8mb4"
    )
    return create_engine(url)


__all__ = ["CONFIG_PATH", "load_config", "connect_mysql"]
