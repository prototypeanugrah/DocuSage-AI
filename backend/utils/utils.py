from pathlib import Path

import yaml

from backend.config import Config


def load_config(config_path: str) -> Config:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load config file
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return Config(**config)
