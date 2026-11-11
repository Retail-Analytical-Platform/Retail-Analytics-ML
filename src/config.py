from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

_REQUIRED = {
    "forecast_config": {"training_window_days": int, "horizon": int},
    "features_config": {"feature_columns": list},
    "models_config": {"models": (list, dict)},
    "evaluation_config": {"metrics": list, "split_strategy": str},
    "params": {"target_column": str, "date_column": str},
}
CONFIG_FILES = sorted(_REQUIRED)


def load_config(name: str) -> dict:
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config '{name}' not found in {CONFIG_DIR}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config '{name}' must contain a YAML mapping at the top level.")

    for key, expected in _REQUIRED.get(name, {}).items():
        if key not in config:
            raise ValueError(f"Config '{name}' is missing required key: {key}")
        if not isinstance(config[key], expected):
            raise ValueError(f"Config '{name}': key '{key}' has the wrong type")

    return config