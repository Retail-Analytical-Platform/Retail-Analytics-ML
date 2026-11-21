import importlib
import json
from pathlib import Path

from src.models.base import METADATA_FILE, BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}
_LAZY = {
    "catboost": "src.models.catboost_model",
    "lightgbm": "src.models.lightgbm_model",
    "xgboost": "src.models.xgboost_model",
}


def register(name: str):
    def wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def available_models() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


def get_model(name: str, **params) -> BaseModel:
    if name not in _REGISTRY and name in _LAZY:
        importlib.import_module(_LAZY[name])
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'; expected one of {available_models()}") from None
    return cls(**params)


def build_model(name, cfg, **overrides):
    params = {**cfg["shared"], **cfg["models"][name], **overrides}
    return get_model(name, **params)


def read_metadata(directory) -> dict:
    return json.loads((Path(directory) / METADATA_FILE).read_text())


def load_model(directory) -> BaseModel:
    """Rebuild a saved model from its metadata, without knowing its class up front."""
    meta = read_metadata(directory)
    return get_model(meta["model"], **meta["params"]).load(directory)