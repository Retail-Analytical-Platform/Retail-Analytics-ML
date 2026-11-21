import json
from pathlib import Path

from src.models.base import METADATA_FILE, BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register(name):
    def wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return wrap

def available_models():
    return sorted(_REGISTRY)

def get_model(name, **params):
    try:
        return _REGISTRY[name](**params)
    except KeyError:
        raise ValueError(f"Unknown model '{name}'; expected one of {available_models()}") from None

def build_model(name, model_cfg):
    """model_cfg is the loaded models_config: shared settings first, then per-model ones."""
    params = {**model_cfg.get("shared", {}), **model_cfg["models"].get(name, {})}
    return get_model(name, **params)

def load_model(directory):
    """Rebuild a saved model from its metadata, without knowing its class up front."""
    meta = json.loads((Path(directory) / METADATA_FILE).read_text())
    return get_model(meta["model"], **meta["params"]).load(directory)