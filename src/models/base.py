import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

FORMAT_VERSION = 1
_REGISTRY = {}
_LAZY = {
    "catboost": "src.models.catboost_model",
    "lightgbm": "src.models.lightgbm_model",
    "xgboost": "src.models.xgboost_model",
}


def register(name):
    def wrap(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def get_model_class(name):
    if name not in _REGISTRY and name in _LAZY:
        importlib.import_module(_LAZY[name])
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(set(_REGISTRY) | set(_LAZY))}")
    return _REGISTRY[name]


def get_model(name, **kwargs):
    return get_model_class(name)(**kwargs)


def load_model(directory):
    meta = json.loads((Path(directory) / "meta.json").read_text(encoding="utf-8"))
    model = get_model(
        meta["model"], params=meta["params"], seed=meta["seed"], categorical=meta["categorical"],
    )
    model.feature_columns = meta["feature_columns"]
    model._load(Path(directory) / model.model_file)
    return model


class BaseModel(ABC):
    name = "base"
    model_file = "model.bin"

    def __init__(self, params=None, seed=42, early_stopping_rounds=None, categorical=()):
        self.params = dict(params or {})
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical = list(categorical)
        self.feature_columns = None
        self.model = None

    @abstractmethod
    def _fit(self, X, y, X_val, y_val):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def _dump(self, path):
        ...

    @abstractmethod
    def _load(self, path):
        ...

    def fit(self, X, y, X_val=None, y_val=None):
        self.feature_columns = list(X.columns)
        self._fit(X, y, X_val, y_val)
        return self

    def save(self, directory, run_id):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._dump(directory / self.model_file)
        meta = {
            "model": self.name, "run_id": run_id, "format_version": FORMAT_VERSION,
            "feature_columns": self.feature_columns, "params": self.params,
            "seed": self.seed, "categorical": self.categorical,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return directory