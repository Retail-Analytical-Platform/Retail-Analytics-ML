import json
from pathlib import Path

import pytest

from src.models.base import BaseModel
from src.models.registry import (
    available_models,
    build_model,
    get_model,
    load_model,
    register,
)


@register("_dummy_mean")
class DummyMean(BaseModel):
    def fit(self, X, y, X_valid=None, y_valid=None):
        self.mean = float(sum(y)) / len(y)
        self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return [self.mean] * len(X)

    def _write(self, path):
        path.write_text(json.dumps({"mean": self.mean}))

    def _read(self, path):
        self.mean = json.loads(path.read_text())["mean"]


def _xy():
    import pandas as pd
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), [10.0, 20.0, 30.0]


def test_fit_save_load_predict_round_trip(tmp_path: Path):
    X, y = _xy()
    model = get_model("_dummy_mean", seed=1).fit(X, y)
    model.save(tmp_path / "m", run_id="run-1")
    loaded = load_model(tmp_path / "m")
    assert loaded.predict(X) == model.predict(X) == [20.0, 20.0, 20.0]
    assert loaded.feature_names == ["a", "b"]
    assert loaded.params == {"seed": 1}


def test_metadata_has_run_id_features_and_version(tmp_path: Path):
    X, y = _xy()
    get_model("_dummy_mean").fit(X, y).save(tmp_path, run_id="run-9")
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["run_id"] == "run-9"
    assert meta["features"] == ["a", "b"]
    assert meta["model"] == "_dummy_mean"
    assert meta["schema_version"] == 1


def test_unknown_and_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nope")
    with pytest.raises(ValueError, match="already registered"):
        register("_dummy_mean")(DummyMean)
    assert "_dummy_mean" in available_models()


def test_build_model_merges_shared_and_specific_params():
    cfg = {"shared": {"seed": 42}, "models": {"_dummy_mean": {"depth": 3}}}
    assert build_model("_dummy_mean", cfg).params == {"seed": 42, "depth": 3}