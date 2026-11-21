import numpy as np
from catboost import CatBoostRegressor

from src.models.base import BaseModel
from src.models.registry import register


@register("catboost")
class CatBoostModel(BaseModel):
    def fit(self, X, y, X_valid=None, y_valid=None):
        params, seed, patience, cats = self._unpack()
        if not cats:
            cats = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
        use_valid = X_valid is not None and y_valid is not None
        self.model = CatBoostRegressor(
            **params,
            random_seed=seed,
            verbose=0,
            cat_features=cats or None,            # native categorical handling
            early_stopping_rounds=patience if use_valid else None,
        )
        self.model.fit(X, y, eval_set=(X_valid, y_valid) if use_valid else None)
        self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X[self.feature_names]), 0, None)  # demand is never negative

    def _write(self, path):
        self.model.save_model(str(path), format="cbm")

    def _read(self, path):
        self.model = CatBoostRegressor()
        self.model.load_model(str(path), format="cbm")