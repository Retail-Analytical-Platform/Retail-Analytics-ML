import numpy as np
import xgboost as xgb

from src.models.base import BaseModel
from src.models.registry import register


@register("xgboost")
class XGBoostModel(BaseModel):
    def fit(self, X, y, X_valid=None, y_valid=None):
        params, seed, patience, _ = self._unpack()
        params.pop("early_stopping_rounds", None)
        params.pop("seed", None)
        params.pop("categorical_columns", None)
        use_valid = X_valid is not None and y_valid is not None
        self.model = xgb.XGBRegressor(
            **params,
            random_state=seed,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=patience if use_valid else None,
        )
        eval_set = [(X_valid, y_valid)] if use_valid else None
        self.model.fit(X, y, eval_set=eval_set)
        self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X[self.feature_names]), 0, None)

    def _write(self, path):
        self.model.save_model(str(path.with_suffix(".json")))   # explicit format

    def _read(self, path):
        self.model = xgb.XGBRegressor(enable_categorical=True)
        self.model.load_model(str(path.with_suffix(".json")))