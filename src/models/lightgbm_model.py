from typing import Sequence, cast

import lightgbm as lgb
import numpy as np

from src.models.base import BaseModel
from src.models.registry import register


@register("lightgbm")
class LightGBMModel(BaseModel):
    def fit(self, X, y, X_valid=None, y_valid=None):
        params, seed, patience, _ = getattr(self, "_unpack")()
        use_valid = X_valid is not None and y_valid is not None
        model = lgb.LGBMRegressor(**params, random_state=seed, verbose=-1)
        callbacks = [lgb.early_stopping(patience, verbose=False)] if use_valid and patience else []
        if use_valid:
            assert X_valid is not None
            assert y_valid is not None
            model.fit(X, y, eval_X=X_valid, eval_y=y_valid, callbacks=callbacks)
        else:
            model.fit(X, y, callbacks=callbacks)
        self.booster = model.booster_
        self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        if hasattr(X, "columns"):
            X = X[self.feature_names]
        predictions = np.asarray(self.booster.predict(X), dtype=float)
        return cast(Sequence[float], np.clip(predictions, 0, None))

    def _write(self, path):
        self.booster.save_model(str(path))

    def _read(self, path):
        self.booster = lgb.Booster(model_file=str(path))