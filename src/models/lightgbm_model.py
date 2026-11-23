from typing import Any, cast

import lightgbm as lgb

from src.models.base import BaseModel, register


@register("lightgbm")
class LightGBMModel(BaseModel):
    model_file = "model.txt"

    def _fit(self, X, y, X_val, y_val):
        self.model = lgb.LGBMRegressor(**self.params, random_state=self.seed, verbose=-1)
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None:
            fit_kwargs["eval_X"] = X_val
            fit_kwargs["eval_y"] = y_val
            if self.early_stopping_rounds is not None:
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                ]
        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X) -> Any:
        return self.model.predict(X)

    def _dump(self, path):
        booster = cast(lgb.Booster, getattr(self.model, "booster_", self.model))
        booster.save_model(str(path))

    def _load(self, path):
        self.model = lgb.Booster(model_file=str(path))