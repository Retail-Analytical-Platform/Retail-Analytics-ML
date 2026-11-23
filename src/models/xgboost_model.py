import xgboost as xgb

from src.models.base import BaseModel, register


@register("xgboost")
class XGBoostModel(BaseModel):
    model_file = "model.json"

    def _fit(self, X, y, X_val, y_val):
        kwargs, fit_kwargs = {}, {}
        if X_val is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        self.model = xgb.XGBRegressor(
            **self.params, tree_method="hist", enable_categorical=True, random_state=self.seed, **kwargs
        )
        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def _dump(self, path):
        self.model.save_model(str(path))

    def _load(self, path):
        self.model = xgb.XGBRegressor(enable_categorical=True)
        self.model.load_model(str(path))