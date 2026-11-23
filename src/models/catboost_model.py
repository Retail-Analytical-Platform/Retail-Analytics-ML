from catboost import CatBoostRegressor

from src.models.base import BaseModel, register


@register("catboost")
class CatBoostModel(BaseModel):
    model_file = "model.cbm"

    def _fit(self, X, y, X_val, y_val):
        kwargs, eval_set = {}, None
        if X_val is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            eval_set = (X_val, y_val)
        self.model = CatBoostRegressor(
            **self.params, loss_function="RMSE", random_seed=self.seed, verbose=0, **kwargs
        )
        self.model.fit(X, y, cat_features=list(self.categorical), eval_set=eval_set)

    def predict(self, X):
        return self.model.predict(X)

    def _dump(self, path):
        self.model.save_model(str(path))

    def _load(self, path):
        self.model = CatBoostRegressor()
        self.model.load_model(str(path))