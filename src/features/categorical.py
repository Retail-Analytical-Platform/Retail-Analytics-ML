import pandas as pd

MODELS = ("catboost", "lightgbm", "xgboost")


def feature_columns(df, date_col, target):
    return [c for c in df.columns if c not in (date_col, target)]


def fit_categories(df, categorical_columns):
    """Category lists from the full dataset. Compute once and pass them to every
    split, so train, validation and forecast rows share the same category codes."""
    return {c: sorted(df[c].dropna().unique()) for c in categorical_columns}


def prepare_model_frame(df, model, categorical_columns, date_col, target, categories=None):
    """Return (X, params): the feature table in the form `model` expects, plus
    constructor parameters to pass to that model."""
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'; expected one of {MODELS}")

    cols = feature_columns(df, date_col, target)
    missing = [c for c in categorical_columns if c not in cols]
    if missing:
        raise ValueError(f"Categorical columns not in the data features: {missing}")

    X = df[cols].copy()

    if model == "catboost":
        for c in categorical_columns:
            X[c] = X[c].astype(str)
        return X, {"cat_features": list(categorical_columns)}

    categories = categories or fit_categories(df, categorical_columns)
    for c in categorical_columns:
        X[c] = X[c].astype(pd.CategoricalDtype(categories=categories[c]))
    return X, ({} if model == "lightgbm" else {"enable_categorical": True})