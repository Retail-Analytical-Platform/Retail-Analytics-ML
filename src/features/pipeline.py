from src.features.calendar import add_calendar_features
from src.features.feature_config import validate_feature_config
from src.features.lags import add_lag_rolling_features


def build_features(df, date_col, grain, target, feature_cfg, horizon):
    validate_feature_config(feature_cfg)
    out = add_calendar_features(df, date_col, feature_cfg["calendar"])
    return add_lag_rolling_features(
        out, date_col, grain, target, feature_cfg["lags_rolling"], horizon
    )