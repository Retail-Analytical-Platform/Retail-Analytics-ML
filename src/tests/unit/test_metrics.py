import math

import pandas as pd
import pytest

from src.evaluation.metrics import (
    check_primary_metric,
    compute_metrics,
    get_metric,
    grouped_metrics,
    mae,
    smape,
    wape,
)

ACTUAL = [100, 200, 0, 50, 150]
FORECAST = [110, 180, 10, 50, 120]


def test_metrics_match_hand_calculation():
    m = compute_metrics(ACTUAL, FORECAST)
    assert m["mae"] == pytest.approx(14.0)
    assert m["rmse"] == pytest.approx(17.3205, abs=1e-3)   # sqrt(300)
    assert m["bias"] == pytest.approx(-6.0)
    assert m["wape"] == pytest.approx(0.14)                 # 70 / 500
    assert m["smape"] == pytest.approx(48.4545, abs=1e-3)


def test_zero_handling_never_raises_or_returns_infinity():
    assert smape([0, 0], [0, 0]) == 0.0
    assert smape([0, 0], [5, 5]) == pytest.approx(200.0)
    assert wape([0, 0], [0, 0]) == 0.0
    assert math.isnan(wape([0, 0], [5, 5]))


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        mae([1, float("nan")], [1, 2])
    with pytest.raises(ValueError):
        mae([], [])


def test_metric_lookup():
    assert get_metric("wape") is wape
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("mape")
    check_primary_metric("wape")
    with pytest.raises(ValueError):
        check_primary_metric("bias")


def test_grouped_metrics_per_fold():
    df = pd.DataFrame({
        "fold": [1, 1, 2, 2],
        "actual": [10, 20, 30, 40],
        "forecast": [12, 18, 30, 50],
    })
    out = grouped_metrics(df, ["fold"]).set_index("fold")
    assert out.loc[1, "mae"] == pytest.approx(2.0)
    assert out.loc[2, "mae"] == pytest.approx(5.0)