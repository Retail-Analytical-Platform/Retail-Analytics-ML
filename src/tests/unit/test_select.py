import pytest

from src.evaluation.select import choose


def rec(model, mean, std=0.01, seconds=1.0):
    return {"model": model, "metrics": {"mean": {"wape": mean}, "std": {"wape": std}, "train_seconds": seconds}}


def test_best_mean_wins_and_gate_passes():
    best, ranked, _, passed = choose([rec("catboost", 0.18), rec("lightgbm", 0.20), rec("seasonal_naive", 0.30)], "wape", 0.05)
    assert best["model"] == "catboost" and passed and len(ranked) == 2


def test_ties_break_on_spread_then_time():
    records = [rec("a", 0.2, 0.02, 1), rec("b", 0.2, 0.01, 9), rec("c", 0.2, 0.01, 2), rec("seasonal_naive", 0.5)]
    assert choose(records, "wape", 0.0)[0]["model"] == "c"


def test_gate_fails_when_margin_not_reached():
    *_, passed = choose([rec("catboost", 0.29), rec("seasonal_naive", 0.30)], "wape", 0.05)
    assert not passed


def test_missing_baseline_is_an_error():
    with pytest.raises(ValueError, match="Seasonal naive"):
        choose([rec("catboost", 0.2)], "wape", 0.05)