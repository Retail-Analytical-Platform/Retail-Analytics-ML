import pytest

from src.config import CONFIG_FILES, load_config


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_config_loads(name):
    assert load_config(name)


def test_missing_key_raises(tmp_path, monkeypatch):
    (tmp_path / "forecast_config.yaml").write_text("horizon: 28\n")
    monkeypatch.setattr("src.config.CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="training_window_days"):
        load_config("forecast_config")
        
def test_wrong_type_raises(tmp_path, monkeypatch):
    (tmp_path / "forecast_config.yaml").write_text(
        "horizon: 'soon'\ntraining_window_days: 365\n"
    )
    monkeypatch.setattr("src.config.CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="horizon"):
<<<<<<< HEAD
        load_config("forecast_config")
        
def test_missing_grain_raises(tmp_path, monkeypatch):
    (tmp_path / "forecast_config.yaml").write_text(
        "training_window_days: 180\nhorizon: 28\n"
        "expected_date_start: '2024-01-01'\nexpected_date_end: '2025-12-31'\n"
        "min_fact_rows: 50000\ntarget_column: quantity\n"
    )
    monkeypatch.setattr("src.config.CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="grain"):
=======
>>>>>>> 0e6b1b4 (Add CI workflow, configuration files, and initial tests)
        load_config("forecast_config")