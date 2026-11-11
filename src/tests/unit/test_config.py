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
        load_config("forecast_config")