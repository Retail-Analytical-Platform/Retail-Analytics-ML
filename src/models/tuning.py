import numpy as np
import yaml

from src.config import CONFIG_DIR, load_config
from src.evaluation.compare import backtest_ml, get_folds, summarize
from src.features.pipeline import DATASET_PATH, FEATURES_PATH, load_frame
from src.models.training import model_params
from src.tracking import record_run


def sample_params(space, rng):
    params = {}
    for key, (kind, low, high) in space.items():
        if kind == "int":
            params[key] = int(rng.integers(low, high + 1))
        elif kind == "float":
            params[key] = float(rng.uniform(low, high))
        elif kind == "log_float":
            params[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            raise ValueError(f"Unknown search space kind '{kind}'")
    return params


def tune(name, run_id, fcfg, xcfg, ecfg, sample=False):
    import src.models.training as training

    mcfg = load_config("models_config")
    space, primary = mcfg["models"][name]["search_space"], ecfg["primary_metric"]
    rng = np.random.default_rng(mcfg["seed"])
    dataset, table = load_frame(DATASET_PATH), load_frame(FEATURES_PATH)
    folds = get_folds(dataset, fcfg)
    base = model_params(name, sample)

    best_score, best_params = float("inf"), None
    for trial in range(mcfg["n_trials"]):
        params = {**base, **sample_params(space, rng)}
        original = training.model_params
        training.model_params = lambda n, s=False, p=params: dict(p)
        try:
            preds, seconds = backtest_ml(name, table, folds, fcfg, xcfg, sample)
        finally:
            training.model_params = original
        summary = summarize(preds, ecfg["metrics"])
        summary["train_seconds"] = seconds
        record_run(f"{run_id}-trial{trial:02d}", name, params, summary, fcfg, [])
        if summary["mean"][primary] < best_score:
            best_score, best_params = summary["mean"][primary], params

    path = CONFIG_DIR / "tuned_params.yaml"
    current = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    if best_params is None:
        raise ValueError(f"No tuning trials completed for model '{name}'")
    current[name] = {k: best_params[k] for k in space}
    path.write_text(yaml.safe_dump(current, sort_keys=True), encoding="utf-8")
    return best_score, best_params