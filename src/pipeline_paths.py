import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("RETAIL_ARTIFACT_DIR", ROOT / "artifacts"))
DATA_DIR = ARTIFACT_DIR / "data"
FOLDS_PATH = ARTIFACT_DIR / "folds" / "folds.json"
METRICS_DIR = ARTIFACT_DIR / "metrics"
MODELS_DIR = ARTIFACT_DIR / "models"
SELECTED_DIR = MODELS_DIR / "selected"