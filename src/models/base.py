import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing_extensions import Self

METADATA_FILE = "metadata.json"
MODEL_FILE = "model.bin"
SCHEMA_VERSION = 1


class BaseModel(ABC):
    name = "base"   # set by @register

    def __init__(self, **params):
        self.params = params
        self.feature_names = None

    @abstractmethod
    def fit(self, X: Any, y: Any, X_valid: Any = None, y_valid: Any = None) -> Self:
        """Train. X_valid/y_valid is a time-ordered set for early stopping. Return self."""

    @abstractmethod
    def predict(self, X: Any) -> Sequence[float]:
        """Return one value per row of X."""

    @abstractmethod
    def _write(self, path: Path) -> None:
        """Write the fitted model to `path`."""

    @abstractmethod
    def _read(self, path: Path) -> None:
        """Restore the fitted model from `path`."""

    def save(self, directory, run_id, feature_names=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        features = list(feature_names if feature_names is not None else self.feature_names or [])
        self._write(directory / MODEL_FILE)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "model": self.name,
            "run_id": run_id,
            "params": self.params,
            "features": features,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (directory / METADATA_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return directory

    def load(self, directory) -> Self:
        directory = Path(directory)
        meta = json.loads((directory / METADATA_FILE).read_text())
        self._read(directory / MODEL_FILE)
        self.feature_names = meta["features"]
        return self