from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_DIR / "data")).resolve()
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", PROJECT_DIR / "artifacts")).resolve()
RAW_DIR = ARTIFACT_DIR / "raw"

TRAIN_FILE = os.environ.get("TRAIN_FILE", "Data_ISPU_2022_2024.csv")
TEST_FILE = os.environ.get("TEST_FILE", "aqms_full_jan_mar_2026.csv")

# Column names after the "landing".
RAW_COLUMNS = ["date", "pm10", "pm25", "so2", "o3", "no2", "co"]

# Mapping from the original CSV headers (Indonesian) to the standard names.
SOURCE_COLUMN_MAP = {
    "Waktu": "date",
    "PM10": "pm10",
    "PM2.5": "pm25",
    "SO2": "so2",
    "O3": "o3",
    "NO2": "no2",
    "CO": "co",
}

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _resolve_data_file(filename: str) -> Path:
    """Search several candidate locations so the script is easy to run."""
    candidates = [
        DATA_DIR / filename,
        PROJECT_DIR / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        SRC_DIR / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Source file '{filename}' was not found.\n"
        f"Searched in:\n  {searched}\n"
        f"Set the DATA_DIR or TRAIN_FILE/TEST_FILE env vars to point at the data directory."
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def read_source_csv(path: str | Path) -> pd.DataFrame:
    """Read one source CSV without transforming its contents.

    All values are left as-is (strings, '-', decimal comma).
    Only `Waktu` is renamed to `date`; the remaining columns are selected.
    """
    path = _resolve_data_file(str(path)) if not Path(str(path)).is_absolute() else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path)
    df = _select_and_rename(df, path)
    return df


def _select_and_rename(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Select the 7 relevant columns and normalise their names to snake_case."""
    source = f" in {path}" if path else ""

    missing = [col for col in SOURCE_COLUMN_MAP if col not in df.columns]
    if missing:
        raise KeyError(
            f"Source columns not found{source}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[list(SOURCE_COLUMN_MAP.keys())].copy()
    out.columns = [SOURCE_COLUMN_MAP[col] for col in out.columns]
    return out[RAW_COLUMNS].reset_index(drop=True)


def extract_raw(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Main extract step: read the training + testing CSVs into the raw zone.

    Returns:
        {"train": DataFrame, "test": DataFrame} with snake_case columns and a
        RangeIndex (not yet cleaned).
    """
    train_path = _resolve_data_file(TRAIN_FILE)
    test_path = _resolve_data_file(TEST_FILE)

    if verbose:
        print("=" * 78)
        print("EXTRACT — READING SOURCE DATA")
        print("=" * 78)
        print(f"Training : {train_path}")
        print(f"Testing  : {test_path}")

    df_train_raw = read_source_csv(train_path)
    df_test_raw = read_source_csv(test_path)

    if verbose:
        print(f"\n  train shape : {df_train_raw.shape}")
        print(f"  test shape  : {df_test_raw.shape}")
        print("\nTraining data preview (first 5 rows, values as-is):")
        print(df_train_raw.head().to_string())

    return {"train": df_train_raw, "test": df_test_raw}


def land_raw(
    data: dict[str, pd.DataFrame] | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Write the extract result to the raw zone (artifacts/raw/*.csv)."""
    data = data if data is not None else extract_raw(verbose=verbose)
    _ensure_dir(RAW_DIR)

    written: dict[str, Path] = {}
    for split, df in data.items():
        out_path = RAW_DIR / f"{split}_raw.csv"
        df.to_csv(out_path, index=False)
        written[split] = out_path

    if verbose:
        print("\nLanding zone:")
        for split, out_path in written.items():
            size_kb = out_path.stat().st_size / 1024
            print(f"  {split:<6} -> {out_path} ({size_kb:,.1f} KB)")

    return written


def load_extracted(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Read the data back from the raw zone (used by transform.py).

    Falls back to the source CSVs if the raw zone has not been created yet.
    """
    paths = {"train": RAW_DIR / "train_raw.csv", "test": RAW_DIR / "test_raw.csv"}

    if all(p.is_file() for p in paths.values()):
        if verbose:
            print("Reading data from the raw zone (artifacts/raw/).")
        return {split: pd.read_csv(p) for split, p in paths.items()}

    if verbose:
        print("Raw zone does not exist yet — reading directly from the source CSVs.")
    return extract_raw(verbose=verbose)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> dict[str, Path]:
    data = extract_raw()
    return land_raw(data)


if __name__ == "__main__":
    main()
