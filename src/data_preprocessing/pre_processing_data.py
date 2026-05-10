"""
Heart disease dataset loading and preprocessing.

**File:** ``src/data_preprocessing/pre_processing_data.py``

**Usage**

- Imported by ``eda/eda.py``, ``model_training/train.py`` (via cleaned CSV), and
  ``model_training/inference.py`` (raw → clean for inference).
- Do not run this file directly; call ``load_data``, ``clean_data``, or
  ``save_cleaned_csv`` from other modules or tests.

**Pipeline (conceptual).**

1. ``load_data`` — Read ``heart_disease_UCI_dataset.csv`` (comma header, irregular rows),
   normalize tokens, build a dataframe including ``target``.
2. ``clean_data`` — Map categories to codes, impute missing features (median/mode),
   binarise ``target`` (healthy vs disease), drop unused columns; all numeric features.
3. ``save_cleaned_csv`` — Persist ``heart_disease_processed_dataset.csv`` for EDA and modelling.

**Consumers.** ``eda/eda.py`` runs this flow end-to-end; ``model_training/train.py``
loads the cleaned CSV; ``model_training/inference.py`` calls ``load_data`` + ``clean_data``
on raw inputs so batch scoring matches training features.

**Outputs.** Files under ``data/``; plots are **not** produced here (see ``eda/eda.py``).

Author
------
SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from __future__ import annotations

__author__ = "SANDIP BHATTACHARYYA"
__roll__ = "2025cs05025"

from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import DEFAULT_CLEAN_CSV, DEFAULT_CSV, PROJECT_ROOT


def load_data(path: str | Path | None = None, *, verbose: bool = True) -> pd.DataFrame:
    """
    Load the heart disease dataset from the project CSV.

    **If you are new here:** This function only **reads** the file into a dataframe.
    It does **not** apply the cleaning rules (missing values, label encoding). Those
    happen in :func:`clean_data`. Training typically uses the cleaned CSV produced by
    EDA; prediction calls ``load_data`` then ``clean_data`` so live rows match training.

    Reads ``data/heart_disease_UCI_dataset.csv`` by default. The file uses a
    comma-separated header line and whitespace-separated body lines. Some rows
    include an extra marker token (e.g. ``buff`` / ``sick``) immediately before
    the class label; parsing keeps the first ``n_columns - 1`` field tokens and
    the final token as ``target``.

    Parameters
    ----------
    path : str, pathlib.Path, or None
        Path to the CSV file. If ``None``, uses ``DEFAULT_CSV`` under the
        project ``data/`` directory.
    verbose : bool
        If True, print load summary (set False for batch inference scripts).

    Returns
    -------
    pandas.DataFrame
        Raw table with string/object columns as read from the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    Notes
    -----
    Prints shape and column names to stdout for a quick sanity check.
    """
    path = Path(path) if path is not None else DEFAULT_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Expected `data/heart_disease_UCI_dataset.csv` in the project (or pass another path)."
        )

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    columns = [c.strip() for c in lines[0].split(",")]
    ncols = len(columns)
    records: list[list[str]] = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= ncols:
            row = parts[: ncols - 1] + [parts[-1]]
        else:
            row = parts + [""] * (ncols - len(parts))
        records.append(row)

    df = pd.DataFrame(records, columns=columns)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    if verbose:
        print("Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """
    Preprocess the raw heart disease table for modelling and EDA.

    Replaces ``'?'`` with NaN, maps string categories to integer codes, coerces
    numeric columns, imputes missing **feature** values (median for numeric,
    mode fallback otherwise), and never imputes the raw string target. The
    target is then binarised: ``H`` (healthy) → ``0``, any disease label
    (e.g. ``S1``–``S4``) → ``1``. Optional column ``num`` is dropped if present;
    any remaining non-numeric feature columns are removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe; must include a ``target`` column.
    verbose : bool
        If True, print imputation / cleaning steps.

    Returns
    -------
    pandas.DataFrame
        Fully numeric features (and integer ``target``), ready for correlation
        plots and sklearn-style pipelines.

    Raises
    ------
    ValueError
        If ``target`` is missing from ``df``.

    Notes
    -----
    Logs imputation choices and final class counts to stdout.
    """
    df = df.copy()
    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' in dataframe.")

    df = df.replace("?", np.nan)

    y_raw = df[target_col].astype(str).str.strip()

    df["sex"] = df["sex"].map({"male": 1, "fem": 0})
    df["fbs"] = df["fbs"].map({"true": 1, "fal": 0})
    df["exang"] = df["exang"].map({"true": 1, "fal": 0})

    cp_map = {"angina": 0, "abnang": 1, "notang": 2, "asympt": 3}
    df["cp"] = df["cp"].map(cp_map)

    restecg_map = {"norm": 0, "abn": 1, "hyp": 2}
    df["restecg"] = df["restecg"].map(restecg_map)

    slope_map = {"up": 0, "flat": 1, "down": 2}
    df["slope"] = df["slope"].map(slope_map)

    thal_map = {"norm": 0, "fix": 1, "rev": 2}
    df["thal"] = df["thal"].map(thal_map)

    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
        else:
            mode = df[col].mode()
            fill_val = mode.iloc[0] if len(mode) else 0
        df[col] = df[col].fillna(fill_val)
        if verbose:
            print(f"   Filled missing in '{col}' with {fill_val}")

    df[target_col] = y_raw.apply(lambda x: 0 if x.upper() == "H" else 1)

    if "num" in df.columns:
        df = df.drop(columns=["num"])

    junk = [
        c for c in df.columns if c != target_col and df[c].dtype == "object"
    ]
    if junk:
        df = df.drop(columns=junk)
        if verbose:
            print(f"   Dropped residual text columns: {junk}")

    if verbose:
        print(f"\nCleaning done. Shape: {df.shape}")
        print(f"   Target distribution:\n{df[target_col].value_counts()}")
    return df


def save_cleaned_csv(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    index: bool = False,
) -> Path:
    """
    Write a preprocessed dataframe to CSV.

    **Typical use:** After :func:`clean_data`, save to ``data/heart_disease_processed_dataset.csv`` so
    ``model_training/train.py`` can load a stable file without re-running EDA.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`clean_data`.
    path : str, pathlib.Path, or None
        Destination file. If ``None``, uses ``data/heart_disease_processed_dataset.csv`` under
        ``PROJECT_ROOT``.
    index : bool
        Passed through to ``DataFrame.to_csv``.

    Returns
    -------
    pathlib.Path
        Path written to.
    """
    out = Path(path) if path is not None else DEFAULT_CLEAN_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    return out


__all__ = [
    "DEFAULT_CLEAN_CSV",
    "DEFAULT_CSV",
    "PROJECT_ROOT",
    "clean_data",
    "load_data",
    "save_cleaned_csv",
]
