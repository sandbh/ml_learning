"""
Batch inference: raw heart-disease CSV → shared preprocessing → trained
pipeline.

**Purpose.** Score new rows with the **same** cleaning path as training (via
``data_preprocessing.load_data`` / ``clean_data``), then apply
``models/best_model.pkl``.
Demonstrates reproducible packaging: no silent feature drift between train
and inference.

**Inputs**

- Raw CSV (default ``data/heart_disease_UCI_dataset.csv`` or ``--raw``).
- ``--model`` — pickled sklearn ``Pipeline`` (default
  ``models/best_model.pkl``).
- ``--feature-names`` — column order saved at train time (default
  ``models/feature_names.pkl``).

**Outputs**

- Optional CSV with predictions and probabilities (``--output``).
- Prints a short preview to stdout.

**Typical run (repo root)**::

    python src/model_training/inference.py --output data/batch_predictions.csv

Train first so ``models/`` exists.

Author: SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_INF_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _INF_DIR.parent  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config.paths import MODELS_DIR, RAW_DATA_CSV  # noqa: E402
from data_preprocessing import clean_data, load_data  # noqa: E402


def run_inference(
    raw_csv: Path,
    model_path: Path,
    feature_names_path: Path,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Load raw data, apply preprocessing, run the trained sklearn Pipeline.

    **Flow**

    1. Load pickled ``Pipeline`` and list of feature names from training.
    2. Parse ``raw_csv`` with :func:`data_preprocessing.load_data` and clean
       with :func:`data_preprocessing.clean_data` (same rules as for
       ``heart_disease_processed_dataset.csv``).
    3. Align columns to ``feature_names`` (order matters for the fitted model).
    4. Output predicted class and probability of the positive class.

    If the cleaned frame still contains ``target`` (typical when scoring the
    training CSV), the returned table includes ``actual_target`` so you can
    compare accuracy; for real unlabeled rows you would drop ``target``
    upstream or ignore that column.

    Parameters
    ----------
    raw_csv
        Path to raw UCI-style heart data (same schema as
        ``data/heart_disease_UCI_dataset.csv``).
    model_path, feature_names_path
        Outputs from training (defaults under ``models/``).
    output_csv
        If set, write predictions to this path as CSV.

    Returns
    -------
    pandas.DataFrame
        At minimum ``predicted_class`` and ``proba_positive_class``; optionally
        ``actual_target`` when labels exist after cleaning.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Train first; missing model: {model_path}")
    if not feature_names_path.is_file():
        raise FileNotFoundError(f"Missing: {feature_names_path}")

    model = joblib.load(model_path)
    feature_names: list[str] = joblib.load(feature_names_path)

    df_raw = load_data(raw_csv, verbose=False)
    df_clean = clean_data(df_raw, verbose=False)

    if "target" in df_clean.columns:
        y_true = df_clean["target"].values
        X = df_clean.drop(columns=["target"])
    else:
        y_true = None
        X = df_clean

    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Cleaned data missing columns: {missing}")

    X = X[feature_names]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame(
        {
            "predicted_class": y_pred,
            "proba_positive_class": np.round(y_proba, 6),
        }
    )
    if y_true is not None:
        out["actual_target"] = y_true

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        print(f"Predictions written to: {output_csv}")

    return out


def main(argv: list[str] | None = None) -> None:
    """
    CLI wrapper around :func:`run_inference`.

    Run ``python src/model_training/train.py`` before this so
    ``models/best_model.pkl`` exists.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score rows from a RAW heart-disease CSV using the saved "
            "training pipeline. Preprocessing matches training (load_data + "
            "clean_data)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/model_training/inference.py --output data/batch_predictions.csv
  python src/model_training/inference.py \\
    --raw data/heart_disease_UCI_dataset.csv \\
    --model models/best_model.pkl
""",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_DATA_CSV,
        help=f"Raw dataset CSV path (default: {RAW_DATA_CSV})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "best_model.pkl",
        help=(
            "Joblib-serialised sklearn Pipeline from training (default: "
            "models/best_model.pkl)"
        ),
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=MODELS_DIR / "feature_names.pkl",
        help=(
            "Pickled list of column names in the order used when training "
            "(default: models/feature_names.pkl)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "If set, save predictions to this CSV path; always prints preview "
            "to stdout"
        ),
    )
    args = parser.parse_args(argv)

    print("Running: raw → preprocess → model.predict")
    out = run_inference(
        args.raw,
        args.model,
        args.feature_names,
        output_csv=args.output,
    )
    if "actual_target" in out.columns:
        acc = (out["predicted_class"] == out["actual_target"]).mean()
        print(f"Hold-out sanity (same file has labels): accuracy = {acc:.4f}")
    print(out.head(10).to_string(index=False))
    print(f"\nTotal rows: {len(out)}")


if __name__ == "__main__":
    main()
