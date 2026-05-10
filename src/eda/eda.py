"""
Exploratory Data Analysis (EDA) for the heart disease dataset.

**File:** ``src/eda/eda.py`` — executable CLI
(``python src/eda/eda.py [step]``).

**Role in the project.** This is the entry point for **data understanding**
and for producing **heart_disease_processed_dataset.csv**. Downstream
``model_training/train.py`` expects that file; run
``python src/eda/eda.py all`` (or at least the preprocess step) before
training.

Preprocessing primitives live in ``data_preprocessing/pre_processing_data.py``.
This module orchestrates them and adds summaries plus plots.

**What it does**

1. **Inspect** — Data-quality summary on the raw table (:func:`inspect_data`).
2. **EDA** — Figures under ``screenshots/`` (class balance, histograms,
   correlation heatmap, age by outcome) via :func:`perform_eda`.

The CLI :func:`main` wires preprocessing to those steps and writes
``data/heart_disease_processed_dataset.csv``.

**CLI (project root)**

::

    python src/eda/eda.py              # full pipeline (default)
    python src/eda/eda.py load         # step 1 — load raw CSV only
    python src/eda/eda.py inspect      # step 2 — load + data-quality summary
    python src/eda/eda.py preprocess   # step 3 — clean + save cleaned CSV
    python src/eda/eda.py eda          # step 4 — plots (after preprocess)
    python src/eda/eda.py all          # chained workflow
    python src/eda/eda.py --help

**Dataset.** ``data/heart_disease_UCI_dataset.csv`` (comma header,
space-separated rows).

**Attribution.** UCI Heart Disease —
https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Author
------
SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from __future__ import annotations

__author__ = "SANDIP BHATTACHARYYA"
__roll__ = "2025cs05025"

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Resolve packages ``config`` and ``data_preprocessing`` from project root
_EDA_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _EDA_DIR.parent  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config.paths import (  # noqa: E402
    DEFAULT_CLEAN_CSV,
    DEFAULT_CSV,
    SCREENSHOTS_DIR,
)
from data_preprocessing import (  # noqa: E402
    clean_data,
    load_data,
    save_cleaned_csv,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# INSPECT (EDA)
# ─────────────────────────────────────────────


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a concise data-quality summary to stdout.

    **When to use:** After ``load`` and before cleaning, to see missing
    symbols (``?``), odd dtypes, and basic statistics. Does not modify the
    dataframe.

    Shows the first rows, dtypes, ``describe`` for all columns, and counts of
    missing values or literal ``'?'`` tokens (common missing sentinel in this
    dataset).

    Parameters
    ----------
    df : pandas.DataFrame
        Typically the output of :func:`data_preprocessing.load_data` before
        cleaning.

    Returns
    -------
    None
        Side effect only (printing).
    """
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe(include="all"))
    print("\nMissing / '?' tokens per column:")
    for col in df.columns:
        if df[col].dtype == "object":
            n_bad = (df[col] == "?").sum() + df[col].isna().sum()
        else:
            n_bad = df[col].isna().sum()
        if n_bad > 0:
            print(f"   {col}: {int(n_bad)}")


# ─────────────────────────────────────────────
# EDA VISUALISATIONS
# ─────────────────────────────────────────────


def perform_eda(
    df: pd.DataFrame, output_dir: str | Path | None = None
) -> None:
    """
    Generate and save exploratory plots for the cleaned dataset.

    **When to use:** Only after :func:`data_preprocessing.clean_data` (numeric
    table). These charts support reports and slides; they are independent of
    model training.

    Produces: (1) class balance bar chart with counts and percentages,
    (2) histograms for key numeric features, (3) lower-triangle Pearson
    correlation heatmap over numeric columns including ``target``, and
    (4) overlapping age histograms by outcome. Uses seaborn/matplotlib styling
    suitable for reports.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned data from :func:`data_preprocessing.clean_data` (numeric
        features + ``target``).
    output_dir : str, pathlib.Path, or None
        Directory for PNG files. If ``None``, writes under ``screenshots/``
        at the project root.

    Returns
    -------
    None
        Figures are written to disk; paths are printed to stdout.

    Notes
    -----
    Output files: ``class_balance.png``, ``feature_histograms.png``,
    ``correlation_heatmap.png``, ``age_by_target.png``.
    """
    out = Path(output_dir) if output_dir is not None else SCREENSHOTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120

    numeric_for_hist = [
        c
        for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    # --- Class balance ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    counts = df["target"].value_counts().sort_index()
    labels = ["No disease (0)", "Heart disease (1)"]
    colors = ["#27ae60", "#c0392b"]
    bars = ax.bar(
        labels,
        [counts.get(0, 0), counts.get(1, 0)],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title("Class balance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    total = counts.sum()
    for bar, n in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        pct = 100.0 * n / total if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{n}\n({pct:.1f}%)",
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out / 'class_balance.png'}")

    # --- Feature histograms ---
    n_feat = len(numeric_for_hist)
    n_cols = 3
    n_rows = int(np.ceil(n_feat / n_cols)) if n_feat else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for i, col in enumerate(numeric_for_hist):
        axes[i].hist(
            df[col].dropna(),
            bins=22,
            color="#2980b9",
            edgecolor="white",
            linewidth=0.5,
        )
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
    for j in range(len(numeric_for_hist), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(
        "Feature distributions", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(out / "feature_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out / 'feature_histograms.png'}")

    # --- Correlation heatmap (numeric features + target) ---
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = num_df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.4,
            ax=ax,
            annot_kws={"size": 7},
            vmin=-1,
            vmax=1,
        )
        ax.set_title(
            "Feature correlation (lower triangle)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(
            out / "correlation_heatmap.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"   Saved: {out / 'correlation_heatmap.png'}")

    # --- Age by outcome ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in [
        (0, "#27ae60", "No disease"),
        (1, "#c0392b", "Heart disease"),
    ]:
        subset = df.loc[df["target"] == label, "age"]
        ax.hist(
            subset.dropna(),
            bins=18,
            alpha=0.65,
            color=color,
            label=name,
            edgecolor="white",
        )
    ax.set_title(
        "Age distribution by outcome", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(out / "age_by_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out / 'age_by_target.png'}")

    print(f"\nEDA plots saved under: {out.resolve()}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def run_step_load(raw_path: Path) -> pd.DataFrame:
    """Step 1 — load raw CSV into memory (no cleaning, no plots)."""
    print("=" * 55)
    print("   STEP 1 — LOAD")
    print("=" * 55)
    return load_data(raw_path)


def run_step_inspect(raw_path: Path) -> pd.DataFrame:
    """
    Step 2 — load raw data and print :func:`inspect_data` quality summary.
    """
    print("=" * 55)
    print("   STEP 2 — INSPECT")
    print("=" * 55)
    df_raw = load_data(raw_path)
    inspect_data(df_raw)
    return df_raw


def run_step_preprocess(raw_path: Path) -> pd.DataFrame:
    """
    Step 3 — load raw data, run :func:`clean_data`, write
    ``data/heart_disease_processed_dataset.csv``.

    Does **not** generate PNG plots; use ``eda`` step or ``all`` for charts.
    """
    print("=" * 55)
    print(
        "   STEP 3 — PREPROCESS "
        "(data_preprocessing/pre_processing_data.py)"
    )
    print("=" * 55)
    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)
    out = save_cleaned_csv(df_clean)
    print(f"\nCleaned dataset saved to {out}")
    return df_clean


def run_step_eda(
    clean_csv: Path,
    screenshots_dir: Path | None = None,
) -> None:
    """
    Step 4 — read an existing cleaned CSV from disk and run
    :func:`perform_eda`.

    Use when preprocessing already ran and you only want to refresh plots.
    """
    print("=" * 55)
    print("   STEP 4 — EDA (plots)")
    print("=" * 55)
    if not clean_csv.is_file():
        raise FileNotFoundError(
            f"Cleaned file not found: {clean_csv}\n"
            "Run first: python src/eda/eda.py preprocess"
        )
    df_clean = pd.read_csv(clean_csv)
    perform_eda(df_clean, output_dir=screenshots_dir)


def run_all(
    raw_path: Path,
    screenshots_dir: Path | None = None,
) -> None:
    """
    Run inspect, clean + save CSV, and EDA plots in one go.

    Equivalent to the CLI command ``python src/eda/eda.py all`` (default step).
    """
    print("=" * 55)
    print("   FULL PIPELINE — load, inspect, preprocess, EDA")
    print("=" * 55)

    df_raw = load_data(raw_path)
    inspect_data(df_raw)

    print("\n" + "=" * 55)
    print(
        "   Preprocessing (data_preprocessing/pre_processing_data.py)"
    )
    print("=" * 55)
    df_clean = clean_data(df_raw)

    print("\n" + "=" * 55)
    print("   EDA")
    print("=" * 55)
    perform_eda(df_clean, output_dir=screenshots_dir)

    clean_path = save_cleaned_csv(df_clean)
    print(f"\nCleaned dataset saved to {clean_path}")


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry: choose a single stage or the full ``all`` pipeline.

    **Steps explained for newcomers**

    - **load** — Read raw CSV only (sanity check file reads).
    - **inspect** — load + print table summary (:func:`inspect_data`).
    - **preprocess** — load, :func:`clean_data`, save
      ``heart_disease_processed_dataset.csv`` (required before training).
    - **eda** — Load an *existing* cleaned CSV and write PNGs to
      ``screenshots/`` (default).
    - **all** — inspect raw data, preprocess, run EDA, ensure cleaned CSV
      exists.

    Parameters
    ----------
    argv : list of str or None
        Arguments (excluding script name). ``None`` uses ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Heart disease pipeline: run load / inspect / preprocess / eda "
            "step-by-step, or all at once."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/eda/eda.py load
  python src/eda/eda.py inspect
  python src/eda/eda.py preprocess
  python src/eda/eda.py eda
  python src/eda/eda.py all
  python src/eda/eda.py eda \\
    --clean-csv data/heart_disease_processed_dataset.csv
""",
    )
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["all", "load", "inspect", "preprocess", "eda"],
        help="Pipeline step (default: all)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help=f"Raw CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=None,
        help=f"Cleaned CSV for 'eda' step (default: {DEFAULT_CLEAN_CSV})",
    )
    parser.add_argument(
        "--screenshots",
        type=Path,
        default=None,
        help=(
            "Directory for EDA PNGs (default: <project>/screenshots)"
        ),
    )
    args = parser.parse_args(argv)

    raw_path = args.raw if args.raw is not None else DEFAULT_CSV
    clean_csv = (
        args.clean_csv if args.clean_csv is not None else DEFAULT_CLEAN_CSV
    )
    shots = args.screenshots

    if args.step == "load":
        run_step_load(raw_path)
    elif args.step == "inspect":
        run_step_inspect(raw_path)
    elif args.step == "preprocess":
        run_step_preprocess(raw_path)
    elif args.step == "eda":
        run_step_eda(clean_csv, screenshots_dir=shots)
    else:
        run_all(raw_path, screenshots_dir=shots)


if __name__ == "__main__":
    main()
