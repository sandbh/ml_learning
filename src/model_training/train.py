"""
Train, compare, and persist heart-disease classifiers with MLflow tracking.

**File:** ``src/model_training/train.py`` — run as a script (not imported for side effects).

**Usage (repo root)**

.. code-block:: bash

    python src/eda/eda.py preprocess   # ensure data/heart_disease_processed_dataset.csv exists
    python src/model_training/train.py
    python src/model_training/train.py --experiment my_run

**Scope (assignment Tasks 2–4).** Loads ``data/heart_disease_processed_dataset.csv``, builds two sklearn
``Pipeline`` objects (Logistic Regression with scaling; Random Forest without scaling),
tunes hyperparameters with stratified ``GridSearchCV`` (ROC-AUC), reports hold-out test
metrics, logs params/metrics/artifacts to MLflow (experiment ``heart_disease_prediction``
by default, store ``./mlruns``), and saves the **best** model by test ROC-AUC into
``models/``.

**Encoding.** Categorical handling is done upstream in
``data_preprocessing/pre_processing_data.py``; this file only adds imputation/scaling
appropriate to each estimator family.

**Artifacts written**

- ``models/best_model.pkl`` — Winning fitted pipeline.
- ``models/feature_names.pkl`` — Training column order for ``inference.py``.
- ``models/training_metadata.json`` — Metrics, versions, paths, timestamps.
- ``screenshots/`` — Confusion matrix and ROC images per model.
- ``mlruns/`` — MLflow runs (models, plots; UI via ``mlflow ui --backend-store-uri ./mlruns``).

**Typical run**::

    python src/model_training/train.py
    python src/model_training/train.py --experiment my_experiment

Author: SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _TRAIN_DIR.parent  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config.paths import CLEAN_DATA_CSV, MLRUNS_DIR, MODELS_DIR, SCREENSHOTS_DIR

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn


def _patch_mlflow_sklearn_pyfunc_if_lzma_missing() -> None:
    """
    MLflow sklearn.save_model calls pyfunc.add_to_model; loading mlflow.pyfunc imports lzma.
    Some Python builds (e.g. pyenv without xz) omit _lzma, so log_model fails and the UI
    shows broken model artifacts or an upload-failed state in the UI. Stubbing add_to_model skips the pyfunc
    flavor only; sklearn flavor and MLmodel still log correctly.
    """
    try:
        import _lzma  # noqa: F401
    except ModuleNotFoundError:

        class _PyfuncStub:
            @staticmethod
            def add_to_model(*_args, **_kwargs) -> None:
                return None

        mlflow.sklearn.pyfunc = _PyfuncStub()


_patch_mlflow_sklearn_pyfunc_if_lzma_missing()

warnings.filterwarnings("ignore", category=FutureWarning)

# Inner CV folds for GridSearchCV / cross_validate (stratified). Test set fraction is
# controlled in prepare_features (default 20% hold-out), separate from these folds.
CV_SPLITS = 5
RANDOM_STATE = 42


def get_parallel_n_jobs() -> int:
    """
    Workers for ``GridSearchCV``, ``cross_validate``, and tree ``n_jobs``.

    ``MLOPS_N_JOBS`` (integer) overrides detection — use ``1`` in CI sandboxes or hosts
    where joblib/loky fails with ``PermissionError`` on startup.

    Otherwise, if ``os.sysconf`` is unavailable or forbidden (common in hardened
    sandboxes), returns ``1`` so training still completes; on normal Unix/macOS returns
    ``-1`` (all CPUs).
    """
    raw = os.environ.get("MLOPS_N_JOBS", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    if hasattr(os, "sysconf"):
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except (PermissionError, OSError, ValueError):
            return 1
    return -1


def load_clean_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the cleaned CSV written by EDA (numeric features + binary ``target``).

    **Why this exists:** Training assumes preprocessing was done once and saved to disk,
    so experiments stay reproducible and fast. If this file is missing, run
    ``python src/eda/eda.py preprocess`` (or ``all``) first.

    Parameters
    ----------
    path
        Defaults to :obj:`config.paths.CLEAN_DATA_CSV` (``data/heart_disease_processed_dataset.csv``).

    Returns
    -------
    pandas.DataFrame
        Ready for ``prepare_features`` (all feature columns numeric).
    """
    p = Path(path) if path is not None else CLEAN_DATA_CSV
    if not p.is_file():
        raise FileNotFoundError(
            f"Clean data not found: {p}\n"
            "Run: python src/eda/eda.py preprocess"
        )
    df = pd.read_csv(p)
    print(f"Loaded clean data: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split features ``X`` and label ``y``, with stratified train/test split.

    **Why stratify:** Keeps similar class balance in train and test when computing
    final metrics. The **test** portion is never used during ``GridSearchCV``; it only
    measures how the tuned model generalises.

    All columns except ``target`` are numeric (already encoded upstream in preprocessing).

    Parameters
    ----------
    df
        Clean dataframe including ``target`` (0 = no disease, 1 = disease).
    test_size
        Fraction of rows for the hold-out test set (default 20%).
    """
    if "target" not in df.columns:
        raise ValueError("Expected column 'target'.")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_logistic_pipeline() -> Pipeline:
    """
    Linear model pipeline: median imputation (defensive) + standardization + LR.

    Scaling puts features on comparable scales for regularized logistic regression.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """
    Tree model pipeline: imputation only (no StandardScaler).

    Random forests are scale-invariant; scaling is omitted by design.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=get_parallel_n_jobs(),
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _binary_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, and ROC-AUC for binary classification."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def run_cross_validation_report(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = CV_SPLITS,
) -> dict[str, float]:
    """
    Run stratified K-fold cross-validation on the **training** portion only.

    **Purpose:** Summarise how stable metrics are across folds (mean ± std). This is
    separate from the single hold-out **test** set used in :func:`evaluate_on_test`.
    Metrics are logged to MLflow as ``cv_*`` keys.
    """
    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = ["accuracy", "precision", "recall", "roc_auc", "f1"]
    out = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=get_parallel_n_jobs(),
        return_train_score=False,
    )
    summary = {}
    for name in scoring:
        key = f"test_{name}"
        scores = out[key]
        summary[f"cv_{name}_mean"] = float(np.mean(scores))
        summary[f"cv_{name}_std"] = float(np.std(scores))
    print(
        f"   CV (mean ± std): "
        f"ROC-AUC={summary['cv_roc_auc_mean']:.4f}±{summary['cv_roc_auc_std']:.4f}, "
        f"accuracy={summary['cv_accuracy_mean']:.4f}±{summary['cv_accuracy_std']:.4f}"
    )
    return summary


def evaluate_on_test(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
) -> dict[str, float]:
    """
    Evaluate the fitted pipeline on the **held-out test set** and save diagnostic plots.

    Returns a dict of metrics (same keys as :func:`_binary_metrics`). Prints sklearn's
    classification report and writes confusion-matrix and ROC PNGs under ``output_dir``.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _binary_metrics(y_test, y_pred, y_prob)

    print(f"\n{model_name} — hold-out test metrics:")
    for k, v in metrics.items():
        print(f"   {k:12s}: {v:.4f}")
    print(
        "\n"
        + classification_report(
            y_test,
            y_pred,
            target_names=["No disease (0)", "Heart disease (1)"],
            zero_division=0,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace(" ", "_").lower()
    _plot_confusion_matrix(y_test, y_pred, model_name, output_dir / f"cm_{safe}.png")
    _plot_roc_curve(
        y_test, y_prob, model_name, output_dir / f"roc_{safe}.png", metrics["roc_auc"]
    )
    return metrics


def _plot_confusion_matrix(y_test, y_pred, model_name: str, path: Path) -> None:
    """Save a seaborn heatmap of true vs predicted labels."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax.set_title(f"Confusion matrix — {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")


def _plot_roc_curve(
    y_test, y_prob, model_name: str, path: Path, auc_score: float
) -> None:
    """Save ROC curve (TPR vs FPR) with diagonal baseline for comparison."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")


def tune_train_log_mlflow(
    pipeline: Pipeline,
    param_grid: dict,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_name: str,
    base_params_for_log: dict,
) -> tuple[Pipeline, dict[str, float], dict[str, float]]:
    """
    Tune ``pipeline`` over ``param_grid``, log everything to MLflow, return winners.

    **What happens inside**

    1. ``GridSearchCV`` picks hyperparameters by **ROC-AUC** on inner CV folds.
    2. Best pipeline is refit on all of ``X_train``.
    3. Cross-validation summary (:func:`run_cross_validation_report`) and test metrics
       (:func:`evaluate_on_test`) are logged as MLflow metrics.
    4. Confusion/ROC images and the sklearn model artifact are attached to the run.

    Parameters
    ----------
    pipeline
        Unfitted sklearn ``Pipeline`` (see ``build_*_pipeline`` helpers).
    param_grid
        Keys must match pipeline step names (e.g. ``lr__C``).
    model_name
        Human-readable run name in MLflow UI.
    X_train, y_train, X_test, y_test
        Splits from :func:`prepare_features`.
    experiment_name
        MLflow experiment id string.
    base_params_for_log
        Extra static params recorded alongside best-grid params.

    Returns
    -------
    tuple
        ``(best_estimator, test_metrics_dict, cv_summary_dict)`` — test_metrics are used
        later by :func:`save_best_model` to choose between Logistic Regression and RF.
    """
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(experiment_name)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=get_parallel_n_jobs(),
        refit=True,
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_family", model_name.replace(" ", "_").lower())
        mlflow.set_tag("assignment", "MLOps_heart_disease")
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_splits", CV_SPLITS)

        print(f"\nTuning {model_name} (GridSearchCV, {CV_SPLITS}-fold stratified, scoring=roc_auc)...")
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        print(f"   Best params: {grid.best_params_}")
        print(f"   Best CV ROC-AUC (GridSearch): {grid.best_score_:.4f}")

        best_param_log = {
            f"best__{k}": ("null" if v is None else v) for k, v in grid.best_params_.items()
        }
        mlflow.log_params({**base_params_for_log, **best_param_log})
        mlflow.log_metric("gridsearch_best_cv_roc_auc", float(grid.best_score_))

        cv_summary = run_cross_validation_report(best, X_train, y_train)
        for k, v in cv_summary.items():
            mlflow.log_metric(k, v)

        test_metrics = evaluate_on_test(
            best, X_test, y_test, model_name, SCREENSHOTS_DIR
        )
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        safe = model_name.replace(" ", "_").lower()
        for rel in [SCREENSHOTS_DIR / f"cm_{safe}.png", SCREENSHOTS_DIR / f"roc_{safe}.png"]:
            if rel.is_file():
                mlflow.log_artifact(str(rel))

        try:
            mlflow.sklearn.log_model(best, artifact_path="model")
        except Exception as exc:
            # Some Python builds lack lzma; MLflow pyfunc path may fail. Fallback: pickle artifact.
            warnings.warn(f"mlflow.sklearn.log_model failed ({exc!r}); logging joblib pickle instead.")
            tmpd = tempfile.mkdtemp()
            try:
                pkl = Path(tmpd) / "sklearn_pipeline.pkl"
                joblib.dump(best, pkl)
                mlflow.log_artifact(str(pkl), artifact_path="model_pickled")
            finally:
                shutil.rmtree(tmpd, ignore_errors=True)

        print(f"   MLflow run_id: {mlflow.active_run().info.run_id}")

    return best, test_metrics, cv_summary


def save_best_model(
    results: list[tuple[Pipeline, dict[str, float], str]],
    X_train: pd.DataFrame,
) -> tuple[Pipeline, str]:
    """
    Pick the winning estimator by **test ROC-AUC** and write deployment-ready artifacts.

    **Selection rule:** Compare ``roc_auc`` in each candidate's test metrics dict; the
    highest wins (tie-breaking is arbitrary but rare). Saves:

    - ``models/best_model.pkl`` — joblib-serialised winning ``Pipeline``.
    - ``models/feature_names.pkl`` — column order of ``X_train`` (must match prediction).
    - ``models/training_metadata.json`` — JSON with metrics, library versions, paths.

    ``inference.py`` loads the pickle and feature list by default.
    """
    best = max(results, key=lambda x: x[1]["roc_auc"])
    best_model, best_metrics, best_name = best[0], best[1], best[2]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(
        f"\nBest model by test ROC-AUC: {best_name} (roc_auc={best_metrics['roc_auc']:.4f})"
    )
    print(f"Saved: {model_path}")

    feat_path = MODELS_DIR / "feature_names.pkl"
    feature_list = X_train.columns.tolist()
    joblib.dump(feature_list, feat_path)
    print(f"Saved: {feat_path}")

    meta = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model_name": best_name,
        "test_metrics": best_metrics,
        "all_runs_test_metrics": {name: m for _, m, name in results},
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "feature_names": feature_list,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "clean_training_csv": str(CLEAN_DATA_CSV),
        "artifacts": {
            "model_pickle": str(model_path),
            "feature_names_pickle": str(feat_path),
        },
    }
    meta_path = MODELS_DIR / "training_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {meta_path}")

    return best_model, best_name


def main(argv: list[str] | None = None) -> None:
    """
    Entry point: train both models, log to MLflow, save best pipeline under ``models/``.

    Expects cleaned training data from EDA. Does not start the MLflow UI server.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train Logistic Regression and Random Forest pipelines, compare test ROC-AUC, "
            "log runs to MLflow, and save the best estimator to models/best_model.pkl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python src/model_training/train.py
  python src/model_training/train.py --experiment my_run

Requires cleaned CSV (default: {CLEAN_DATA_CSV}). Produce it with:
  python src/eda/eda.py preprocess
""",
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=None,
        help=(
            "Path to heart_disease_processed_dataset.csv — numeric features + target column "
            f"(default: {CLEAN_DATA_CSV})"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="heart_disease_prediction",
        help="MLflow experiment name (creates experiment if missing)",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Training: feature pipelines, tuning, CV, metrics, MLflow")
    print("=" * 60)
    print(
        "\nModel selection notes:\n"
        "  • Features are already label-encoded in preprocessing; this step adds\n"
        "    median imputation (defensive) + StandardScaler for Logistic Regression.\n"
        "  • Random Forest uses imputation only (trees do not require scaling).\n"
        "  • Hyperparameters are chosen via GridSearchCV maximising ROC-AUC on\n"
        "    stratified inner CV; reported metrics include hold-out test performance.\n"
    )

    df = load_clean_data(args.clean_csv)
    X_train, X_test, y_train, y_test = prepare_features(df)

    results: list[tuple[Pipeline, dict[str, float], str]] = []

    # --- Logistic Regression ---
    lr_pipe = build_logistic_pipeline()
    lr_grid = {
        "lr__C": [0.01, 0.1, 1.0, 10.0],
    }
    lr_base = {"family": "logistic_regression", "scaler": "StandardScaler"}
    lr_model, lr_test, _ = tune_train_log_mlflow(
        lr_pipe,
        lr_grid,
        "Logistic Regression",
        X_train,
        X_test,
        y_train,
        y_test,
        args.experiment,
        lr_base,
    )
    results.append((lr_model, lr_test, "Logistic Regression"))

    # --- Random Forest ---
    rf_pipe = build_random_forest_pipeline()
    rf_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 6, 12],
        "rf__min_samples_leaf": [1, 2],
    }
    rf_base = {"family": "random_forest", "scaler": "none"}
    rf_model, rf_test, _ = tune_train_log_mlflow(
        rf_pipe,
        rf_grid,
        "Random Forest",
        X_train,
        X_test,
        y_train,
        y_test,
        args.experiment,
        rf_base,
    )
    results.append((rf_model, rf_test, "Random Forest"))

    save_best_model(results, X_train)

    print("\nDone. MLflow UI: mlflow ui --backend-store-uri ./mlruns (from project root)")


if __name__ == "__main__":
    main()
