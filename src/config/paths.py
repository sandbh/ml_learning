"""
Central path layout for reproducibility (single source of truth).

**File:** ``src/config/paths.py`` — import path constants only; no side effects.

**Usage**

.. code-block:: python

    from config.paths import PROJECT_ROOT, RAW_DATA_CSV, MODELS_DIR

Every script that reads or writes project data should import constants from this
module instead of hard-coding paths. That keeps training, EDA, inference, and
documentation aligned when folders move or the repo is cloned elsewhere.

**Resolved paths**

``PROJECT_ROOT``
    Repository root (parent of the ``src/`` directory that contains ``config/``).

``DATA_DIR`` / ``RAW_DATA_CSV`` / ``CLEAN_DATA_CSV``
    Bundled UCI-style raw file and the cleaned CSV produced by preprocessing.

``SCREENSHOTS_DIR``
    EDA and evaluation plots (confusion matrices, ROC curves, etc.).

``MODELS_DIR``
    Serialized sklearn pipelines and metadata (**not** the ``model_training/`` code package).

``MLRUNS_DIR``
    Local MLflow tracking store (``./mlruns``; use the same URI when starting ``mlflow ui``).

Legacy aliases ``DEFAULT_CSV`` and ``DEFAULT_CLEAN_CSV`` match the raw/clean paths above for
older imports.

**Author.** SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from __future__ import annotations

from pathlib import Path

# Repository root: .../src/config/paths.py → parents: config, src, repo
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_CSV = DATA_DIR / "heart_disease_UCI_dataset.csv"
CLEAN_DATA_CSV = DATA_DIR / "heart_disease_processed_dataset.csv"

SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
# Trained artifact outputs (pickles, metadata) — not the `model_training/` script package.
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Back-compat aliases used across older modules
DEFAULT_CSV = RAW_DATA_CSV
DEFAULT_CLEAN_CSV = CLEAN_DATA_CSV
