"""
Model training and batch inference **scripts** (**File:**
``src/model_training/__init__.py``).

**Contents**

- ``train.py`` — Builds sklearn pipelines, ``GridSearchCV``, evaluation
  metrics, MLflow experiment logging under ``mlruns/``, and writes the winning
  pipeline plus metadata to ``models/``.
- ``inference.py`` — Loads ``models/best_model.pkl``, reapplies the same
  preprocessing as training on a raw CSV, writes optional batch predictions.

**Distinction.** The directory ``models/`` (plural) holds **artifacts**
(``best_model.pkl``, ``feature_names.pkl``, ``training_metadata.json``). This
package ``model_training/`` holds **code** only.

**Upstream.** Requires ``data/heart_disease_processed_dataset.csv`` from
``eda/eda.py`` for training and ``data_preprocessing`` for prediction feature
parity.

This package intentionally exports no runtime API beyond submodule imports;
run scripts with ``python src/model_training/train.py`` from the repo root
(see README).

**Author.** SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

__all__: list[str] = []
