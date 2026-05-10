"""
Data preprocessing package (**File:** ``src/data_preprocessing/__init__.py``).

Load raw heart-disease CSV, clean, export.

**Purpose.** Shared logic for turning ``data/heart_disease_UCI_dataset.csv``
into a model-ready table (encoding, imputation, binary target). Used by
``eda/eda.py`` when building ``data/heart_disease_processed_dataset.csv`` and
by ``model_training/inference.py`` so inference matches training inputs.

**Naming.** The folder is ``data_preprocessing`` (underscores) for valid Python
imports; documentation may refer to the same stage as "data-preprocessing".

**Exports.** ``load_data``, ``clean_data``, ``save_cleaned_csv``, plus path
aliases mirroring :mod:`config.paths`. Implementation lives in
:mod:`data_preprocessing.pre_processing_data`.

**Author.** SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from .pre_processing_data import (
    DEFAULT_CLEAN_CSV,
    DEFAULT_CSV,
    PROJECT_ROOT,
    clean_data,
    load_data,
    save_cleaned_csv,
)

__all__ = [
    "DEFAULT_CLEAN_CSV",
    "DEFAULT_CSV",
    "PROJECT_ROOT",
    "clean_data",
    "load_data",
    "save_cleaned_csv",
]
