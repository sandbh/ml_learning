"""
Project configuration package (**File:** ``src/config/__init__.py``).

Re-exports path constants from :mod:`config.paths` so other modules can use::

    from config import MODELS_DIR, CLEAN_DATA_CSV

instead of importing ``paths`` directly. Extend this package if you add shared
non-path settings (random seeds, API URLs) later.

**Author.** SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""

from config.paths import (
    CLEAN_DATA_CSV,
    DATA_DIR,
    DEFAULT_CLEAN_CSV,
    DEFAULT_CSV,
    MLRUNS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RAW_DATA_CSV,
    SCREENSHOTS_DIR,
)

__all__ = [
    "CLEAN_DATA_CSV",
    "DATA_DIR",
    "DEFAULT_CLEAN_CSV",
    "DEFAULT_CSV",
    "MLRUNS_DIR",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_CSV",
    "SCREENSHOTS_DIR",
]
