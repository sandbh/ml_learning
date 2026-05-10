"""
Project ``src`` package root.

Ensures imports like ``import config`` / ``import api`` resolve when ``src`` is on
``PYTHONPATH`` (Docker ``PYTHONPATH=/app/src``, ``pytest.ini`` ``pythonpath = src``).
Contains ``config``, ``data_preprocessing``, ``eda``, ``model_training``, and ``api`` — no
application logic at this level.

**Author.** SANDIP BHATTACHARYYA — BITS Pilani ID 2025cs05025
"""
