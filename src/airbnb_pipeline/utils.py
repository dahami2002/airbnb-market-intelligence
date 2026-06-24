"""Shared utilities for the pipeline."""

import logging
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """Create directory and parents if it does not exist. Return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the pipeline logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("airbnb_pipeline")
    logger.setLevel(log_level)
    return logger


def file_size_mb(filepath: Path) -> float:
    """Return file size in megabytes, rounded to 2 decimal places."""
    if filepath.exists():
        return round(filepath.stat().st_size / (1024 * 1024), 2)
    return 0.0


def safe_read_csv(filepath: Path, **kwargs: Any):
    """
    Read a CSV or CSV.GZ file with pandas.

    Args:
        filepath: Path to CSV or CSV.GZ file.
        **kwargs: Extra arguments passed to pandas.read_csv.
    """
    import pandas as pd

    compression = "gzip" if str(filepath).endswith(".gz") else None
    return pd.read_csv(filepath, compression=compression, **kwargs)
