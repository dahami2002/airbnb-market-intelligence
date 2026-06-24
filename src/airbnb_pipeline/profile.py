"""Generate data profiling summaries."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import get_city_config, get_data_paths
from .utils import ensure_dir, safe_read_csv

logger = logging.getLogger("airbnb_pipeline.profile")

# Files to profile (key -> expected format)
PROFILABLE_FILES = {
    "listings_detailed": "csv.gz",
    "calendar": "csv.gz",
    "reviews_detailed": "csv.gz",
    "listings_summary": "csv",
    "reviews_summary": "csv",
    "neighbourhoods": "csv",
}


def profile(city_key: str) -> pd.DataFrame:
    """
    Profile all raw data files and save a summary report.

    Args:
        city_key: City identifier.

    Returns:
        DataFrame containing profiling results for all files.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    raw_dir = paths["raw_dir"]
    reports_dir = ensure_dir(paths["reports_dir"])

    logger.info("Starting data profiling for %s", city_cfg["display_name"])

    all_profiles: List[Dict[str, Any]] = []

    for file_key, file_cfg in city_cfg["files"].items():
        local_name = file_cfg["local_name"]

        # Skip geojson — not a tabular file
        if local_name.endswith(".geojson"):
            logger.info("Skipping profiling for %s (GeoJSON)", local_name)
            continue

        filepath = raw_dir / local_name
        if not filepath.exists():
            logger.warning("File not found, skipping: %s", filepath)
            continue

        logger.info("Profiling %s ...", local_name)
        try:
            df = safe_read_csv(filepath, low_memory=False)
            profile_rows = _profile_dataframe(file_key, local_name, df)
            all_profiles.extend(profile_rows)
        except Exception as e:
            logger.error("Error profiling %s: %s", local_name, e)
            all_profiles.append({
                "file_key": file_key,
                "file_name": local_name,
                "column_name": "ERROR",
                "dtype": str(e),
                "row_count": 0,
                "column_count": 0,
                "missing_count": 0,
                "missing_pct": 0.0,
                "unique_count": 0,
                "duplicate_rows": 0,
                "min_value": None,
                "max_value": None,
            })

    profile_df = pd.DataFrame(all_profiles)
    output_path = reports_dir / "profiling_summary.csv"
    profile_df.to_csv(output_path, index=False)
    logger.info("Profiling summary saved to %s (%d rows)", output_path, len(profile_df))

    return profile_df


def _profile_dataframe(
    file_key: str, file_name: str, df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generate per-column profiling statistics for a single DataFrame.

    Args:
        file_key: Logical file key (e.g., 'listings_detailed').
        file_name: Local filename.
        df: The DataFrame to profile.

    Returns:
        List of dicts, one per column, with profiling metrics.
    """
    row_count = len(df)
    col_count = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())

    profiles = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / row_count * 100, 2) if row_count > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        min_val = None
        max_val = None

        # Attempt numeric min/max
        if pd.api.types.is_numeric_dtype(series):
            min_val = series.min()
            max_val = series.max()
            # Convert numpy types to Python natives for clean CSV output
            if pd.notna(min_val):
                min_val = float(min_val)
            if pd.notna(max_val):
                max_val = float(max_val)
        else:
            # Attempt date parsing for date-like columns
            if _is_date_like_column(col):
                try:
                    date_series = pd.to_datetime(series, errors="coerce")
                    valid_dates = date_series.dropna()
                    if len(valid_dates) > 0:
                        min_val = str(valid_dates.min().date())
                        max_val = str(valid_dates.max().date())
                except Exception:
                    pass

        profiles.append({
            "file_key": file_key,
            "file_name": file_name,
            "column_name": col,
            "dtype": str(series.dtype),
            "row_count": row_count,
            "column_count": col_count,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "duplicate_rows": duplicate_rows,
            "min_value": min_val,
            "max_value": max_val,
        })

    return profiles


def _is_date_like_column(col_name: str) -> bool:
    """Heuristic: check if column name suggests a date field."""
    date_keywords = ["date", "since", "last_review", "first_review", "created", "updated"]
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in date_keywords)
