"""Data quality validation checks."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import get_city_config, get_data_paths
from .utils import ensure_dir

logger = logging.getLogger("airbnb_pipeline.quality")


# Individual quality checks


def check_not_null(df: pd.DataFrame, column: str, table_name: str, severity: str = "ERROR") -> Dict[str, Any]:
    """Check that a column contains no null values."""
    total = len(df)
    failed = int(df[column].isna().sum()) if column in df.columns else total
    status = "PASS" if failed == 0 else "FAIL"
    return {
        "check_name": f"{column}_not_null",
        "table_name": table_name,
        "status": status,
        "failed_rows": failed,
        "total_rows": total,
        "severity": severity,
        "notes": f"{column} has {failed} null values" if failed > 0 else "All values present",
    }


def check_no_duplicates(df: pd.DataFrame, column: str, table_name: str, severity: str = "WARNING") -> Dict[str, Any]:
    """Check for duplicate values in a column."""
    total = len(df)
    if column not in df.columns:
        return {
            "check_name": f"{column}_no_duplicates",
            "table_name": table_name,
            "status": "FAIL",
            "failed_rows": total,
            "total_rows": total,
            "severity": severity,
            "notes": f"Column {column} not found in {table_name}",
        }
    dup_count = int(df[column].duplicated().sum())
    status = "PASS" if dup_count == 0 else "WARN"
    return {
        "check_name": f"{column}_no_duplicates",
        "table_name": table_name,
        "status": status,
        "failed_rows": dup_count,
        "total_rows": total,
        "severity": severity,
        "notes": f"{dup_count} duplicate {column} values" if dup_count > 0 else "No duplicates",
    }


def check_range(
    df: pd.DataFrame, column: str, table_name: str,
    min_val: float, max_val: float, severity: str = "ERROR"
) -> Dict[str, Any]:
    """Check that numeric values fall within [min_val, max_val]."""
    total = len(df)
    if column not in df.columns:
        return {
            "check_name": f"{column}_range_{min_val}_{max_val}",
            "table_name": table_name,
            "status": "FAIL",
            "failed_rows": total,
            "total_rows": total,
            "severity": severity,
            "notes": f"Column {column} not found",
        }
    # Only check non-null values
    non_null = df[column].dropna()
    out_of_range = int(((non_null < min_val) | (non_null > max_val)).sum())
    status = "PASS" if out_of_range == 0 else "FAIL"
    return {
        "check_name": f"{column}_in_range_[{min_val},{max_val}]",
        "table_name": table_name,
        "status": status,
        "failed_rows": out_of_range,
        "total_rows": total,
        "severity": severity,
        "notes": f"{out_of_range} values outside [{min_val}, {max_val}]" if out_of_range > 0 else "All in range",
    }


def check_positive(df: pd.DataFrame, column: str, table_name: str, severity: str = "WARNING") -> Dict[str, Any]:
    """Check that numeric values are greater than 0 (for price analysis)."""
    total = len(df)
    if column not in df.columns:
        return {
            "check_name": f"{column}_positive",
            "table_name": table_name,
            "status": "FAIL",
            "failed_rows": total,
            "total_rows": total,
            "severity": severity,
            "notes": f"Column {column} not found",
        }
    non_null = df[column].dropna()
    non_positive = int((non_null <= 0).sum())
    status = "PASS" if non_positive == 0 else "WARN"
    return {
        "check_name": f"{column}_greater_than_0",
        "table_name": table_name,
        "status": status,
        "failed_rows": non_positive,
        "total_rows": total,
        "severity": severity,
        "notes": f"{non_positive} values <= 0" if non_positive > 0 else "All positive",
    }


def check_boolean(df: pd.DataFrame, column: str, table_name: str, severity: str = "WARNING") -> Dict[str, Any]:
    """Check that a column only contains boolean values (True/False/None)."""
    total = len(df)
    if column not in df.columns:
        return {
            "check_name": f"{column}_is_boolean",
            "table_name": table_name,
            "status": "FAIL",
            "failed_rows": total,
            "total_rows": total,
            "severity": severity,
            "notes": f"Column {column} not found",
        }
    non_null = df[column].dropna()
    invalid = int((~non_null.isin([True, False])).sum())
    status = "PASS" if invalid == 0 else "WARN"
    return {
        "check_name": f"{column}_is_boolean",
        "table_name": table_name,
        "status": status,
        "failed_rows": invalid,
        "total_rows": total,
        "severity": severity,
        "notes": f"{invalid} non-boolean values" if invalid > 0 else "All boolean",
    }


def check_valid_date(df: pd.DataFrame, column: str, table_name: str, severity: str = "WARNING") -> Dict[str, Any]:
    """Check that a date column has valid (non-NaT) datetime values."""
    total = len(df)
    if column not in df.columns:
        return {
            "check_name": f"{column}_valid_date",
            "table_name": table_name,
            "status": "FAIL",
            "failed_rows": total,
            "total_rows": total,
            "severity": severity,
            "notes": f"Column {column} not found",
        }
    invalid = int(pd.to_datetime(df[column], errors="coerce").isna().sum())
    status = "PASS" if invalid == 0 else "WARN"
    return {
        "check_name": f"{column}_valid_date",
        "table_name": table_name,
        "status": status,
        "failed_rows": invalid,
        "total_rows": total,
        "severity": severity,
        "notes": f"{invalid} invalid/missing dates" if invalid > 0 else "All dates valid",
    }


# Run all checks and save report


def run_quality_checks(city_key: str) -> pd.DataFrame:
    """
    Execute all data quality checks on cleaned data and save report.

    Args:
        city_key: City identifier.

    Returns:
        DataFrame of quality check results.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    processed_dir = paths["processed_dir"]
    reports_dir = ensure_dir(paths["reports_dir"])

    results: List[Dict[str, Any]] = []

    # ── Listings checks ──
    listings_path = processed_dir / "clean_listings.parquet"
    if listings_path.exists():
        logger.info("Running quality checks on listings...")
        df = pd.read_parquet(listings_path)

        results.append(check_not_null(df, "id", "listings", "ERROR"))
        results.append(check_no_duplicates(df, "id", "listings", "ERROR"))
        results.append(check_positive(df, "price", "listings", "WARNING"))
        results.append(check_range(df, "latitude", "listings", -90, 90, "ERROR"))
        results.append(check_range(df, "longitude", "listings", -180, 180, "ERROR"))
        results.append(check_range(df, "availability_365", "listings", 0, 365, "WARNING"))
        results.append(check_not_null(df, "room_type", "listings", "ERROR"))
    else:
        logger.warning("Clean listings file not found: %s", listings_path)

    # ── Calendar checks ──
    calendar_path = processed_dir / "clean_calendar.parquet"
    if calendar_path.exists():
        logger.info("Running quality checks on calendar...")
        df_cal = pd.read_parquet(calendar_path)

        results.append(check_not_null(df_cal, "listing_id", "calendar", "ERROR"))
        results.append(check_valid_date(df_cal, "date", "calendar", "ERROR"))
        results.append(check_boolean(df_cal, "available", "calendar", "WARNING"))
    else:
        logger.warning("Clean calendar file not found: %s", calendar_path)

    # ── Reviews checks ──
    reviews_path = processed_dir / "clean_reviews.parquet"
    if reviews_path.exists():
        logger.info("Running quality checks on reviews...")
        df_rev = pd.read_parquet(reviews_path)

        results.append(check_not_null(df_rev, "listing_id", "reviews", "ERROR"))
        results.append(check_valid_date(df_rev, "date", "reviews", "WARNING"))
    else:
        logger.warning("Clean reviews file not found: %s", reviews_path)

    # ── Save report ──
    results_df = pd.DataFrame(results)
    output_path = reports_dir / "data_quality_report.csv"
    results_df.to_csv(output_path, index=False)

    # Summary
    pass_count = (results_df["status"] == "PASS").sum()
    total_checks = len(results_df)
    logger.info(
        "Quality checks complete: %d/%d passed. Report saved to %s",
        pass_count, total_checks, output_path,
    )

    # Log failures but do NOT crash
    failures = results_df[results_df["status"] == "FAIL"]
    for _, row in failures.iterrows():
        logger.warning(
            "QUALITY FAIL [%s] %s.%s: %s",
            row["severity"], row["table_name"], row["check_name"], row["notes"],
        )

    return results_df
