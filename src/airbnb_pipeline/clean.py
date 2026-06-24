"""Data cleaning and standardization."""

import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import get_city_config, get_data_paths
from .utils import ensure_dir, safe_read_csv

logger = logging.getLogger("airbnb_pipeline.clean")


# Cleaning helper functions


def clean_price(value) -> Optional[float]:
    """
    Convert a price string like '$1,234.56' to float.

    Handles: currency symbols, commas, whitespace, empty strings.
    Returns None if the value cannot be parsed.

    >>> clean_price('$1,234.56')
    1234.56
    >>> clean_price(None) is None
    True
    """
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Remove currency symbols, commas, whitespace
    cleaned = re.sub(r"[^\d.]", "", str(value).strip())
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def convert_tf_to_bool(value) -> Optional[bool]:
    """
    Convert 't'/'f' strings to Python booleans.

    >>> convert_tf_to_bool('t')
    True
    >>> convert_tf_to_bool('f')
    False
    >>> convert_tf_to_bool(None) is None
    True
    """
    if pd.isna(value):
        return None
    val_str = str(value).strip().lower()
    if val_str in ("t", "true", "1"):
        return True
    if val_str in ("f", "false", "0"):
        return False
    return None


def parse_date(value) -> Optional[pd.Timestamp]:
    """
    Parse a date string to pandas Timestamp.

    Returns None on failure.

    >>> parse_date('2023-01-15')
    Timestamp('2023-01-15 00:00:00')
    """
    if pd.isna(value) or value == "":
        return None
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


def standardise_text(value: str) -> Optional[str]:
    """Strip whitespace and title-case a text field."""
    if pd.isna(value):
        return None
    return str(value).strip()


# Dataset-specific cleaners


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the detailed listings DataFrame.

    Steps:
    1. Clean price columns.
    2. Parse date columns.
    3. Convert boolean columns.
    4. Standardise text fields.
    5. Handle missing values carefully.
    """
    logger.info("Cleaning listings: %d rows, %d columns", len(df), len(df.columns))
    df = df.copy()

    # --- Price columns ---
    price_cols = [c for c in df.columns if "price" in c.lower()]
    for col in price_cols:
        df[col] = df[col].apply(clean_price)
    logger.debug("Cleaned price columns: %s", price_cols)

    # --- Date columns ---
    date_cols = [c for c in df.columns if c in (
        "host_since", "last_review", "first_review",
        "calendar_last_scraped", "last_scraped",
    )]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.debug("Parsed date columns: %s", date_cols)

    # --- Boolean columns ---
    bool_cols = [c for c in df.columns if c in (
        "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
        "has_availability", "instant_bookable",
    )]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_tf_to_bool)
    logger.debug("Converted boolean columns: %s", bool_cols)

    # --- Text standardisation ---
    if "neighbourhood_cleansed" in df.columns:
        df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].apply(standardise_text)
    if "neighbourhood_group_cleansed" in df.columns:
        df["neighbourhood_group_cleansed"] = df["neighbourhood_group_cleansed"].apply(standardise_text)
    if "room_type" in df.columns:
        df["room_type"] = df["room_type"].apply(standardise_text)
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].apply(standardise_text)

    # --- Missing value handling (careful, not blind imputation) ---
    # Set reviews_per_month to 0 ONLY when number_of_reviews == 0
    if "reviews_per_month" in df.columns and "number_of_reviews" in df.columns:
        mask = (df["number_of_reviews"] == 0) & (df["reviews_per_month"].isna())
        df.loc[mask, "reviews_per_month"] = 0.0
        logger.debug(
            "Set reviews_per_month=0 for %d listings with zero reviews", mask.sum()
        )

    # Do NOT impute review scores — keep nulls intentionally

    logger.info("Listings cleaning complete: %d rows", len(df))
    return df


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the calendar DataFrame.

    Steps:
    1. Parse date column.
    2. Clean price columns.
    3. Convert available column to boolean.
    """
    logger.info("Cleaning calendar: %d rows", len(df))
    df = df.copy()

    # Date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Price
    price_cols = [c for c in df.columns if "price" in c.lower()]
    for col in price_cols:
        df[col] = df[col].apply(clean_price)

    # Available: t/f -> bool
    if "available" in df.columns:
        df["available"] = df["available"].apply(convert_tf_to_bool)

    logger.info("Calendar cleaning complete: %d rows", len(df))
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the detailed reviews DataFrame.

    Steps:
    1. Parse date column.
    2. Strip whitespace from text fields.
    """
    logger.info("Cleaning reviews: %d rows", len(df))
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("Reviews cleaning complete: %d rows", len(df))
    return df


def clean_neighbourhoods(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the neighbourhoods CSV."""
    logger.info("Cleaning neighbourhoods: %d rows", len(df))
    df = df.copy()

    if "neighbourhood" in df.columns:
        df["neighbourhood"] = df["neighbourhood"].apply(standardise_text)
    if "neighbourhood_group" in df.columns:
        df["neighbourhood_group"] = df["neighbourhood_group"].apply(standardise_text)

    logger.info("Neighbourhoods cleaning complete: %d rows", len(df))
    return df


# Main entry point


def clean(city_key: str) -> None:
    """
    Run all cleaning steps and save cleaned parquet files.

    Args:
        city_key: City identifier.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    raw_dir = paths["raw_dir"]
    processed_dir = ensure_dir(paths["processed_dir"])

    logger.info("Starting data cleaning for %s", city_cfg["display_name"])

    # --- Listings ---
    listings_path = raw_dir / "listings_detailed.csv.gz"
    if listings_path.exists():
        df_listings = safe_read_csv(listings_path, low_memory=False)
        df_listings = clean_listings(df_listings)
        out = processed_dir / "clean_listings.parquet"
        df_listings.to_parquet(out, index=False, engine="pyarrow")
        logger.info("Saved %s (%d rows)", out.name, len(df_listings))
    else:
        logger.warning("Listings file not found: %s", listings_path)

    # --- Calendar ---
    calendar_path = raw_dir / "calendar.csv.gz"
    if calendar_path.exists():
        df_calendar = safe_read_csv(calendar_path, low_memory=False)
        df_calendar = clean_calendar(df_calendar)
        out = processed_dir / "clean_calendar.parquet"
        df_calendar.to_parquet(out, index=False, engine="pyarrow")
        logger.info("Saved %s (%d rows)", out.name, len(df_calendar))
    else:
        logger.warning("Calendar file not found: %s", calendar_path)

    # --- Reviews ---
    reviews_path = raw_dir / "reviews_detailed.csv.gz"
    if reviews_path.exists():
        df_reviews = safe_read_csv(reviews_path, low_memory=False)
        df_reviews = clean_reviews(df_reviews)
        out = processed_dir / "clean_reviews.parquet"
        df_reviews.to_parquet(out, index=False, engine="pyarrow")
        logger.info("Saved %s (%d rows)", out.name, len(df_reviews))
    else:
        logger.warning("Reviews file not found: %s", reviews_path)

    # --- Neighbourhoods ---
    neighbourhoods_path = raw_dir / "neighbourhoods.csv"
    if neighbourhoods_path.exists():
        df_neighbourhoods = safe_read_csv(neighbourhoods_path)
        df_neighbourhoods = clean_neighbourhoods(df_neighbourhoods)
        out = processed_dir / "clean_neighbourhoods.parquet"
        df_neighbourhoods.to_parquet(out, index=False, engine="pyarrow")
        logger.info("Saved %s (%d rows)", out.name, len(df_neighbourhoods))
    else:
        logger.warning("Neighbourhoods file not found: %s", neighbourhoods_path)

    logger.info("Data cleaning complete for %s.", city_key)
