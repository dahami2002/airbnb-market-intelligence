"""Data enrichment - creates derived fields and aggregates."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import get_city_config, get_data_paths
from .utils import ensure_dir

logger = logging.getLogger("airbnb_pipeline.enrich")


def enrich(city_key: str) -> pd.DataFrame:
    """
    Build an enriched listings master table and save as parquet.

    Args:
        city_key: City identifier.

    Returns:
        Enriched DataFrame.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    processed_dir = paths["processed_dir"]

    logger.info("Starting enrichment for %s", city_cfg["display_name"])

    # ── Load cleaned data ──
    df_listings = _load_parquet(processed_dir / "clean_listings.parquet", "listings")
    if df_listings is None:
        raise FileNotFoundError("clean_listings.parquet is required for enrichment.")

    df_calendar = _load_parquet(processed_dir / "clean_calendar.parquet", "calendar")
    df_reviews = _load_parquet(processed_dir / "clean_reviews.parquet", "reviews")

    # ── Calendar aggregates ──
    if df_calendar is not None:
        cal_agg = _aggregate_calendar(df_calendar)
        df_listings = df_listings.merge(cal_agg, left_on="id", right_on="listing_id", how="left")
        if "listing_id" in df_listings.columns:
            df_listings.drop(columns=["listing_id"], inplace=True)
        logger.info("Merged calendar aggregates: %d rows", len(df_listings))
    else:
        logger.warning("Calendar data not available; skipping calendar enrichment.")

    # ── Review aggregates ──
    if df_reviews is not None:
        rev_agg = _aggregate_reviews(df_reviews)
        df_listings = df_listings.merge(rev_agg, left_on="id", right_on="listing_id", how="left")
        if "listing_id" in df_listings.columns:
            df_listings.drop(columns=["listing_id"], inplace=True)
        logger.info("Merged review aggregates: %d rows", len(df_listings))
    else:
        logger.warning("Reviews data not available; skipping review enrichment.")

    # ── Derived fields ──
    df_listings = _add_derived_fields(df_listings, city_cfg)

    # ── Save ──
    output_path = ensure_dir(processed_dir) / "enriched_listings.parquet"
    df_listings.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Enriched listings saved to %s (%d rows, %d columns)",
                output_path, len(df_listings), len(df_listings.columns))

    return df_listings


def _load_parquet(path: Path, name: str) -> Optional[pd.DataFrame]:
    """Load a parquet file, returning None if not found."""
    if path.exists():
        df = pd.read_parquet(path)
        logger.info("Loaded %s: %d rows", name, len(df))
        return df
    logger.warning("File not found: %s", path)
    return None


def _aggregate_calendar(df_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-listing calendar aggregates.

    Returns DataFrame with columns:
        listing_id, estimated_occupancy_rate_365d, average_calendar_price,
        calendar_days_total, calendar_days_unavailable
    """
    agg = df_cal.groupby("listing_id").agg(
        calendar_days_total=("available", "count"),
        calendar_days_unavailable=("available", lambda x: (~x.astype(bool)).sum()),
        average_calendar_price=("price", "mean"),
    ).reset_index()

    # Occupancy proxy: unavailable / total (caveat: includes host-blocked dates)
    agg["estimated_occupancy_rate_365d"] = (
        agg["calendar_days_unavailable"] / agg["calendar_days_total"]
    ).round(4)

    agg["average_calendar_price"] = agg["average_calendar_price"].round(2)

    return agg


def _aggregate_reviews(df_rev: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-listing review aggregates.

    Returns DataFrame with columns:
        listing_id, total_review_count, review_first_date, review_last_date
    """
    agg = df_rev.groupby("listing_id").agg(
        total_review_count=("id", "count") if "id" in df_rev.columns else ("listing_id", "count"),
        review_first_date=("date", "min"),
        review_last_date=("date", "max"),
    ).reset_index()

    return agg


def _add_derived_fields(df: pd.DataFrame, city_cfg: dict) -> pd.DataFrame:
    """Add all derived / enrichment columns to the listings DataFrame."""

    # ── Host tenure (years) ──
    if "host_since" in df.columns:
        reference_date = pd.Timestamp(city_cfg["snapshot_date"])
        df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_tenure_years"] = (
            (reference_date - df["host_since_dt"]).dt.days / 365.25
        ).round(1)
        df.drop(columns=["host_since_dt"], inplace=True, errors="ignore")
    else:
        df["host_tenure_years"] = np.nan

    # ── Price per bedroom ──
    if "price" in df.columns and "bedrooms" in df.columns:
        df["price_per_bedroom"] = np.where(
            df["bedrooms"] > 0,
            (df["price"] / df["bedrooms"]).round(2),
            np.nan,
        )
    else:
        df["price_per_bedroom"] = np.nan

    # ── Professional host flag ──
    host_count_col = "calculated_host_listings_count"
    if host_count_col in df.columns:
        df["is_professional_host"] = df[host_count_col] >= 3
    else:
        df["is_professional_host"] = False

    # ── Host tier ──
    if host_count_col in df.columns:
        df["host_tier"] = pd.cut(
            df[host_count_col].fillna(1),
            bins=[0, 1, 2, 9, float("inf")],
            labels=["Solo Host", "Small Portfolio", "Mid Portfolio", "Commercial Operator"],
            right=True,
        )
    else:
        df["host_tier"] = "Solo Host"

    # ── Estimated annual revenue proxy ──
    # Revenue = average_calendar_price * days_unavailable (proxy for nights booked)
    # CAVEAT: unavailable != booked; this is an upper-bound proxy.
    if "average_calendar_price" in df.columns and "calendar_days_unavailable" in df.columns:
        df["estimated_annual_revenue_proxy"] = (
            df["average_calendar_price"] * df["calendar_days_unavailable"]
        ).round(2)
    else:
        df["estimated_annual_revenue_proxy"] = np.nan

    logger.info("Added derived fields: host_tenure_years, price_per_bedroom, "
                "is_professional_host, host_tier, estimated_annual_revenue_proxy")

    return df
