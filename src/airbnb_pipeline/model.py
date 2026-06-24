"""Build DuckDB warehouse with star schema."""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .config import get_city_config, get_data_paths
from .utils import ensure_dir

logger = logging.getLogger("airbnb_pipeline.model")


# Schema SQL definition

SCHEMA_SQL = """
-- ===================================================================
-- Airbnb Bangkok Data Warehouse — Star Schema
-- Database: DuckDB
-- ===================================================================

-- ─────────────────── Dimension: Neighbourhood ───────────────────────
CREATE OR REPLACE TABLE dim_neighbourhood (
    neighbourhood_key   INTEGER PRIMARY KEY,
    neighbourhood_name  VARCHAR,
    neighbourhood_group VARCHAR
);

-- ─────────────────── Dimension: Host ────────────────────────────────
CREATE OR REPLACE TABLE dim_host (
    host_key                        INTEGER PRIMARY KEY,
    host_id                         BIGINT,
    host_name                       VARCHAR,
    host_since                      DATE,
    host_tenure_years               DOUBLE,
    host_is_superhost               BOOLEAN,
    host_total_listings_count       INTEGER,
    calculated_host_listings_count  INTEGER,
    is_professional_host            BOOLEAN,
    host_tier                       VARCHAR
);

-- ─────────────────── Dimension: Listing ─────────────────────────────
CREATE OR REPLACE TABLE dim_listing (
    listing_key             INTEGER PRIMARY KEY,
    listing_id              BIGINT,
    name                    VARCHAR,
    description             VARCHAR,
    neighbourhood_key       INTEGER,          -- FK -> dim_neighbourhood
    host_key                INTEGER,          -- FK -> dim_host
    latitude                DOUBLE,
    longitude               DOUBLE,
    property_type           VARCHAR,
    room_type               VARCHAR,
    accommodates            INTEGER,
    bedrooms                DOUBLE,
    beds                    DOUBLE,
    bathrooms_text          VARCHAR,
    minimum_nights          INTEGER,
    maximum_nights          INTEGER,
    instant_bookable        BOOLEAN
);

-- ─────────────────── Dimension: Date ────────────────────────────────
CREATE OR REPLACE TABLE dim_date (
    date_key        INTEGER PRIMARY KEY,     -- YYYYMMDD
    full_date       DATE,
    year            INTEGER,
    month           INTEGER,
    day             INTEGER,
    day_of_week     INTEGER,                 -- 0=Monday ... 6=Sunday
    day_name        VARCHAR,
    is_weekend      BOOLEAN,
    month_name      VARCHAR,
    quarter         INTEGER
);

-- ─────────────────── Fact: Listing Snapshot ──────────────────────────
-- One row per listing at snapshot time; contains pricing and scores.
CREATE OR REPLACE TABLE fact_listing_snapshot (
    listing_key                     INTEGER,  -- FK -> dim_listing
    price                           DOUBLE,
    price_per_bedroom               DOUBLE,
    number_of_reviews               INTEGER,
    number_of_reviews_ltm           INTEGER,
    review_scores_rating            DOUBLE,
    review_scores_accuracy          DOUBLE,
    review_scores_cleanliness       DOUBLE,
    review_scores_checkin           DOUBLE,
    review_scores_communication     DOUBLE,
    review_scores_location          DOUBLE,
    review_scores_value             DOUBLE,
    reviews_per_month               DOUBLE,
    availability_30                 INTEGER,
    availability_60                 INTEGER,
    availability_90                 INTEGER,
    availability_365                INTEGER,
    estimated_occupancy_rate_365d   DOUBLE,
    average_calendar_price          DOUBLE,
    estimated_annual_revenue_proxy  DOUBLE
);

-- ─────────────────── Fact: Calendar ─────────────────────────────────
CREATE OR REPLACE TABLE fact_calendar (
    listing_id      BIGINT,
    date_key        INTEGER,                 -- FK -> dim_date
    available       BOOLEAN,
    price           DOUBLE,
    adjusted_price  DOUBLE,
    minimum_nights  INTEGER,
    maximum_nights  INTEGER
);

-- ─────────────────── Fact: Reviews ──────────────────────────────────
CREATE OR REPLACE TABLE fact_reviews (
    review_id       BIGINT,
    listing_id      BIGINT,
    date_key        INTEGER,                 -- FK -> dim_date
    reviewer_id     BIGINT,
    reviewer_name   VARCHAR
);
"""


def build_warehouse(city_key: str) -> None:
    """
    Create DuckDB warehouse, build star schema, and load data.

    Args:
        city_key: City identifier.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    processed_dir = paths["processed_dir"]
    warehouse_dir = ensure_dir(paths["warehouse_dir"])
    reports_dir = ensure_dir(paths["reports_dir"])

    db_path = warehouse_dir / "airbnb_bangkok.duckdb"
    logger.info("Building DuckDB warehouse at %s", db_path)

    # Save schema SQL file
    sql_dir = ensure_dir(paths["reports_dir"].parent / "sql")
    schema_sql_path = sql_dir / "create_schema.sql"
    schema_sql_path.write_text(SCHEMA_SQL, encoding="utf-8")
    logger.info("Schema SQL saved to %s", schema_sql_path)

    con = duckdb.connect(str(db_path))

    try:
        # Create schema
        con.execute(SCHEMA_SQL)
        logger.info("Schema created successfully.")

        # Load data
        _load_dimensions(con, processed_dir)
        _load_facts(con, processed_dir)

        # Verify
        _verify_warehouse(con)

    finally:
        con.close()

    logger.info("Warehouse build complete.")


def _load_dimensions(con: duckdb.DuckDBPyConnection, processed_dir: Path) -> None:
    """Load dimension tables from processed parquet files."""

    # ── dim_neighbourhood ──
    nbr_path = processed_dir / "clean_neighbourhoods.parquet"
    if nbr_path.exists():
        df_nbr = pd.read_parquet(nbr_path)
        df_nbr = df_nbr.drop_duplicates(subset=["neighbourhood"] if "neighbourhood" in df_nbr.columns else None)
        df_nbr["neighbourhood_key"] = range(1, len(df_nbr) + 1)
        df_nbr = df_nbr.rename(columns={
            "neighbourhood": "neighbourhood_name",
            "neighbourhood_group": "neighbourhood_group",
        })
        cols = ["neighbourhood_key", "neighbourhood_name"]
        if "neighbourhood_group" in df_nbr.columns:
            cols.append("neighbourhood_group")
        else:
            df_nbr["neighbourhood_group"] = None
            cols.append("neighbourhood_group")
        con.execute("DELETE FROM dim_neighbourhood")
        con.execute("INSERT INTO dim_neighbourhood SELECT * FROM df_nbr[cols]")
        logger.info("Loaded dim_neighbourhood: %d rows", len(df_nbr))
    else:
        logger.warning("Neighbourhoods parquet not found.")

    # ── dim_host ──
    listings_path = processed_dir / "enriched_listings.parquet"
    if listings_path.exists():
        df = pd.read_parquet(listings_path)

        host_cols = ["host_id", "host_name", "host_since", "host_tenure_years",
                     "host_is_superhost", "host_total_listings_count",
                     "calculated_host_listings_count", "is_professional_host", "host_tier"]
        available_cols = [c for c in host_cols if c in df.columns]
        df_hosts = df[available_cols].copy()
        df_hosts = df_hosts.drop_duplicates(subset=["host_id"])
        df_hosts["host_key"] = range(1, len(df_hosts) + 1)

        # Ensure all columns exist
        for c in host_cols:
            if c not in df_hosts.columns:
                df_hosts[c] = None

        if "host_since" in df_hosts.columns:
            df_hosts["host_since"] = pd.to_datetime(df_hosts["host_since"], errors="coerce")
        if "host_tier" in df_hosts.columns:
            df_hosts["host_tier"] = df_hosts["host_tier"].astype(str)

        insert_cols = ["host_key"] + host_cols
        con.execute("DELETE FROM dim_host")
        con.execute(f"INSERT INTO dim_host SELECT {', '.join(insert_cols)} FROM df_hosts")
        logger.info("Loaded dim_host: %d rows", len(df_hosts))

        # ── dim_listing ──
        # Build neighbourhood lookup
        nbr_lookup = {}
        try:
            nbr_data = con.execute("SELECT neighbourhood_key, neighbourhood_name FROM dim_neighbourhood").fetchdf()
            nbr_lookup = dict(zip(nbr_data["neighbourhood_name"], nbr_data["neighbourhood_key"]))
        except Exception:
            pass

        host_lookup = dict(zip(df_hosts["host_id"], df_hosts["host_key"]))

        listing_cols_map = {
            "id": "listing_id", "name": "name", "description": "description",
            "latitude": "latitude", "longitude": "longitude",
            "property_type": "property_type", "room_type": "room_type",
            "accommodates": "accommodates", "bedrooms": "bedrooms",
            "beds": "beds", "bathrooms_text": "bathrooms_text",
            "minimum_nights": "minimum_nights", "maximum_nights": "maximum_nights",
            "instant_bookable": "instant_bookable",
        }

        df_dim_listing = pd.DataFrame()
        df_dim_listing["listing_key"] = range(1, len(df) + 1)
        for src, dst in listing_cols_map.items():
            df_dim_listing[dst] = df[src].values if src in df.columns else None

        # Map foreign keys
        nbr_col = "neighbourhood_cleansed" if "neighbourhood_cleansed" in df.columns else None
        if nbr_col:
            df_dim_listing["neighbourhood_key"] = df[nbr_col].map(nbr_lookup).values
        else:
            df_dim_listing["neighbourhood_key"] = None

        df_dim_listing["host_key"] = df["host_id"].map(host_lookup).values if "host_id" in df.columns else None

        dim_listing_insert = [
            "listing_key", "listing_id", "name", "description",
            "neighbourhood_key", "host_key", "latitude", "longitude",
            "property_type", "room_type", "accommodates", "bedrooms",
            "beds", "bathrooms_text", "minimum_nights", "maximum_nights",
            "instant_bookable",
        ]
        con.execute("DELETE FROM dim_listing")
        con.execute(f"INSERT INTO dim_listing SELECT {', '.join(dim_listing_insert)} FROM df_dim_listing")
        logger.info("Loaded dim_listing: %d rows", len(df_dim_listing))

    # ── dim_date ──
    cal_path = processed_dir / "clean_calendar.parquet"
    if cal_path.exists():
        df_cal = pd.read_parquet(cal_path, columns=["date"])
        dates = pd.to_datetime(df_cal["date"].dropna().unique())
        df_date = pd.DataFrame({"full_date": sorted(dates)})
        df_date["date_key"] = df_date["full_date"].dt.strftime("%Y%m%d").astype(int)
        df_date["year"] = df_date["full_date"].dt.year
        df_date["month"] = df_date["full_date"].dt.month
        df_date["day"] = df_date["full_date"].dt.day
        df_date["day_of_week"] = df_date["full_date"].dt.dayofweek
        df_date["day_name"] = df_date["full_date"].dt.day_name()
        df_date["is_weekend"] = df_date["day_of_week"].isin([5, 6])
        df_date["month_name"] = df_date["full_date"].dt.month_name()
        df_date["quarter"] = df_date["full_date"].dt.quarter

        con.execute("DELETE FROM dim_date")
        con.execute("INSERT INTO dim_date SELECT date_key, full_date, year, month, day, "
                    "day_of_week, day_name, is_weekend, month_name, quarter FROM df_date")
        logger.info("Loaded dim_date: %d rows", len(df_date))


def _load_facts(con: duckdb.DuckDBPyConnection, processed_dir: Path) -> None:
    """Load fact tables."""

    # ── fact_listing_snapshot ──
    enriched_path = processed_dir / "enriched_listings.parquet"
    if enriched_path.exists():
        df = pd.read_parquet(enriched_path)

        # Build listing key lookup
        try:
            lk = con.execute("SELECT listing_key, listing_id FROM dim_listing").fetchdf()
            lk_map = dict(zip(lk["listing_id"], lk["listing_key"]))
        except Exception:
            lk_map = {}

        fact_cols = [
            "price", "price_per_bedroom", "number_of_reviews", "number_of_reviews_ltm",
            "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication", "review_scores_location",
            "review_scores_value", "reviews_per_month",
            "availability_30", "availability_60", "availability_90", "availability_365",
            "estimated_occupancy_rate_365d", "average_calendar_price",
            "estimated_annual_revenue_proxy",
        ]

        df_fact = pd.DataFrame()
        df_fact["listing_key"] = df["id"].map(lk_map) if "id" in df.columns else None
        for c in fact_cols:
            df_fact[c] = df[c].values if c in df.columns else None

        con.execute("DELETE FROM fact_listing_snapshot")
        insert_cols = ["listing_key"] + fact_cols
        con.execute(f"INSERT INTO fact_listing_snapshot SELECT {', '.join(insert_cols)} FROM df_fact")
        logger.info("Loaded fact_listing_snapshot: %d rows", len(df_fact))

    # ── fact_calendar ──
    cal_path = processed_dir / "clean_calendar.parquet"
    if cal_path.exists():
        df_cal = pd.read_parquet(cal_path)
        df_cal["date"] = pd.to_datetime(df_cal["date"], errors="coerce")
        df_cal["date_key"] = df_cal["date"].dt.strftime("%Y%m%d").astype("Int64")

        cal_insert_cols = ["listing_id", "date_key", "available", "price"]
        if "adjusted_price" in df_cal.columns:
            cal_insert_cols.append("adjusted_price")
        else:
            df_cal["adjusted_price"] = None
            cal_insert_cols.append("adjusted_price")
        if "minimum_nights" in df_cal.columns:
            cal_insert_cols.append("minimum_nights")
        else:
            df_cal["minimum_nights"] = None
            cal_insert_cols.append("minimum_nights")
        if "maximum_nights" in df_cal.columns:
            cal_insert_cols.append("maximum_nights")
        else:
            df_cal["maximum_nights"] = None
            cal_insert_cols.append("maximum_nights")

        con.execute("DELETE FROM fact_calendar")
        con.execute(f"INSERT INTO fact_calendar SELECT {', '.join(cal_insert_cols)} FROM df_cal")
        logger.info("Loaded fact_calendar: %d rows", len(df_cal))

    # ── fact_reviews ──
    rev_path = processed_dir / "clean_reviews.parquet"
    if rev_path.exists():
        df_rev = pd.read_parquet(rev_path)
        df_rev["date"] = pd.to_datetime(df_rev["date"], errors="coerce")
        df_rev["date_key"] = df_rev["date"].dt.strftime("%Y%m%d").astype("Int64")

        rev_cols = ["listing_id", "date_key"]
        for c in ["id", "reviewer_id", "reviewer_name"]:
            if c in df_rev.columns:
                rev_cols.append(c)

        # Rename 'id' to 'review_id' for the fact table
        df_rev_fact = df_rev[rev_cols].copy()
        if "id" in df_rev_fact.columns:
            df_rev_fact = df_rev_fact.rename(columns={"id": "review_id"})

        for c in ["review_id", "reviewer_id", "reviewer_name"]:
            if c not in df_rev_fact.columns:
                df_rev_fact[c] = None

        fact_rev_insert = ["review_id", "listing_id", "date_key", "reviewer_id", "reviewer_name"]
        con.execute("DELETE FROM fact_reviews")
        con.execute(f"INSERT INTO fact_reviews SELECT {', '.join(fact_rev_insert)} FROM df_rev_fact")
        logger.info("Loaded fact_reviews: %d rows", len(df_rev_fact))


def _verify_warehouse(con: duckdb.DuckDBPyConnection) -> None:
    """Log row counts for all tables to verify load."""
    tables = ["dim_neighbourhood", "dim_host", "dim_listing", "dim_date",
              "fact_listing_snapshot", "fact_calendar", "fact_reviews"]
    for table in tables:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info("  %-25s %10d rows", table, count)
        except Exception as e:
            logger.warning("  %-25s ERROR: %s", table, e)
