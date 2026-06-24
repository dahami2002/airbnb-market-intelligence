
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
