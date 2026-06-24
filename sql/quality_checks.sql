-- ===================================================================
-- Airbnb Bangkok — Data Quality Check Queries
-- Run against: data/warehouse/airbnb_bangkok.duckdb
-- ===================================================================

-- 1. Null listing IDs
SELECT 'listing_id_null' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN listing_id IS NULL THEN 1 ELSE 0 END) AS failed_rows
FROM dim_listing;

-- 2. Duplicate listing IDs
SELECT 'duplicate_listing_ids' AS check_name,
       COUNT(*) - COUNT(DISTINCT listing_id) AS duplicate_count
FROM dim_listing;

-- 3. Price > 0 for price analysis
SELECT 'price_positive' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN price <= 0 OR price IS NULL THEN 1 ELSE 0 END) AS failed_rows
FROM fact_listing_snapshot;

-- 4. Valid latitude range
SELECT 'latitude_range' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN latitude < -90 OR latitude > 90 THEN 1 ELSE 0 END) AS failed_rows
FROM dim_listing
WHERE latitude IS NOT NULL;

-- 5. Valid longitude range
SELECT 'longitude_range' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN longitude < -180 OR longitude > 180 THEN 1 ELSE 0 END) AS failed_rows
FROM dim_listing
WHERE longitude IS NOT NULL;

-- 6. availability_365 in [0, 365]
SELECT 'availability_365_range' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN availability_365 < 0 OR availability_365 > 365 THEN 1 ELSE 0 END) AS failed_rows
FROM fact_listing_snapshot
WHERE availability_365 IS NOT NULL;

-- 7. Room type not null
SELECT 'room_type_not_null' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN room_type IS NULL THEN 1 ELSE 0 END) AS failed_rows
FROM dim_listing;

-- 8. Calendar listing_id not null
SELECT 'calendar_listing_id_not_null' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN listing_id IS NULL THEN 1 ELSE 0 END) AS failed_rows
FROM fact_calendar;

-- 9. Calendar date valid (not null)
SELECT 'calendar_date_valid' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN date_key IS NULL THEN 1 ELSE 0 END) AS failed_rows
FROM fact_calendar;

-- 10. Reviews listing_id not null
SELECT 'reviews_listing_id_not_null' AS check_name,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN listing_id IS NULL THEN 1 ELSE 0 END) AS failed_rows
FROM fact_reviews;
