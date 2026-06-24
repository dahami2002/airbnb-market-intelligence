-- ===================================================================
-- Airbnb Bangkok — Analytical SQL Queries
-- Run against: data/warehouse/airbnb_bangkok.duckdb
-- ===================================================================

-- ─────────────────────────────────────────────────────────────────────
-- 1. Median Price by Neighbourhood
-- ─────────────────────────────────────────────────────────────────────
SELECT
    n.neighbourhood_name,
    MEDIAN(f.price)          AS median_price,
    COUNT(*)                 AS listing_count
FROM fact_listing_snapshot f
JOIN dim_listing l ON f.listing_key = l.listing_key
JOIN dim_neighbourhood n ON l.neighbourhood_key = n.neighbourhood_key
WHERE f.price > 0
GROUP BY n.neighbourhood_name
ORDER BY median_price DESC;


-- ─────────────────────────────────────────────────────────────────────
-- 2. Price by Room Type
-- ─────────────────────────────────────────────────────────────────────
SELECT
    l.room_type,
    MEDIAN(f.price)          AS median_price,
    AVG(f.price)             AS avg_price,
    COUNT(*)                 AS listing_count
FROM fact_listing_snapshot f
JOIN dim_listing l ON f.listing_key = l.listing_key
WHERE f.price > 0
GROUP BY l.room_type
ORDER BY median_price DESC;


-- ─────────────────────────────────────────────────────────────────────
-- 3. Top Hosts by Listing Count
-- ─────────────────────────────────────────────────────────────────────
SELECT
    h.host_id,
    h.host_name,
    h.calculated_host_listings_count,
    h.host_is_superhost,
    h.host_tier
FROM dim_host h
ORDER BY h.calculated_host_listings_count DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────────────
-- 4. Professional Host vs Casual Host Comparison
-- ─────────────────────────────────────────────────────────────────────
SELECT
    h.is_professional_host,
    COUNT(DISTINCT l.listing_key)     AS listing_count,
    COUNT(DISTINCT h.host_id)         AS host_count,
    MEDIAN(f.price)                   AS median_price,
    AVG(f.review_scores_rating)       AS avg_review_score,
    AVG(f.availability_365)           AS avg_availability_365
FROM fact_listing_snapshot f
JOIN dim_listing l ON f.listing_key = l.listing_key
JOIN dim_host h ON l.host_key = h.host_key
GROUP BY h.is_professional_host;


-- ─────────────────────────────────────────────────────────────────────
-- 5. Weekend vs Weekday Calendar Pricing
-- ─────────────────────────────────────────────────────────────────────
SELECT
    d.is_weekend,
    CASE WHEN d.is_weekend THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    MEDIAN(c.price)          AS median_price,
    AVG(c.price)             AS avg_price,
    COUNT(*)                 AS record_count
FROM fact_calendar c
JOIN dim_date d ON c.date_key = d.date_key
WHERE c.price > 0
GROUP BY d.is_weekend
ORDER BY d.is_weekend;


-- ─────────────────────────────────────────────────────────────────────
-- 6. Monthly Average Price and Availability
-- ─────────────────────────────────────────────────────────────────────
SELECT
    d.year,
    d.month,
    d.month_name,
    ROUND(AVG(c.price), 2)                                  AS avg_price,
    ROUND(AVG(CASE WHEN c.available THEN 1.0 ELSE 0.0 END) * 100, 1) AS availability_pct,
    COUNT(*)                                                 AS records
FROM fact_calendar c
JOIN dim_date d ON c.date_key = d.date_key
WHERE c.price > 0
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;


-- ─────────────────────────────────────────────────────────────────────
-- 7. Listings with High Review Count but Low Review Score
--    (potential quality concerns)
-- ─────────────────────────────────────────────────────────────────────
SELECT
    l.listing_id,
    l.name,
    l.room_type,
    n.neighbourhood_name,
    f.number_of_reviews,
    f.review_scores_rating,
    f.price
FROM fact_listing_snapshot f
JOIN dim_listing l ON f.listing_key = l.listing_key
LEFT JOIN dim_neighbourhood n ON l.neighbourhood_key = n.neighbourhood_key
WHERE f.number_of_reviews >= 20
  AND f.review_scores_rating < 4.0
  AND f.review_scores_rating IS NOT NULL
ORDER BY f.number_of_reviews DESC
LIMIT 30;


-- ─────────────────────────────────────────────────────────────────────
-- 8. Neighbourhood Summary
--    Listing count, median price, average rating, availability
-- ─────────────────────────────────────────────────────────────────────
SELECT
    n.neighbourhood_name,
    COUNT(*)                             AS listing_count,
    MEDIAN(f.price)                      AS median_price,
    ROUND(AVG(f.review_scores_rating), 2) AS avg_rating,
    ROUND(AVG(f.availability_365), 0)    AS avg_availability_365,
    ROUND(AVG(f.estimated_occupancy_rate_365d) * 100, 1) AS avg_occupancy_pct
FROM fact_listing_snapshot f
JOIN dim_listing l ON f.listing_key = l.listing_key
JOIN dim_neighbourhood n ON l.neighbourhood_key = n.neighbourhood_key
WHERE f.price > 0
GROUP BY n.neighbourhood_name
ORDER BY listing_count DESC;
