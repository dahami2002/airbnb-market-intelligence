"""Generate rule-based business recommendations from analysis results."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import get_city_config, get_data_paths
from .utils import ensure_dir

logger = logging.getLogger("airbnb_pipeline.business_insights")


def generate_business_recommendations(
    enriched_df: pd.DataFrame,
    calendar_df: Optional[pd.DataFrame] = None,
    stats_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate rule-based business recommendations.

    Args:
        enriched_df: Cleaned and enriched listings DataFrame.
        calendar_df: Calendar DataFrame.
        stats_df: Statistical tests summary DataFrame (if available).

    Returns:
        DataFrame containing formatted business recommendations.
    """
    logger.info("Generating business recommendations based on rules...")
    recommendations: List[Dict[str, Any]] = []

    if enriched_df.empty:
        logger.warning("Enriched dataframe is empty. Cannot generate recommendations.")
        return pd.DataFrame()

    # Rule 1: Host Strategy
    if "is_professional_host" in enriched_df.columns:
        prof_share = enriched_df["is_professional_host"].mean()
        if prof_share > 0.3:
            recommendations.append(
                {
                    "theme": "Host Strategy",
                    "finding": f"{prof_share:.1%} of listings are managed by professional hosts (≥3 listings).",
                    "business_meaning": "The market is heavily professionalized. Solo hosts may struggle to compete on operational efficiency.",
                    "recommended_action": "Develop separate operational and pricing playbooks for commercial operators vs. casual hosts.",
                    "evidence_metric": f"Professional Host Share: {prof_share:.1%}",
                    "confidence_level": "High",
                    "limitation": "Threshold for 'professional' is strictly defined as ≥3 listings.",
                }
            )

    # Rule 2: Pricing Strategy
    if "room_type" in enriched_df.columns and "price" in enriched_df.columns:
        med_prices = enriched_df.groupby("room_type")["price"].median()
        entire_home = med_prices.get("Entire home/apt", 0)
        private_room = med_prices.get("Private room", 0)

        if entire_home > 0 and private_room > 0 and entire_home > (private_room * 1.5):
            recommendations.append(
                {
                    "theme": "Pricing",
                    "finding": f"Entire homes typically price at ฿{entire_home:,.0f}, while private rooms are ฿{private_room:,.0f}.",
                    "business_meaning": "There is a substantial premium for privacy and full-property access.",
                    "recommended_action": "Avoid using city-wide average pricing. Segment pricing strategies strictly by stay type.",
                    "evidence_metric": f"Entire vs Private gap: +{((entire_home-private_room)/private_room):.0%}",
                    "confidence_level": "High",
                    "limitation": "Medians used to reduce outlier impact.",
                }
            )

    # Rule 3: Neighbourhood Opportunity
    col_nbr = (
        "neighbourhood_cleansed"
        if "neighbourhood_cleansed" in enriched_df.columns
        else "neighbourhood"
    )
    if col_nbr in enriched_df.columns and "price" in enriched_df.columns:
        nbr_stats = enriched_df.groupby(col_nbr).agg(
            count=("id" if "id" in enriched_df.columns else col_nbr, "count"),
            median_price=("price", "median"),
        )
        high_price_threshold = nbr_stats["median_price"].quantile(0.8)
        low_supply_threshold = nbr_stats["count"].median()

        opportunity_zones = nbr_stats[
            (nbr_stats["median_price"] >= high_price_threshold)
            & (nbr_stats["count"] <= low_supply_threshold)
        ]

        if not opportunity_zones.empty:
            top_zone = opportunity_zones.sort_values(
                "median_price", ascending=False
            ).index[0]
            recommendations.append(
                {
                    "theme": "Neighbourhood Strategy",
                    "finding": f"'{top_zone}' has low supply but commands top-tier pricing.",
                    "business_meaning": "Premium, low-supply areas represent unfulfilled demand or high barrier-to-entry luxury markets.",
                    "recommended_action": "Investigate zoning and real estate availability in these areas for strategic investment.",
                    "evidence_metric": f"Opportunity Neighbourhoods: {len(opportunity_zones)} found",
                    "confidence_level": "Medium",
                    "limitation": "Low supply means small sample sizes; pricing medians may be volatile.",
                }
            )

    # Rule 4: Demand / Guest Experience
    if (
        "review_scores_rating" in enriched_df.columns
        and "availability_365" in enriched_df.columns
    ):
        med_score = enriched_df["review_scores_rating"].median()
        poor_exp_demand = enriched_df[
            (enriched_df["review_scores_rating"] < med_score)
            & (enriched_df["availability_365"] < 180)  # proxy for higher occupancy
        ]

        if len(poor_exp_demand) > 100:
            recommendations.append(
                {
                    "theme": "Demand / Reviews",
                    "finding": f"Found {len(poor_exp_demand):,} listings with below-average reviews but high implied occupancy.",
                    "business_meaning": "Location or price is overriding poor guest experience in these listings.",
                    "recommended_action": "Target these operators for property management services to immediately improve yield via better guest experience.",
                    "evidence_metric": f"Sub-par listings: {len(poor_exp_demand):,}",
                    "confidence_level": "Medium",
                    "limitation": "Availability is a proxy, not a direct booking metric.",
                }
            )

    # Rule 5: Seasonality (Weekend Premium)
    if (
        calendar_df is not None
        and not calendar_df.empty
        and "price" in calendar_df.columns
        and "date" in calendar_df.columns
    ):
        cal = calendar_df.copy()
        cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
        cal["is_weekend"] = cal["date"].dt.dayofweek.isin([4, 5])

        wd_prices = cal[~cal["is_weekend"]]["price"].median()
        we_prices = cal[cal["is_weekend"]]["price"].median()

        if we_prices > wd_prices:
            recommendations.append(
                {
                    "theme": "Seasonality",
                    "finding": f"Weekend prices (฿{we_prices:,.0f}) are higher than weekday prices (฿{wd_prices:,.0f}).",
                    "business_meaning": "The market demonstrates leisure-driven demand peaks on weekends.",
                    "recommended_action": "Implement dynamic weekend pricing premiums if not already active.",
                    "evidence_metric": f"Weekend Premium: +{((we_prices-wd_prices)/wd_prices):.1%}",
                    "confidence_level": "High",
                    "limitation": "Calendar prices reflect host intent, not necessarily realized bookings.",
                }
            )

    # Convert to DataFrame and format
    df_recs = pd.DataFrame(recommendations)
    if not df_recs.empty:
        df_recs.insert(0, "recommendation_id", range(1, len(df_recs) + 1))

    return df_recs


def run_business_insights(city_key: str) -> pd.DataFrame:
    """
    Load necessary data, generate business recommendations, and save to disk.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    processed_dir = paths["processed_dir"]
    reports_dir = ensure_dir(paths["reports_dir"])

    enriched_path = processed_dir / "enriched_listings.parquet"
    calendar_path = processed_dir / "clean_calendar.parquet"
    stats_path = reports_dir / "statistical_tests_summary.csv"

    if not enriched_path.exists():
        logger.error("Enriched listings not found. Cannot generate business insights.")
        return pd.DataFrame()

    enriched_df = pd.read_parquet(enriched_path)
    calendar_df = pd.read_parquet(calendar_path) if calendar_path.exists() else None
    stats_df = pd.read_csv(stats_path) if stats_path.exists() else None

    recs_df = generate_business_recommendations(enriched_df, calendar_df, stats_df)

    if not recs_df.empty:
        out_path = reports_dir / "business_recommendations.csv"
        recs_df.to_csv(out_path, index=False)
        logger.info("Saved %d business recommendations to %s", len(recs_df), out_path)

    return recs_df
