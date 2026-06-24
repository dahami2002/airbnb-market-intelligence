"""Exploratory data analysis - generates charts and insights."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import get_city_config, get_data_paths
from .utils import ensure_dir

logger = logging.getLogger("airbnb_pipeline.analysis")

# Plot styling
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def run_analysis(city_key: str) -> pd.DataFrame:
    """
    Generate all EDA charts and save insights summary.

    Args:
        city_key: City identifier.

    Returns:
        DataFrame of EDA insights.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    processed_dir = paths["processed_dir"]
    figures_dir = ensure_dir(paths["figures_dir"])
    reports_dir = ensure_dir(paths["reports_dir"])

    logger.info("Starting EDA for %s", city_cfg["display_name"])

    # Load data
    enriched_path = processed_dir / "enriched_listings.parquet"
    if not enriched_path.exists():
        logger.error("Enriched listings not found. Run enrichment first.")
        return pd.DataFrame()

    df = pd.read_parquet(enriched_path)
    logger.info(
        "Loaded enriched listings: %d rows, %d columns", len(df), len(df.columns)
    )

    cal_path = processed_dir / "clean_calendar.parquet"
    df_cal = pd.read_parquet(cal_path) if cal_path.exists() else None

    insights: List[Dict[str, Any]] = []

    #  Chart 1: Price Distribution
    insights.append(_chart_price_distribution(df, figures_dir))

    # Chart 2: Median Price by Room Type
    insights.append(_chart_price_by_room_type(df, figures_dir))

    #  Chart 3: Top Neighbourhoods by Median Price
    insights.append(_chart_top_neighbourhoods_price(df, figures_dir))

    # Chart 4: Listings by Neighbourhood
    insights.append(_chart_listings_by_neighbourhood(df, figures_dir))

    # Chart 5: Host Listing Count Distribution
    insights.append(_chart_host_listing_distribution(df, figures_dir))

    # Chart 6: Professional vs Casual Host
    insights.append(_chart_professional_vs_casual(df, figures_dir))

    # Chart 7: Review Score Distribution
    insights.append(_chart_review_score_distribution(df, figures_dir))

    # Chart 8: Availability by Month
    if df_cal is not None:
        insights.append(_chart_availability_by_month(df_cal, figures_dir))

    # Chart 9: Weekend vs Weekday Price
    if df_cal is not None:
        insights.append(_chart_weekend_vs_weekday(df_cal, figures_dir))

    # Save insights
    insights_df = pd.DataFrame(insights)
    insights_df.insert(0, "insight_id", range(1, len(insights_df) + 1))
    out_path = reports_dir / "eda_insights_summary.csv"
    insights_df.to_csv(out_path, index=False)
    logger.info("EDA insights saved to %s (%d insights)", out_path, len(insights_df))

    return insights_df


def _chart_price_distribution(df: pd.DataFrame, figures_dir: Path) -> Dict[str, str]:
    """Histogram of listing prices."""
    prices = df["price"].dropna()
    prices = prices[
        (prices > 0) & (prices < prices.quantile(0.99))
    ]  # Trim extreme outliers

    fig, ax = plt.subplots()
    ax.hist(prices, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_title("Distribution of Listing Prices (Bangkok)")
    ax.set_xlabel("Price (THB)")
    ax.set_ylabel("Number of Listings")
    ax.axvline(
        prices.median(),
        color="red",
        linestyle="--",
        label=f"Median: {prices.median():,.0f}",
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "price_distribution.png")
    plt.close(fig)

    return {
        "chart_name": "price_distribution",
        "finding": f"Median price is {prices.median():,.0f} THB. Distribution is right-skewed with {len(prices):,} listings.",
        "business_interpretation": "The market is dominated by affordable listings, with a long tail of luxury properties.",
        "limitation": "Top 1% outliers excluded for visualisation. Currency is Thai Baht (THB).",
    }


def _chart_price_by_room_type(df: pd.DataFrame, figures_dir: Path) -> Dict[str, str]:
    """Bar chart of median price by room type."""
    rt = df.groupby("room_type")["price"].median().dropna().sort_values(ascending=False)

    fig, ax = plt.subplots()
    bars = ax.barh(rt.index, rt.values, color="#55A868", edgecolor="white")
    ax.set_title("Median Price by Room Type")
    ax.set_xlabel("Median Price (THB)")
    for bar, val in zip(bars, rt.values):
        ax.text(
            val + rt.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(figures_dir / "price_by_room_type.png")
    plt.close(fig)

    top = rt.index[0] if len(rt) > 0 else "N/A"
    return {
        "chart_name": "price_by_room_type",
        "finding": f"'{top}' has the highest median price. {len(rt)} room types observed.",
        "business_interpretation": "Entire homes/apartments command premium pricing. Shared/private rooms serve the budget segment.",
        "limitation": "Median used to reduce impact of extreme values.",
    }


def _chart_top_neighbourhoods_price(
    df: pd.DataFrame, figures_dir: Path
) -> Dict[str, str]:
    """Top 15 neighbourhoods by median price."""
    col = (
        "neighbourhood_cleansed"
        if "neighbourhood_cleansed" in df.columns
        else "neighbourhood"
    )
    if col not in df.columns:
        return {
            "chart_name": "top_neighbourhoods_price",
            "finding": "Column not found",
            "business_interpretation": "N/A",
            "limitation": "Missing data",
        }

    nbr = (
        df.groupby(col)["price"].median().dropna().sort_values(ascending=False).head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(nbr.index[::-1], nbr.values[::-1], color="#C44E52", edgecolor="white")
    ax.set_title("Top 15 Neighbourhoods by Median Price")
    ax.set_xlabel("Median Price (THB)")
    plt.tight_layout()
    fig.savefig(figures_dir / "top_neighbourhoods_price.png")
    plt.close(fig)

    return {
        "chart_name": "top_neighbourhoods_price",
        "finding": f"Top neighbourhood: {nbr.index[0]} (median {nbr.iloc[0]:,.0f} THB).",
        "business_interpretation": "Premium neighbourhoods likely correlate with tourist/CBD areas.",
        "limitation": "Only top 15 shown. Small-sample neighbourhoods may have unreliable medians.",
    }


def _chart_listings_by_neighbourhood(
    df: pd.DataFrame, figures_dir: Path
) -> Dict[str, str]:
    """Bar chart of listing counts by neighbourhood (top 15)."""
    col = (
        "neighbourhood_cleansed"
        if "neighbourhood_cleansed" in df.columns
        else "neighbourhood"
    )
    if col not in df.columns:
        return {
            "chart_name": "listings_by_neighbourhood",
            "finding": "Column not found",
            "business_interpretation": "N/A",
            "limitation": "Missing data",
        }

    counts = df[col].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#8172B2", edgecolor="white")
    ax.set_title("Top 15 Neighbourhoods by Listing Count")
    ax.set_xlabel("Number of Listings")
    plt.tight_layout()
    fig.savefig(figures_dir / "listings_by_neighbourhood.png")
    plt.close(fig)

    return {
        "chart_name": "listings_by_neighbourhood",
        "finding": f"Most listings in {counts.index[0]} ({counts.iloc[0]:,} listings).",
        "business_interpretation": "High-density areas indicate popular tourist or business districts.",
        "limitation": "Only top 15 shown.",
    }


def _chart_host_listing_distribution(
    df: pd.DataFrame, figures_dir: Path
) -> Dict[str, str]:
    """Distribution of listings per host."""
    col = "calculated_host_listings_count"
    if col not in df.columns:
        return {
            "chart_name": "host_listing_distribution",
            "finding": "Column not found",
            "business_interpretation": "N/A",
            "limitation": "Missing data",
        }

    counts = df[col].dropna()

    fig, ax = plt.subplots()
    ax.hist(
        counts.clip(upper=20), bins=20, color="#CCB974", edgecolor="white", alpha=0.8
    )
    ax.set_title("Host Listing Count Distribution")
    ax.set_xlabel("Number of Listings per Host")
    ax.set_ylabel("Number of Hosts")
    ax.axvline(3, color="red", linestyle="--", label="Professional threshold (≥3)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "host_listing_distribution.png")
    plt.close(fig)

    pct_pro = (counts >= 3).mean() * 100
    return {
        "chart_name": "host_listing_distribution",
        "finding": f"{pct_pro:.1f}% of listings belong to professional hosts (≥3 listings).",
        "business_interpretation": "Indicates the degree of market professionalisation.",
        "limitation": "Threshold of ≥3 is arbitrary but commonly used in Airbnb research.",
    }


def _chart_professional_vs_casual(
    df: pd.DataFrame, figures_dir: Path
) -> Dict[str, str]:
    """Compare professional vs casual hosts on key metrics."""
    if "is_professional_host" not in df.columns:
        return {
            "chart_name": "professional_vs_casual",
            "finding": "Column not found",
            "business_interpretation": "N/A",
            "limitation": "Missing data",
        }

    groups = (
        df.groupby("is_professional_host")
        .agg(
            median_price=("price", "median"),
            avg_review_score=("review_scores_rating", "mean"),
            count=("id" if "id" in df.columns else df.columns[0], "count"),
        )
        .round(2)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Casual", "Professional"]

    if len(groups) == 2:
        axes[0].bar(labels, groups["median_price"].values, color=["#4C72B0", "#C44E52"])
        axes[0].set_title("Median Price")
        axes[0].set_ylabel("THB")

        axes[1].bar(
            labels, groups["avg_review_score"].values, color=["#4C72B0", "#C44E52"]
        )
        axes[1].set_title("Avg Review Score")
        axes[1].set_ylabel("Score")

    fig.suptitle("Professional vs Casual Host Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / "professional_vs_casual.png")
    plt.close(fig)

    return {
        "chart_name": "professional_vs_casual",
        "finding": f"Professional hosts: {groups['count'].iloc[-1]:,} listings. Casual: {groups['count'].iloc[0]:,} listings.",
        "business_interpretation": "Professional hosts may operate differently in pricing and service quality.",
        "limitation": "Professional defined as ≥3 listings per host. Does not account for multi-platform hosts.",
    }


def _chart_review_score_distribution(
    df: pd.DataFrame, figures_dir: Path
) -> Dict[str, str]:
    """Histogram of review_scores_rating."""
    col = "review_scores_rating"
    if col not in df.columns:
        return {
            "chart_name": "review_score_distribution",
            "finding": "Column not found",
            "business_interpretation": "N/A",
            "limitation": "Missing data",
        }

    scores = df[col].dropna()

    fig, ax = plt.subplots()
    ax.hist(scores, bins=30, color="#64B5CD", edgecolor="white", alpha=0.8)
    ax.set_title("Distribution of Review Scores (Rating)")
    ax.set_xlabel("Review Score")
    ax.set_ylabel("Number of Listings")
    ax.axvline(
        scores.median(),
        color="red",
        linestyle="--",
        label=f"Median: {scores.median():.2f}",
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "review_score_distribution.png")
    plt.close(fig)

    return {
        "chart_name": "review_score_distribution",
        "finding": f"Median review score: {scores.median():.2f}. {len(scores):,} listings with reviews.",
        "business_interpretation": "Scores are typically left-skewed (most hosts rated highly), reflecting selection bias in reviews.",
        "limitation": "Only listings with reviews included. Listings without reviews are excluded.",
    }


def _chart_availability_by_month(
    df_cal: pd.DataFrame, figures_dir: Path
) -> Dict[str, str]:
    """Monthly average availability rate from calendar data."""
    df_cal["date"] = pd.to_datetime(df_cal["date"], errors="coerce")
    df_cal["month"] = df_cal["date"].dt.to_period("M")

    monthly = (
        df_cal.groupby("month")
        .agg(available_pct=("available", lambda x: x.astype(bool).mean() * 100))
        .reset_index()
    )
    monthly["month_str"] = monthly["month"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        monthly["month_str"],
        monthly["available_pct"],
        marker="o",
        color="#4C72B0",
        linewidth=2,
    )
    ax.set_title("Average Availability Rate by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Availability (%)")
    ax.set_xticklabels(monthly["month_str"], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / "availability_by_month.png")
    plt.close(fig)

    return {
        "chart_name": "availability_by_month",
        "finding": f"Availability ranges from {monthly['available_pct'].min():.1f}% to {monthly['available_pct'].max():.1f}%.",
        "business_interpretation": "Lower availability months may indicate higher demand or host blocking.",
        "limitation": "available=False may mean host-blocked, not necessarily booked.",
    }


def _chart_weekend_vs_weekday(
    df_cal: pd.DataFrame, figures_dir: Path
) -> Dict[str, str]:
    """Compare weekend vs weekday pricing from calendar."""
    df_cal["date"] = pd.to_datetime(df_cal["date"], errors="coerce")
    df_cal["is_weekend"] = df_cal["date"].dt.dayofweek.isin([4, 5])  # Fri, Sat

    prices = df_cal[df_cal["price"] > 0].groupby("is_weekend")["price"].median()

    labels = ["Weekday", "Weekend"]
    values = [
        prices.get(False, 0),
        prices.get(True, 0),
    ]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["#4C72B0", "#C44E52"], edgecolor="white")
    ax.set_title("Weekend vs Weekday Median Calendar Price")
    ax.set_ylabel("Median Price (THB)")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:,.0f}", ha="center", fontsize=12)
    plt.tight_layout()
    fig.savefig(figures_dir / "weekend_vs_weekday_price.png")
    plt.close(fig)

    diff_pct = ((values[1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
    return {
        "chart_name": "weekend_vs_weekday_price",
        "finding": f"Weekend median: {values[1]:,.0f} THB, Weekday: {values[0]:,.0f} THB ({diff_pct:+.1f}% difference).",
        "business_interpretation": "Weekend premiums suggest leisure-driven demand.",
        "limitation": "Weekend defined as Friday–Saturday nights. Definition may vary by market.",
    }
