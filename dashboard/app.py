"""Interactive Streamlit dashboard for Airbnb market intelligence."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# CONFIGURATION & PAGE SETUP


st.set_page_config(
    page_title="Airbnb Market Intelligence - Bangkok",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Airbnb Market Intelligence Dashboard - Bangkok")
st.markdown(
    "An executive-friendly view of pricing, demand, supply, host behavior, "
    "and neighbourhood opportunity using Inside Airbnb data."
)

st.info(
    "Revenue and occupancy values are estimates based on public availability data. "
    "Unavailable dates may mean bookings or host-blocked dates, so they should be "
    "interpreted as directional signals, not exact financial results."
)


# HELPER FUNCTIONS & SAFE FORMATTING


def is_missing(value: Any) -> bool:
    try:
        if value is None:
            return True
        if pd.isna(value):
            return True
        if isinstance(value, (int, float, np.number)) and math.isinf(float(value)):
            return True
        return False
    except Exception:
        return True


def format_currency(value: Any) -> str:
    if is_missing(value):
        return "N/A"
    try:
        return f"฿{float(value):,.0f}"
    except Exception:
        return "N/A"


def format_number(value: Any) -> str:
    if is_missing(value):
        return "N/A"
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "N/A"


def format_percent(value: Any) -> str:
    if is_missing(value):
        return "N/A"
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "N/A"


def safe_median(series: pd.Series) -> Any:
    if series is None:
        return np.nan
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return numeric.median()


def safe_mean(series: pd.Series) -> Any:
    if series is None:
        return np.nan
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return numeric.mean()


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def bool_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])


def safe_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def render_insight_box(title: str, what: str, why: str, action: str) -> None:
    st.markdown(
        f"""
        <div style="background-color:#f8f9fa; padding:15px; border-left:5px solid #ff5a5f; border-radius:6px; margin-bottom:20px; color:#111827;">
            <h4 style="margin-top:0; color:#111827;">{title}</h4>
            <b>What this shows:</b> {what}<br>
            <b>Why it matters:</b> {why}<br>
            <b>Recommended action:</b> {action}
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_executive_summary(metrics: Dict[str, Any], recs_df: pd.DataFrame) -> str:
    total_l = format_number(metrics.get("total_listings", 0))
    med_p = format_currency(metrics.get("median_price", np.nan))
    prof_share = format_percent(metrics.get("professional_share", np.nan))

    summary = (
        f"**Market Overview:** The Bangkok market currently has **{total_l}** listings "
        f"in the selected view. The typical nightly price is **{med_p}**. "
        f"Professional operators manage **{prof_share}** of listings where this field is available.\n\n"
    )

    if recs_df is not None and not recs_df.empty:
        row = recs_df.iloc[0]
        action = row.get(
            "recommended_action",
            row.get("action", "Review the highest-impact market segment first."),
        )
        summary += f"**Key Recommended Action:** {action}"
    else:
        summary += (
            "**Key Recommended Action:** Use the pricing, neighbourhood, host, and demand tabs "
            "to identify where pricing can be improved and where operational risk is concentrated."
        )
    return summary


# DATA LOADING


@st.cache_data(show_spinner=True)
def load_data() -> Dict[str, pd.DataFrame]:
    project_root = Path(__file__).resolve().parent.parent
    warehouse_path = project_root / "data" / "warehouse" / "airbnb_bangkok.duckdb"
    processed_dir = project_root / "data" / "processed" / "bangkok"
    reports_dir = project_root / "reports"

    data = {
        "listings": pd.DataFrame(),
        "calendar": pd.DataFrame(),
        "recommendations": pd.DataFrame(),
        "statistics": pd.DataFrame(),
        "quality": pd.DataFrame(),
        "profiling": pd.DataFrame(),
        "model_results": pd.DataFrame(),
    }

    # Try loading from Parquet first (more reliable than DuckDB for version compatibility)
    listings_parquet = processed_dir / "enriched_listings.parquet"
    if listings_parquet.exists():
        try:
            data["listings"] = pd.read_parquet(listings_parquet)
        except Exception as exc:
            st.warning(f"Could not load listings parquet: {exc}")

    calendar_parquet = processed_dir / "clean_calendar.parquet"
    if calendar_parquet.exists():
        try:
            data["calendar"] = pd.read_parquet(
                calendar_parquet, columns=["date", "available", "price"]
            ).head(100_000)
        except Exception as exc:
            st.warning(f"Could not load calendar parquet: {exc}")

    # If Parquet failed, try DuckDB as fallback
    if data["listings"].empty and warehouse_path.exists():
        conn = None
        try:
            conn = duckdb.connect(str(warehouse_path), read_only=True)
            tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

            if "dim_listing" in tables and "fact_listing_snapshot" in tables:
                data["listings"] = conn.execute(
                    """
                    SELECT *
                    FROM dim_listing l
                    LEFT JOIN fact_listing_snapshot f
                    ON l.listing_key = f.listing_key
                    """
                ).df()
            elif "enriched_listings" in tables:
                data["listings"] = conn.execute("SELECT * FROM enriched_listings").df()

            if "fact_calendar" in tables:
                calendar_cols = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info('fact_calendar')"
                    ).fetchall()
                ]

                if "date" in calendar_cols:
                    date_select = "date"
                elif "date_key" in calendar_cols:
                    date_select = (
                        "STRPTIME(CAST(date_key AS VARCHAR), '%Y%m%d') AS date"
                    )
                else:
                    date_select = "NULL AS date"

                available_select = (
                    "available" if "available" in calendar_cols else "NULL AS available"
                )
                price_select = "price" if "price" in calendar_cols else "NULL AS price"
                raw_price_col = price_select.split()[0]

                data["calendar"] = conn.execute(
                    f"""
                    SELECT
                        {date_select},
                        {available_select},
                        {price_select}
                    FROM fact_calendar
                    WHERE {raw_price_col} IS NOT NULL
                    LIMIT 100000
                    """
                ).df()

            elif "clean_calendar" in tables:
                data["calendar"] = conn.execute(
                    """
                    SELECT date, available, price
                    FROM clean_calendar
                    WHERE price IS NOT NULL
                    LIMIT 100000
                    """
                ).df()

        except Exception as exc:
            # Silently fall back to Parquet (already loaded above)
            pass
        finally:
            if conn is not None:
                conn.close()

    # Load report CSV files safely
    report_files = {
        "recommendations": "business_recommendations.csv",
        "statistics": "statistical_tests_summary.csv",
        "quality": "data_quality_report.csv",
        "profiling": "profiling_summary.csv",
        "model_results": "model_results.csv",
    }
    for key, filename in report_files.items():
        path = reports_dir / filename
        if path.exists():
            try:
                data[key] = pd.read_csv(path)
            except Exception as exc:
                st.warning(f"Could not read {filename}: {exc}")
                data[key] = pd.DataFrame()

    return data


data = load_data()
df = data["listings"].copy()

if df.empty:
    st.error(
        "Processed data not found. Run: `python -m src.airbnb_pipeline.pipeline --city bangkok`"
    )
    st.stop()

# Normalize important columns for dashboard use.
price_col = first_existing_col(df, ["price", "listing_price", "nightly_price"])
if price_col:
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

nbr_col = first_existing_col(
    df, ["neighbourhood_cleansed", "neighbourhood", "neighborhood"]
)
room_col = first_existing_col(df, ["room_type", "stay_type"])
superhost_col = first_existing_col(df, ["host_is_superhost", "is_superhost"])
professional_col = first_existing_col(df, ["is_professional_host", "professional_host"])
host_id_col = first_existing_col(df, ["host_id", "host_key"])
rating_col = first_existing_col(
    df, ["review_scores_rating", "avg_review_score", "review_score_overall"]
)
reviews_col = first_existing_col(df, ["number_of_reviews", "reviews_count"])
lat_col = first_existing_col(df, ["latitude", "lat"])
lon_col = first_existing_col(df, ["longitude", "lon", "lng"])

# SIDEBAR & FILTERS


st.sidebar.header("Market Filters")

if st.sidebar.button("Reset Filters"):
    st.session_state.clear()
    st.rerun()

neighbourhoods = sorted(df[nbr_col].dropna().astype(str).unique()) if nbr_col else []
rooms = sorted(df[room_col].dropna().astype(str).unique()) if room_col else []

selected_nbrs = st.sidebar.multiselect(
    "Neighbourhood", options=neighbourhoods, default=[]
)
selected_rooms = st.sidebar.multiselect("Stay Type", options=rooms, default=[])

if price_col and df[price_col].notna().any():
    valid_prices = df[price_col].dropna()
    low_price = max(0, int(valid_prices.quantile(0.01)))
    high_price = max(low_price + 1, int(valid_prices.quantile(0.99)))
else:
    low_price, high_price = 0, 10000

price_range = st.sidebar.slider(
    "Nightly Price Range (THB)",
    min_value=0,
    max_value=high_price,
    value=(low_price, high_price),
)

selected_sh = st.sidebar.selectbox(
    "Superhost Status", ["All", "Superhost", "Regular Host"], index=0
)
selected_host = st.sidebar.selectbox(
    "Host Type", ["All", "Professional (>=3 listings)", "Casual (<3 listings)"], index=0
)

filtered_df = df.copy()

if selected_nbrs and nbr_col:
    filtered_df = filtered_df[filtered_df[nbr_col].astype(str).isin(selected_nbrs)]
if selected_rooms and room_col:
    filtered_df = filtered_df[filtered_df[room_col].astype(str).isin(selected_rooms)]
if price_col:
    filtered_df = filtered_df[
        (filtered_df[price_col] >= price_range[0])
        & (filtered_df[price_col] <= price_range[1])
    ]
if superhost_col:
    superhost_bool = bool_series(filtered_df[superhost_col])
    if selected_sh == "Superhost":
        filtered_df = filtered_df[superhost_bool]
    elif selected_sh == "Regular Host":
        filtered_df = filtered_df[~superhost_bool]
if professional_col:
    prof_bool = bool_series(filtered_df[professional_col])
    if selected_host == "Professional (>=3 listings)":
        filtered_df = filtered_df[prof_bool]
    elif selected_host == "Casual (<3 listings)":
        filtered_df = filtered_df[~prof_bool]

st.sidebar.markdown("---")
st.sidebar.write(
    f"Showing **{len(filtered_df):,}** listings out of **{len(df):,}** total listings."
)
st.sidebar.download_button(
    label="Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_airbnb_data.csv",
    mime="text/csv",
)

if filtered_df.empty:
    st.warning(
        "No listings match this combination. Try removing one filter or click Reset Filters. "
        "Showing the unfiltered market overview below."
    )
    filtered_df = df.copy()


# TABS SETUP


tabs = st.tabs(
    [
        "1. Executive Summary",
        "2. Market Overview",
        "3. Pricing Signals",
        "4. Neighbourhoods",
        "5. Host Strategy",
        "6. Demand & Guest Exp.",
        "7. Seasonality",
        "8. Statistical Evidence",
        "9. Data Quality",
        "10. Methodology",
        "11. ML Results",
    ]
)


# TAB 1: EXECUTIVE SUMMARY

with tabs[0]:
    metrics = {
        "total_listings": len(filtered_df),
        "median_price": safe_median(filtered_df[price_col]) if price_col else np.nan,
        "professional_share": (
            bool_series(filtered_df[professional_col]).mean()
            if professional_col
            else np.nan
        ),
        "avg_review": safe_mean(filtered_df[rating_col]) if rating_col else np.nan,
    }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", format_number(metrics["total_listings"]))
    col2.metric("Typical Nightly Price", format_currency(metrics["median_price"]))
    col3.metric(
        "Professional Host Share", format_percent(metrics["professional_share"])
    )
    col4.metric(
        "Avg Review Score",
        (
            f"{metrics['avg_review']:.2f}"
            if not is_missing(metrics["avg_review"])
            else "N/A"
        ),
    )

    st.markdown(" Rule-Based Market Summary")
    st.write(generate_executive_summary(metrics, data["recommendations"]))

    st.markdown("Executive Decision Cards")
    recs_df = data["recommendations"]
    if not recs_df.empty:
        for idx, row in recs_df.iterrows():
            theme = row.get("theme", "Business")
            confidence = row.get("confidence_level", "Medium")
            with st.expander(
                f"{theme} Decision: {confidence} Confidence", expanded=(idx == 0)
            ):
                st.write(f"Finding: {row.get('finding', 'N/A')}")
                st.write(f"Business Meaning: {row.get('business_meaning', 'N/A')}")
                st.write(f"Action: {row.get('recommended_action', 'N/A')}")
                st.caption(
                    f"Evidence: {row.get('evidence_metric', 'N/A')} | "
                    f"Limitation: {row.get('limitation', 'N/A')}"
                )
    else:
        st.info("Run the pipeline to generate decision cards.")


# TAB 2: MARKET OVERVIEW

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        if room_col:
            st.subheader("Listings by Stay Type")
            room_counts = filtered_df[room_col].value_counts().reset_index()
            room_counts.columns = ["Stay Type", "Count"]
            fig1 = px.pie(room_counts, values="Count", names="Stay Type", hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
            render_insight_box(
                "Stay Type Distribution",
                "The chart shows how the market is split across entire homes, private rooms, shared rooms, and hotel-style listings.",
                "This determines what kind of supply dominates the market and which customer segments are most important.",
                "Build separate pricing and quality benchmarks for each stay type.",
            )
        else:
            st.info("Stay type column not available.")

    with col2:
        if price_col:
            st.subheader("Price Distribution")
            plot_df = filtered_df[filtered_df[price_col].between(0, high_price)]
            fig2 = px.histogram(plot_df, x=price_col, nbins=50)
            fig2.update_layout(
                xaxis_title="Nightly Price (THB)", yaxis_title="Listing Count"
            )
            st.plotly_chart(fig2, use_container_width=True)
            render_insight_box(
                "Price Concentration",
                "Most listings are concentrated in the lower and middle price bands, while a small number of listings are much more expensive.",
                "Average price can be misleading when a market has extreme luxury listings.",
                "Use median price as the main market benchmark for executive decisions.",
            )
        else:
            st.info("Price column not available.")


# TAB 3: PRICING SIGNALS

with tabs[2]:
    if room_col and price_col:
        st.subheader("Typical Price by Stay Type")
        med_prices = (
            filtered_df.groupby(room_col, dropna=False)[price_col]
            .median()
            .reset_index()
        )
        fig3 = px.bar(med_prices, x=room_col, y=price_col, text_auto=".0f")
        fig3.update_layout(
            xaxis_title="Stay Type", yaxis_title="Typical Nightly Price (THB)"
        )
        st.plotly_chart(fig3, use_container_width=True)
        render_insight_box(
            "Premium Pricing",
            "Different stay types have different typical prices.",
            "Guests often pay more for privacy, location, and full property access.",
            "Avoid one city wide pricing rule; benchmark by stay type and neighbourhood.",
        )
    else:
        st.info("Pricing analysis requires price and stay type columns.")


# TAB 4: NEIGHBOURHOOD OPPORTUNITY

with tabs[3]:
    if nbr_col and price_col:
        nbr_stats = (
            filtered_df.groupby(nbr_col, dropna=False)
            .agg(
                listing_count=(price_col, "count"), typical_price=(price_col, "median")
            )
            .reset_index()
        )
        st.subheader("Supply vs Price by Neighbourhood")
        fig4 = px.scatter(
            nbr_stats,
            x="listing_count",
            y="typical_price",
            hover_name=nbr_col,
            size="listing_count",
            color="typical_price",
        )
        fig4.update_layout(
            xaxis_title="Total Listings", yaxis_title="Typical Nightly Price (THB)"
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(
            nbr_stats.sort_values("listing_count", ascending=False).head(20),
            use_container_width=True,
        )
        render_insight_box(
            "Neighbourhood Opportunity Matrix",
            "Neighbourhoods differ by supply volume and typical price.",
            "High-price, low supply areas may indicate premium opportunity; high supply areas may require stronger competition management.",
            "Prioritize neighbourhood level decisions rather than city-wide averages.",
        )

        if lat_col and lon_col:
            st.subheader("Geographic Price Distribution")
            map_df = filtered_df.dropna(subset=[lat_col, lon_col]).head(5000)
            if not map_df.empty:
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat=lat_col,
                    lon=lon_col,
                    color=price_col,
                    color_continuous_scale=px.colors.sequential.YlOrRd,
                    range_color=[0, high_price],
                    mapbox_style="carto-positron",
                    zoom=10,
                    hover_name=nbr_col,
                )
                st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Neighbourhood analysis requires neighbourhood and price columns.")


# TAB 5: HOST STRATEGY

with tabs[4]:
    if professional_col:
        prof_label = bool_series(filtered_df[professional_col]).map(
            {True: "Professional", False: "Casual"}
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Host Professionalization")
            prof_counts = prof_label.value_counts().reset_index()
            prof_counts.columns = ["Host Segment", "Count"]
            fig5 = px.pie(prof_counts, values="Count", names="Host Segment", hole=0.5)
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            if price_col:
                st.subheader("Price by Host Type")
                temp = filtered_df.copy()
                temp["Host Segment"] = prof_label
                host_prices = (
                    temp.groupby("Host Segment")[price_col].median().reset_index()
                )
                fig6 = px.bar(
                    host_prices, x="Host Segment", y=price_col, text_auto=".0f"
                )
                fig6.update_layout(yaxis_title="Typical Nightly Price (THB)")
                st.plotly_chart(fig6, use_container_width=True)
        render_insight_box(
            "Commercial Operators",
            "The chart separates casual hosts from professional operators.",
            "Professional hosts may behave more like businesses, with different pricing and availability strategies.",
            "Manage professional hosts as a distinct B2B segment rather than treating all hosts the same.",
        )
    else:
        st.info("Professional host field is not available.")


# TAB 6: DEMAND & GUEST EXPERIENCE

with tabs[5]:
    if rating_col:
        st.subheader("Guest Satisfaction Distribution")
        fig7 = px.histogram(filtered_df, x=rating_col, nbins=20)
        fig7.update_layout(xaxis_title="Overall Rating", yaxis_title="Listing Count")
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("Operational Improvement Targets")
        st.write("Listings with high review volume but below-market review scores.")
        if reviews_col:
            med_score = safe_median(filtered_df[rating_col])
            if not is_missing(med_score):
                targets = (
                    filtered_df[
                        (
                            pd.to_numeric(filtered_df[rating_col], errors="coerce")
                            < med_score
                        )
                        & (
                            pd.to_numeric(filtered_df[reviews_col], errors="coerce")
                            > 10
                        )
                    ]
                    .sort_values(reviews_col, ascending=False)
                    .head(10)
                )
                display_cols = safe_columns(
                    targets,
                    [nbr_col, room_col, price_col, rating_col, reviews_col],
                )
                if display_cols:
                    st.dataframe(targets[display_cols], use_container_width=True)
                else:
                    st.info(
                        "No suitable display columns are available for improvement targets."
                    )
        render_insight_box(
            "Experience Gaps",
            "Some listings may have strong demand signals but below-market review scores.",
            "Improving operational quality can protect revenue and reputation.",
            "Target cleaning, communication, check-in, and value improvements for high-volume low-score listings.",
        )
    else:
        st.info("Review score data is not available.")


# TAB 7: SEASONALITY & CALENDAR

with tabs[6]:
    cal_df = data["calendar"].copy()
    cal_price_col = first_existing_col(
        cal_df, ["price", "calendar_price", "adjusted_price"]
    )
    if not cal_df.empty and "date" in cal_df.columns and cal_price_col:
        cal_df["date"] = pd.to_datetime(cal_df["date"], errors="coerce")
        cal_df[cal_price_col] = pd.to_numeric(cal_df[cal_price_col], errors="coerce")
        cal_df = cal_df.dropna(subset=["date", cal_price_col])
        cal_df["month"] = cal_df["date"].dt.to_period("M").astype(str)
        cal_df["is_weekend"] = cal_df["date"].dt.dayofweek.isin([4, 5])

        st.subheader("Weekend vs Weekday Pricing")
        wknd_price = cal_df.groupby("is_weekend")[cal_price_col].median().reset_index()
        wknd_price["Day Type"] = wknd_price["is_weekend"].map(
            {True: "Weekend (Fri-Sat)", False: "Weekday (Sun-Thu)"}
        )
        fig8 = px.bar(wknd_price, x="Day Type", y=cal_price_col, text_auto=".0f")
        fig8.update_layout(yaxis_title="Typical Nightly Price (THB)")
        st.plotly_chart(fig8, use_container_width=True)

        st.subheader("Monthly Calendar Price Signal")
        monthly = cal_df.groupby("month")[cal_price_col].median().reset_index()
        fig9 = px.line(monthly, x="month", y=cal_price_col, markers=True)
        fig9.update_layout(
            xaxis_title="Month", yaxis_title="Typical Calendar Price (THB)"
        )
        st.plotly_chart(fig9, use_container_width=True)

        render_insight_box(
            "Seasonality and Weekend Pricing",
            "Calendar data shows how hosts intend to price across dates and day types.",
            "It helps identify whether the market has leisure/weekend demand or seasonal pricing pressure.",
            "Use dynamic pricing only where the data shows a meaningful premium.",
        )
    else:
        st.info("Calendar data not available for seasonality analysis.")


# TAB 8: STATISTICAL EVIDENCE

with tabs[7]:
    stats_df = data["statistics"]
    if not stats_df.empty:
        st.subheader("Hypothesis Testing Results")
        st.markdown(
            "Formal statistical testing validates visual observations. Do not interpret these tests as causal proof."
        )
        for _, row in stats_df.iterrows():
            question = row.get(
                "business_question", row.get("hypothesis", "Business question")
            )
            result = row.get(
                "result_plain_english", row.get("conclusion", "Result available")
            )
            with st.expander(f"Q: {question} | Result: {result}"):
                st.write(f"Test Used:{row.get('test_used', 'N/A')}")
                st.write(
                    f"P-Value: {row.get('p_value', 'N/A')} | Effect Size: {row.get('effect_size', 'N/A')}"
                )
                st.write(
                    f"Business Meaning:{row.get('business_interpretation', 'N/A')}"
                )
                st.caption(f"Limitation: {row.get('limitation', 'N/A')}")
    else:
        st.info("Run the pipeline to generate statistical tests.")


# TAB 9: DATA QUALITY & TRUST

with tabs[8]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Quality Report")
        qual_df = data.get("quality", pd.DataFrame())
        if qual_df is not None and not qual_df.empty:
            if "status" in qual_df.columns:
                failed = qual_df[qual_df["status"].astype(str).str.upper() == "FAIL"]
                st.metric("Failed Quality Checks", len(failed))
            else:
                st.metric("Failed Quality Checks", "N/A")
                st.warning("The quality report does not contain a status column.")
            st.dataframe(qual_df, use_container_width=True)
        else:
            st.info(
                "Quality report missing. Run: python -m src.airbnb_pipeline.pipeline --city bangkok"
            )

    with col2:
        st.subheader("Data Profiling Summary")
        prof_df = data.get("profiling", pd.DataFrame())
        profile_display_cols = [
            "file_key",
            "file_name",
            "column",
            "column_name",
            "file",
            "table_name",
            "dtype",
            "data_type",
            "null_pct",
            "missing_pct",
            "missing_percentage",
            "unique_count",
            "n_unique",
            "duplicate_rows",
            "min_val",
            "min_value",
            "max_val",
            "max_value",
        ]
        if prof_df is None or prof_df.empty:
            st.info(
                "Profiling report missing or empty. Run: python -m src.airbnb_pipeline.pipeline --city bangkok"
            )
        else:
            available_profile_cols = [
                col for col in profile_display_cols if col in prof_df.columns
            ]
            if available_profile_cols:
                st.dataframe(
                    prof_df[available_profile_cols].head(100), use_container_width=True
                )
            else:
                st.warning(
                    "Profiling summary file exists, but expected display columns were not found."
                )
                st.write("Available columns:")
                st.write(list(prof_df.columns))
                st.dataframe(prof_df.head(50), use_container_width=True)

    st.markdown(
        "These checks help stakeholders trust the dashboard because prices, dates, IDs, and coordinates were validated before analysis."
    )


# TAB 10: METHODOLOGY

with tabs[9]:
    st.subheader("Methodology & Limitations")
    st.markdown(
        """
        Dataset Source: Inside Airbnb
        City: Bangkok
        Snapshot Date: 2025-09-26

        Key Assumptions
        1. Prices are in THB.
        2. Revenue is a proxy.Calendar unavailable dates may mean a booking or a host-blocked date.
        3. Professional threshold. A professional host is defined as having 3 or more listings.
        4. Outliers.Extreme prices are kept in the data but charts use a high percentile cap for readability.
        5. Snapshot limitation.Data represents one specific point in time.
        """
    )


# TAB 11: ML PROTOTYPE RESULTS

with tabs[10]:
    st.subheader("ML Prototype Results")

    st.markdown(
        """
        This section shows the optional machine learning prototype for listing price prediction.
        The models are baseline experiments only. They should not be treated as production pricing systems.
        """
    )

    mod_df = data.get("model_results", pd.DataFrame())

    if mod_df is not None and not mod_df.empty:
        st.dataframe(mod_df, use_container_width=True)

        if "model" in mod_df.columns and "r2_score" in mod_df.columns:
            score_df = mod_df.copy()
            score_df["r2_score"] = pd.to_numeric(score_df["r2_score"], errors="coerce")
            if score_df["r2_score"].notna().any():
                best_model = score_df.sort_values("r2_score", ascending=False).iloc[0]
                st.success(
                    f"Best model by R2: {best_model['model']} "
                    f"with R2 = {best_model['r2_score']:.3f}"
                )

        if "model" in mod_df.columns:
            xgb_rows = mod_df[
                mod_df["model"]
                .astype(str)
                .str.contains("XGBoost", case=False, na=False)
            ]
            if not xgb_rows.empty:
                st.markdown("### XGBoost Result")
                st.dataframe(xgb_rows, use_container_width=True)
            else:
                st.info(
                    "XGBoost result not found. Install xgboost and run the ML pipeline."
                )
    else:
        st.info(
            "Optional ML prototype not yet generated. Run: "
            "python -m src.airbnb_pipeline.pipeline --city bangkok --include-ml"
        )
