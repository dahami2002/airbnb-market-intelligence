# Airbnb Market Intelligence Data Pipeline: Bangkok Case Study

Role: Data Engineering Intern Technical Assessment
Dataset: Inside Airbnb — Bangkok, Thailand
Snapshot Date:2025-09-26
Pipeline Version: 1.0.0

#Executive Overview

This project demonstrates a production style, end-to-end data engineering pipeline built on public Inside Airbnb data for Bangkok, Thailand. Starting from raw compressed CSV files, the pipeline ingests, profiles, cleans, validates, enriches, and loads data into a DuckDB analytical warehouse modelled as a Star Schema. It then executes formal statistical hypothesis tests, trains a baseline ML price model (Linear Regression, Random Forest, XGBoost), generates rule-based business recommendations, and surfaces all findings through an interactive 11-tab Streamlit executive dashboard.

The project is designed to demonstrate the core competencies expected of a Data Engineering Intern: pipeline design, data quality thinking, SQL modelling, analytical rigour, and the ability to translate technical outputs into business language.

# Business Problem

Short term rental operators, property investors, and hospitality consultants operating in Bangkok need reliable, structured intelligence on:

- Where the market is priced and why
- Which neighbourhoods offer the strongest pricing leverage
- Whether platform signals (Superhost badge, host professionalization) correlate with measurable outcomes
- How demand patterns vary across seasons, weekends, and property types

Inside Airbnb provides public data, but it arrives as raw CSV files with mixed types, currency strings, and missing values. This pipeline transforms that raw data into a trusted, queryable analytical asset.

#_Why Bangkok?_

Bangkok was selected for four reasons:

1. It is one of Southeast Asia's highest traffic tourism destinations, making the short-term rental market commercially relevant.
2. The dataset offers a good volume balance — large enough to produce statistically meaningful results, small enough to run on a single machine.
3. As a non-Western market priced in Thai Baht (THB), it provides differentiated insights compared to the commonly analysed New York or London datasets.
4. The market shows clear segmentation across property types, neighbourhoods, and host professionalization levels, producing rich analytical signal.

Full rationale is documented in `reports/decision_log.md`.

#Dataset Summary



Source - [Inside Airbnb](http://insideairbnb.com/get-the-data/) |
 City -           Bangkok, Thailand
Snapshot Date -   2025-09-26
Listings (raw) -  28,806 rows, 79 columns
Calendar records -10,514,202 rows
Reviews          -583,333
Neighbourhoods -  50 areas

# Repository Structure

Airbnb Market Intelligence/
│
├── config/
│   └── cities.yml                  # City configuration (URL, snapshot date, file mappings)
│
├── dashboard/
│   └── app.py                      # 11-tab Streamlit executive dashboard
│
├── data/
│   ├── raw/bangkok/2025-09-26/     # Raw compressed CSVs downloaded at ingest time
│   ├── processed/bangkok/          # Cleaned and enriched Parquet files
│   └── warehouse/
│       └── airbnb_bangkok.duckdb   # DuckDB analytical warehouse (Star Schema)
│
├── reports/
│   ├── figures/                    # 9 EDA charts (PNG)
│   ├── business_recommendations.csv
│   ├── data_quality_report.csv
│   ├── decision_log.md
│   ├── eda_insights_summary.csv
│   ├── final_report_template.md
│   ├── model_feature_importance.csv
│   ├── model_results.csv
│   ├── profiling_summary.csv
│   └── statistical_tests_summary.csv
│
├── sql/
│   ├── create_schema.sql           # Star Schema DDL (DuckDB)
│   ├── analytical_queries.sql      # Business intelligence SQL queries
│   └── quality_checks.sql         # SQL-level data quality checks
│
├── src/airbnb_pipeline/
│   ├── pipeline.py                 # CLI orchestrator — runs all stages in order
│   ├── ingest.py                   # Downloads raw files from Inside Airbnb URLs
│   ├── profile.py                  # Column-level data profiling
│   ├── clean.py                    # Parsing, type conversion, standardisation
│   ├── quality.py                  # Custom data quality check framework
│   ├── enrich.py                   # Derived fields and calendar
│   ├── model.py                    # DuckDB warehouse loader (Star Schema population)

│   ├── analysis.py                 # EDA charts and insights summary
│   ├── statistics.py               # Formal statistical hypothesis tests
│   ├── business_insights.py        # Rule-based business recommendation engine
│   ├── ml_price_model.py           # Baseline ML price prediction prototype
│   ├── config.py                   # Config loader (reads cities.yml)
│   └── utils.py                    # Shared utilities (logging, file helpers)
│
├── tests/
│   ├── test_cleaning.py            # Unit tests for clean.py helpers
│   └── test_quality.py             # Unit tests for quality check logic
│
├── .env.example                    # Environment variable template
├── requirements.txt                # Pinned Python dependencies
├── setup_env.ps1                   # Creates virtual environment and installs packages
├── run_all.ps1                     # Runs full pipeline + tests end-to-end
└── run_dashboard.ps1               # Launches Streamlit dashboard
```

**📊 Architecture Diagrams:** Visual representations available in `docs/architecture_diagrams.md`:
- Pipeline architecture (data flow from raw CSVs to dashboard)
- DuckDB Star Schema (fact and dimension tables)
- Dashboard structure (11 interactive tabs)
- Technology stack layers

## What I Built

1. Modular ETL Pipeline

A CLI driven pipeline with 10 discrete stages executed in order via `pipeline.py`. Stages are decoupled — any individual stage can be re run without restarting the full pipeline. Stage failures are caught and logged without halting downstream execution.

| Stage | Module | Output |
|---|---|---|
| ingest | `ingest.py` | Raw CSV/GZ files in `data/raw/` |
| profile | `profile.py` | `reports/profiling_summary.csv` |
| clean | `clean.py` | Parquet files in `data/processed/` |
| quality | `quality.py` | `reports/data_quality_report.csv` |
| enrich | `enrich.py` | `data/processed/bangkok/enriched_listings.parquet` |
| model | `model.py` | `data/warehouse/airbnb_bangkok.duckdb` |
| analysis | `analysis.py` | 9 PNG charts + `reports/eda_insights_summary.csv` |
| statistics | `statistics.py` | `reports/statistical_tests_summary.csv` |
| business_insights | `business_insights.py` | `reports/business_recommendations.csv` |
| ml_price_model | `ml_price_model.py` | `reports/model_results.csv` + feature importance |

2. DuckDB Star Schema Warehouse

A dimensional warehouse with three fact tables and four dimension tables:

- `fact_listing_snapshot` — pricing, scores, and availability per listing at snapshot time
- `fact_calendar` — daily calendar data (availability, price) for all 28,806 listings
- `fact_reviews` — review records linked to listing and date dimensions
- `dim_listing`, `dim_host`, `dim_neighbourhood`, `dim_date`

3. Data Quality Framework

A custom quality check engine in `quality.py` that validates null constraints, duplicate keys, range checks, and type correctness across all three core tables. Results are written to CSV with severity levels (ERROR / WARNING) and passed/failed status.

#Pipeline quality result: 12/12 checks passed. 0 failed checks across 39,906,341 total rows validated.

4. Statistical Hypothesis Testing

Four formal tests on key business questions using Mann-Whitney U and Kruskal-Wallis:

| Business Question | Test | p-value | Result |
| Do entire homes cost more than private rooms? | Mann-Whitney U (log) | 0.0 | Statistically significant |
| Do superhosts have higher review scores? | Mann-Whitney U | 0.0 | Statistically significant |
| Do professional hosts price differently? | Mann-Whitney U (log) | 0.374 | Not significant |
| Do neighbourhood prices differ? | Kruskal-Wallis | 0.0 | Statistically significant |

5. ML Price Prediction Prototype

Three baseline models trained on `log(price)` to predict nightly listing price. 15 features including room type, neighbourhood, bedrooms, bathrooms, review score, host tenure, and occupancy proxy.

| Model | R² | MAE (THB) | RMSE (THB) |
|---|---|---|---|
| Linear Regression | 0.7947 | 747.85 | 7,682.78 |
| Random Forest | 0.9738 | 33.36 | 206.30 |
| XGBoost | 0.9799 | 57.92 | 243.69 |

Top XGBoost feature: `price_per_bedroom` (importance: 0.355), followed by `bedrooms` (0.243).

These are prototype models for understanding price drivers. They are not production pricing tools.

#6. Streamlit Executive Dashboard

An 11 tab interactive dashboard designed for a non-technical executive audience. Loads from the DuckDB warehouse with a Parquet fallback. Includes sidebar filters (neighbourhood, stay type, price range, host type), safe handling of empty filter results, and a filtered CSV download button.

Tabs: Executive Summary · Market Overview · Pricing Signals · Neighbourhoods · Host Strategy · Demand & Guest Experience · Seasonality · Statistical Evidence · Data Quality · Methodology · ML Results

#7. Business Recommendations Engine

A rule based engine that reads enriched data and generates structured recommendations with finding, business meaning, recommended action, confidence level, and limitations. Three actionable recommendations generated for Bangkok:

1. Host Strategy — 72.6% professional host share requires dual-track operations strategy
2. Neighbourhood Strategy — Pom Prap Sattru Phai: low supply, top-tier pricing
3. Demand / Reviews — 2,529 high-occupancy, below-average-rating listings are prime service improvement targets

# Technical Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas 2.2.2, NumPy 1.26.4 |
| File formats | Parquet (PyArrow), CSV.GZ |
| Warehouse | DuckDB 0.10.3 (embedded OLAP) |
| Statistics | SciPy 1.11.4, statsmodels 0.14.2 |
| ML | scikit-learn 1.4.2, XGBoost |
| Dashboard | Streamlit 1.35.0, Plotly 5.22.0 |
| Visualisation | Matplotlib 3.8.4 |
| Config | PyYAML 6.0.1 |
| Testing | pytest 8.2.2 |
| Environment | Python venv, PowerShell scripts |

# Quick Start

#Prerequisites:

- Python 3.10 or higher
- PowerShell (Windows)
- Internet connection (for data download at ingest stage)

#Step 1: Set up the environment

```powershell
cd "D:\Airbnb Market Intelligence"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1
```

This creates a `.venv` virtual environment and installs all pinned dependencies.

#Step 2: Run the full pipeline

```powershell
.\run_all.ps1
```

This activates the environment, clears any stale DuckDB files, runs all 10 pipeline stages for Bangkok, and executes the test suite.

#Expected runtime: approximately 5–15 minutes depending on internet speed (data download) and machine specs.

#Step 3: Launch the dashboard

```powershell
.\run_dashboard.ps1
```

Opens the Streamlit dashboard at `http://localhost:8501`.

---

# Manual Stage-by-Stage Commands

Activate the environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run individual pipeline stages:

```powershell
# Full pipeline (all stages)
python -m src.airbnb_pipeline.pipeline --city bangkok

# Specific stages only
python -m src.airbnb_pipeline.pipeline --city bangkok --stages ingest,clean,quality

# Force re-download of raw data
python -m src.airbnb_pipeline.pipeline --city bangkok --force

# Include the ML model stage
python -m src.airbnb_pipeline.pipeline --city bangkok --include-ml

# Debug logging
python -m src.airbnb_pipeline.pipeline --city bangkok --log-level DEBUG
```

Run tests:

```powershell
pytest
pytest tests/test_cleaning.py -v
pytest tests/test_quality.py -v
```

Launch dashboard manually:

```powershell
streamlit run dashboard/app.py
```

---

#Dashboard Explanation

The Streamlit dashboard (`dashboard/app.py`) reads from the DuckDB warehouse with an automatic fallback to Parquet files. It is designed for an executive or analyst audience with no requirement to read code.

| Tab | Content |
|---|---|
| 1. Executive Summary | KPI cards (total listings, median price, professional host share, avg rating), rule-based market summary, decision cards from `business_recommendations.csv` |
| 2. Market Overview | Stay type distribution (donut chart), price distribution histogram |
| 3. Pricing Signals | Median price by stay type (bar chart) with insight box |
| 4. Neighbourhoods | Supply vs price scatter plot, top 20 neighbourhood table, geographic price map |
| 5. Host Strategy | Professional vs casual split (donut), price by host type (bar) |
| 6. Demand & Guest Exp. | Review score distribution, operational improvement targets (low score, high volume listings) |
| 7. Seasonality | Weekend vs weekday pricing, monthly calendar price trend |
| 8. Statistical Evidence | All hypothesis test results with p-values and business interpretation |
| 9. Data Quality | Quality check results table, data profiling summary |
| 10. Methodology | Assumptions, limitations, data provenance |
| 11. ML Results | Model performance metrics and feature importance from `model_results.csv` |

Sidebar filters (neighbourhood, stay type, price range, superhost status, host type) apply across all tabs. An empty filter result shows a warning and defaults to the unfiltered view. A filtered CSV download button is available for further offline analysis.

---

#Generated Outputs

After running the pipeline, the following files are created:

```
data/
  processed/bangkok/
    clean_listings.parquet          # 28,806 rows — cleaned listing data
    clean_calendar.parquet          # 10,514,202 rows — cleaned calendar data
    clean_reviews.parquet           # 583,333 rows — cleaned review data
    clean_neighbourhoods.parquet    # 50 rows
    enriched_listings.parquet       # 28,806 rows — enriched with derived fields
    ingestion_metadata.csv          # File download timestamps and checksums

  warehouse/
    airbnb_bangkok.duckdb           # DuckDB warehouse (Star Schema, 7 tables)

reports/
  profiling_summary.csv             # Column-level profiling for all raw files
  data_quality_report.csv           # 12 quality checks — all PASS
  eda_insights_summary.csv          # 9 chart findings
  statistical_tests_summary.csv     # 4 hypothesis test results with p-values
  business_recommendations.csv      # 3 actionable recommendations
  model_results.csv                 # R², MAE, RMSE for 3 ML models
  model_feature_importance.csv      # Top 30 XGBoost feature importances
  decision_log.md                   # Technical trade-off rationale

  figures/
    price_distribution.png
    price_by_room_type.png
    top_neighbourhoods_price.png
    listings_by_neighbourhood.png
    host_listing_distribution.png
    professional_vs_casual.png
    review_score_distribution.png
    availability_by_month.png
    weekend_vs_weekday_price.png
```

---

# Key Findings (from actual pipeline output)

- Market scale: 28,806 listings across 50 Bangkok neighbourhoods.

- Pricing: Median nightly price is 1,370 THB. Distribution is right-skewed with a small luxury tail.

- Top neighbourhood: Parthum Wan (median 2,248 THB). Highest listing density: Vadhana (4,305 listings).

- Professionalisation: 72.6% of listings are managed by professional hosts (≥3 listings).
- Review quality: Median review score is 4.85 across 18,716 reviewed listings.

- Pricing by type: Entire home/apt commands a statistically significant premium over private rooms (p ≈ 0.0, rank-biserial r = -0.286).

- Superhost signal: Superhosts score measurably higher in reviews (p ≈ 0.0, rank-biserial r = -0.326).

- Neighbourhood pricing:Kruskal-Wallis confirms significant price variation across top-5 neighbourhoods (H = 736.34, p ≈ 0.0).

---

# Assumptions and Limitations

1. Revenue is a proxy. Calendar `available=False` may mean a host-blocked date, not a confirmed booking. Estimated revenue figures should be treated as directional upper-bound proxies.

2. Single snapshot. All data comes from a single scrape on 2025-09-26. Seasonal trends visible in the calendar data reflect future intended pricing, not historical confirmed bookings.

3. Professional host threshold. A host with ≥3 listings is classified as "professional." This is a commonly used but arbitrary threshold from Airbnb research literature.

4. Price outlier handling. Extreme prices are retained in the data but charts use a 99th-percentile cap for readability. Statistical tests use log-transformed prices.

5. No license data. The `license` column is 100% null in the Bangkok dataset. Regulatory compliance analysis is not possible from this data source.

6. Bangkok-specific findings. All insights apply to Bangkok only and should not be generalised to other markets without separate analysis.

---



# Future Improvements

Given additional time, the following enhancements would be prioritised:

1. Orchestration — Replace the PowerShell script with Apache Airflow or Prefect for scheduled runs, retries, and observability.

2. Multi-city support — The `cities.yml` config already supports multi-city definitions. Extending the pipeline to run Bangkok alongside Tokyo or Singapore would enable cross-market benchmarking.

3. SCD Type 2 for hosts — Track host attribute changes across snapshots to model host lifecycle and professionalization trends.

4. Geospatial enrichment — Join the `neighbourhoods.geojson` with listing coordinates using GeoPandas to enable proximity-based features (distance to BTS Skytrain, tourist landmarks).

5. Cloud deployment — Move the DuckDB warehouse to MotherDuck or export to BigQuery/Redshift for multi-user access.

6. ML improvement — Add cross-validation, hyperparameter tuning, and SHAP explanations to the ML prototype.

8. Real-time data— Integrate with a scraping schedule to maintain a rolling 12-month history for genuine time-series analysis.


Data source: [Inside Airbnb](http://insideairbnb.com) — data made available under a [Creative Commons CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) licence.
