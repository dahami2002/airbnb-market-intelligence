"""Statistical hypothesis testing for business questions."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import get_city_config, get_data_paths
from .utils import ensure_dir

logger = logging.getLogger("airbnb_pipeline.statistics")


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def _rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    return float(1 - (2 * u_stat) / (n1 * n2))


def _interpret_effect_size(d: float) -> str:
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    return "large"


def _choose_and_run_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[str, float, float, str]:
    n1, n2 = len(group1), len(group2)
    sample_size = min(n1, n2, 5000)
    try:
        _, p_normal_1 = stats.shapiro(np.random.choice(group1, size=min(sample_size, n1), replace=False))
        _, p_normal_2 = stats.shapiro(np.random.choice(group2, size=min(sample_size, n2), replace=False))
        is_normal = p_normal_1 > 0.05 and p_normal_2 > 0.05
    except Exception:
        is_normal = False

    if is_normal and n1 >= 30 and n2 >= 30:
        stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        d = _cohens_d(group1, group2)
        effect_str = f"Cohen's d = {d:.3f} ({_interpret_effect_size(d)})"
        return "Welch's t-test", float(stat), float(p_val), effect_str
    else:
        stat, p_val = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        r = _rank_biserial_r(stat, n1, n2)
        effect_str = f"Rank-biserial r = {r:.3f}"
        return "Mann-Whitney U", float(stat), float(p_val), effect_str


def _format_result(p_val: float) -> str:
    if p_val < 0.05:
        return "The difference is unlikely to be random."
    return "The difference could easily be random chance."


def run_statistical_tests(city_key: str) -> pd.DataFrame:
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    processed_dir = paths["processed_dir"]
    reports_dir = ensure_dir(paths["reports_dir"])

    logger.info("Starting statistical tests for %s", city_cfg["display_name"])

    enriched_path = processed_dir / "enriched_listings.parquet"
    cal_path = processed_dir / "clean_calendar.parquet"

    if not enriched_path.exists():
        logger.error("Enriched listings not found.")
        return pd.DataFrame()

    df = pd.read_parquet(enriched_path)
    df_cal = pd.read_parquet(cal_path) if cal_path.exists() else None

    results: List[Dict[str, Any]] = []
    hyp_id = 1

    # 1. Entire Homes vs Private Rooms price difference
    if "room_type" in df.columns and "price" in df.columns:
        entire = df.loc[(df["room_type"].str.contains("Entire", case=False, na=False)) & (df["price"] > 0), "price"].dropna()
        private = df.loc[(df["room_type"].str.contains("Private", case=False, na=False)) & (df["price"] > 0), "price"].dropna()

        if len(entire) >= 10 and len(private) >= 10:
            test_name, stat, p_val, effect_str = _choose_and_run_test(np.log1p(entire.values), np.log1p(private.values))
            results.append({
                "hypothesis_id": hyp_id,
                "business_question": "Do entire homes cost more than private rooms?",
                "null_hypothesis": "There is no price difference between entire homes and private rooms.",
                "alternative_hypothesis": "Entire homes and private rooms have different typical prices.",
                "test_used": f"{test_name} (log-transformed)",
                "sample_size_group_1": len(entire),
                "sample_size_group_2": len(private),
                "statistic": round(stat, 4),
                "p_value": round(p_val, 6),
                "effect_size": effect_str,
                "result_plain_english": _format_result(p_val),
                "business_interpretation": "Entire homes command a price premium. Pricing strategies should strictly segment by stay type.",
                "limitation": "Medians and log-transforms reduce outlier impact. Correlation is not causation."
            })
            hyp_id += 1

    # 2. Superhost vs non-superhost review score
    if "host_is_superhost" in df.columns and "review_scores_rating" in df.columns:
        superhost = df.loc[df["host_is_superhost"] == True, "review_scores_rating"].dropna()
        non_super = df.loc[df["host_is_superhost"] == False, "review_scores_rating"].dropna()

        if len(superhost) >= 10 and len(non_super) >= 10:
            test_name, stat, p_val, effect_str = _choose_and_run_test(superhost.values, non_super.values)
            results.append({
                "hypothesis_id": hyp_id,
                "business_question": "Do superhosts have higher review scores?",
                "null_hypothesis": "Superhosts and non-superhosts have identical review scores.",
                "alternative_hypothesis": "Superhosts have different review scores than non-superhosts.",
                "test_used": test_name,
                "sample_size_group_1": len(superhost),
                "sample_size_group_2": len(non_super),
                "statistic": round(stat, 4),
                "p_value": round(p_val, 6),
                "effect_size": effect_str,
                "result_plain_english": _format_result(p_val),
                "business_interpretation": "Superhosts tend to have higher scores, validating the Superhost badge as a proxy for guest satisfaction.",
                "limitation": "Circular logic risk: Superhost status strictly requires high ratings."
            })
            hyp_id += 1

    # 3. Professional vs casual host price difference
    if "is_professional_host" in df.columns and "price" in df.columns:
        pro = df.loc[(df["is_professional_host"] == True) & (df["price"] > 0), "price"].dropna()
        casual = df.loc[(df["is_professional_host"] == False) & (df["price"] > 0), "price"].dropna()

        if len(pro) >= 10 and len(casual) >= 10:
            test_name, stat, p_val, effect_str = _choose_and_run_test(np.log1p(pro.values), np.log1p(casual.values))
            results.append({
                "hypothesis_id": hyp_id,
                "business_question": "Do professional hosts price differently from casual hosts?",
                "null_hypothesis": "Professional and casual hosts price similarly.",
                "alternative_hypothesis": "Professional hosts apply different pricing strategies.",
                "test_used": f"{test_name} (log-transformed)",
                "sample_size_group_1": len(pro),
                "sample_size_group_2": len(casual),
                "statistic": round(stat, 4),
                "p_value": round(p_val, 6),
                "effect_size": effect_str,
                "result_plain_english": _format_result(p_val),
                "business_interpretation": "Professional hosts often use dynamic pricing or operate different property types, causing meaningful price gaps.",
                "limitation": "Professional threshold defined loosely as ≥3 listings."
            })
            hyp_id += 1

    # 4. Weekend vs weekday calendar price
    if df_cal is not None and "date" in df_cal.columns and "price" in df_cal.columns:
        df_cal["date"] = pd.to_datetime(df_cal["date"], errors="coerce")
        df_cal["is_weekend"] = df_cal["date"].dt.dayofweek.isin([4, 5])

        weekend = df_cal.loc[(df_cal["is_weekend"] == True) & (df_cal["price"] > 0), "price"].dropna()
        weekday = df_cal.loc[(df_cal["is_weekend"] == False) & (df_cal["price"] > 0), "price"].dropna()

        if len(weekend) >= 10 and len(weekday) >= 10:
            sample_n = min(50000, len(weekend), len(weekday))
            wnd_s = np.log1p(weekend.sample(n=sample_n, random_state=42).values)
            wdy_s = np.log1p(weekday.sample(n=sample_n, random_state=42).values)

            test_name, stat, p_val, effect_str = _choose_and_run_test(wnd_s, wdy_s)
            results.append({
                "hypothesis_id": hyp_id,
                "business_question": "Are weekend prices different from weekday prices?",
                "null_hypothesis": "Calendar prices are identical on weekends and weekdays.",
                "alternative_hypothesis": "There is a weekend pricing premium (or discount).",
                "test_used": f"{test_name} (sampled, log-transformed)",
                "sample_size_group_1": sample_n,
                "sample_size_group_2": sample_n,
                "statistic": round(stat, 4),
                "p_value": round(p_val, 6),
                "effect_size": effect_str,
                "result_plain_english": _format_result(p_val),
                "business_interpretation": "A significant difference implies host recognition of leisure demand. Weekend pricing should be managed as a distinct strategy.",
                "limitation": "Calendar prices are host intent, not realized bookings."
            })
            hyp_id += 1

    # 5. Kruskal-Wallis for Neighbourhoods
    nbr_col = "neighbourhood_cleansed" if "neighbourhood_cleansed" in df.columns else "neighbourhood"
    if nbr_col in df.columns and "price" in df.columns:
        top_nbrs = df[nbr_col].value_counts().head(5).index
        groups = [df.loc[(df[nbr_col] == n) & (df["price"] > 0), "price"].dropna().values for n in top_nbrs]

        if all(len(g) >= 10 for g in groups):
            stat, p_val = stats.kruskal(*groups)
            results.append({
                "hypothesis_id": hyp_id,
                "business_question": "Do neighbourhood prices differ?",
                "null_hypothesis": "All top 5 neighbourhoods have the same median price.",
                "alternative_hypothesis": "At least one neighbourhood has a different median price.",
                "test_used": "Kruskal-Wallis H-test",
                "sample_size_group_1": len(groups[0]),
                "sample_size_group_2": sum(len(g) for g in groups[1:]),
                "statistic": round(stat, 4),
                "p_value": round(p_val, 6),
                "effect_size": "N/A (Omnibus test)",
                "result_plain_english": _format_result(p_val),
                "business_interpretation": "Prices vary geographically. A city-wide pricing model is inadequate; hyper-local benchmarks are necessary.",
                "limitation": "Omnibus test only confirms a difference exists, not which specific pairs differ."
            })
            hyp_id += 1

    results_df = pd.DataFrame(results)
    out_path = reports_dir / "statistical_tests_summary.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Statistical test results saved to %s", out_path)

    return results_df
