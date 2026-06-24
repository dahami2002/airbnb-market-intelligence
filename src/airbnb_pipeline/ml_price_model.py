"""ML prototype for price prediction."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

from .config import get_city_config, get_data_paths
from .utils import ensure_dir

logger = logging.getLogger("airbnb_pipeline.ml_price_model")


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column from candidates that exists in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_binary(series: pd.Series) -> pd.Series:
    """Convert booleans and t/f text columns into 1/0 numeric values."""
    if series.dtype == bool:
        return series.fillna(False).astype(float)

    mapping = {
        "true": 1,
        "t": 1,
        "1": 1,
        "yes": 1,
        "y": 1,
        "false": 0,
        "f": 0,
        "0": 0,
        "no": 0,
        "n": 0,
        "nan": 0,
        "none": 0,
        "": 0,
    }
    return (
        series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(float)
    )


def _safe_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute RMSE in a version-safe way."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_evaluate_models(city_key: str) -> None:
    """
    Train baseline price prediction models and save results.

    This function is intentionally simple and transparent for assessment purposes.
    It does not claim production readiness.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    processed_dir = paths["processed_dir"]
    reports_dir = ensure_dir(paths["reports_dir"])

    enriched_path = processed_dir / "enriched_listings.parquet"
    if not enriched_path.exists():
        logger.error(
            "Enriched listings not found at %s. Cannot run ML model.", enriched_path
        )
        return

    logger.info("Loading enriched listings for ML price model...")
    df = pd.read_parquet(enriched_path)

    price_col = _first_existing_col(df, ["price", "listing_price", "nightly_price"])
    if price_col is None:
        logger.error("No usable price column found. Cannot run ML model.")
        return

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[df[price_col] > 0].copy()
    df = df[df[price_col] < df[price_col].quantile(0.99)].copy()

    if len(df) < 100:
        logger.error("Not enough valid rows for ML model after filtering: %s", len(df))
        return

    y = np.log1p(df[price_col])

    neighbourhood_col = _first_existing_col(
        df, ["neighbourhood_cleansed", "neighbourhood", "neighborhood"]
    )
    room_col = _first_existing_col(df, ["room_type", "stay_type"])
    superhost_col = _first_existing_col(df, ["host_is_superhost", "is_superhost"])
    professional_col = _first_existing_col(
        df, ["is_professional_host", "professional_host"]
    )

    categorical_features = []
    if room_col:
        categorical_features.append(room_col)
    if neighbourhood_col:
        categorical_features.append(neighbourhood_col)

    numeric_candidates = [
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms",
        "review_scores_rating",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "host_tenure_years",
        "price_per_bedroom",
        "estimated_occupancy_rate_365d",
    ]
    numeric_features = [col for col in numeric_candidates if col in df.columns]

    boolean_features = []
    if superhost_col:
        boolean_features.append(superhost_col)
    if professional_col:
        boolean_features.append(professional_col)

    all_features = categorical_features + numeric_features + boolean_features
    if not all_features:
        logger.error("No usable features found for ML model.")
        return

    X = df[all_features].copy()

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in boolean_features:
        X[col] = _to_binary(X[col])
        if col not in numeric_features:
            numeric_features.append(col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        logger.warning("xgboost is not installed. XGBoost model will be skipped.")

    results = []
    best_pipeline = None
    best_model_name = None
    best_r2 = -np.inf

    logger.info("Training ML models: %s", ", ".join(models.keys()))
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred_log = pipeline.predict(X_test)

        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred_log)
        y_pred_orig = np.where(np.isfinite(y_pred_orig), y_pred_orig, np.nan)

        valid_mask = ~np.isnan(y_pred_orig)
        mae = mean_absolute_error(y_test_orig[valid_mask], y_pred_orig[valid_mask])
        rmse = _safe_rmse(y_test_orig[valid_mask], y_pred_orig[valid_mask])
        r2 = r2_score(y_test, y_pred_log)

        results.append(
            {
                "model": name,
                "target": "price via log1p transform",
                "rows_used": len(df),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "features_used": ", ".join(all_features),
                "mae_thb": round(float(mae), 2),
                "rmse_thb": round(float(rmse), 2),
                "r2_score": round(float(r2), 4),
                "business_interpretation": "Baseline model for understanding price drivers; not production pricing advice.",
            }
        )
        logger.info("%s -> R2: %.3f, MAE: %.0f THB", name, r2, mae)

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipeline
            best_model_name = name

    results_df = pd.DataFrame(results)
    results_path = reports_dir / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved model results to %s", results_path)

    if best_pipeline is None:
        return

    regressor = best_pipeline.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        logger.info(
            "Best model %s has no feature_importances_ attribute.", best_model_name
        )
        return

    try:
        feature_names = []
        if numeric_features:
            feature_names.extend(numeric_features)

        if categorical_features:
            ohe = (
                best_pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["onehot"]
            )
            feature_names.extend(list(ohe.get_feature_names_out(categorical_features)))

        importances = regressor.feature_importances_
        min_len = min(len(feature_names), len(importances))

        fi_df = (
            pd.DataFrame(
                {
                    "model": best_model_name,
                    "feature": feature_names[:min_len],
                    "importance": importances[:min_len],
                }
            )
            .sort_values("importance", ascending=False)
            .head(30)
        )

        fi_path = reports_dir / "model_feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        logger.info("Saved feature importance to %s", fi_path)
    except Exception as exc:
        logger.warning("Could not extract feature importance: %s", exc)
