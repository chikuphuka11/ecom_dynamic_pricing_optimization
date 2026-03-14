"""
src/features/pipeline.py

Scikit-learn preprocessing pipeline for demand elasticity modeling.

Economics note: Feature engineering here operationalizes core micro concepts —
price indices, cross-elasticity proxies, and demand shifters (income, seasonality).
The pipeline is built to be identical at training and inference time, preventing
the most common production ML failure: train-serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from category_encoders import TargetEncoder
from feature_engine.outliers import Winsorizer
from loguru import logger


# ─── Custom Transformers ──────────────────────────────────────────────────────

class PriceRelativeFeatures(BaseEstimator, TransformerMixin):
    """
    Compute price features relative to category benchmarks.

    Economics rationale: Consumers respond to RELATIVE prices, not absolute ones.
    A $5 coffee is cheap; a $5 pen is expensive. We operationalize this as
    price relative to category median (a proxy for reference price in prospect theory).
    """

    def __init__(self, price_col: str = "price", category_col: str = "department"):
        self.price_col = price_col
        self.category_col = category_col
        self._category_medians: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self._category_medians = (
            X.groupby(self.category_col)[self.price_col].median().to_dict()
        )
        self._overall_median = X[self.price_col].median()
        logger.info(f"Fitted price relatives for {len(self._category_medians)} categories")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cat_median = X[self.category_col].map(self._category_medians).fillna(self._overall_median)
        X["price_vs_category_median"] = X[self.price_col] / cat_median
        X["log_price"] = np.log1p(X[self.price_col])
        X["log_price_vs_category"] = np.log1p(X[self.price_col]) - np.log1p(cat_median)
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract time-based demand shifters from a date column.

    Economics rationale: Demand curves shift with time due to seasonality,
    income effects (paydays), and preference changes. These are the 'other factors
    held constant' in standard demand analysis — we make them explicit features.
    """

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        return self  # Stateless transformer

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dates = pd.to_datetime(X[self.date_col])

        X["day_of_week"] = dates.dt.dayofweek            # 0=Mon, 6=Sun
        X["day_of_month"] = dates.dt.day
        X["month"] = dates.dt.month
        X["quarter"] = dates.dt.quarter
        X["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        X["week_of_year"] = dates.dt.isocalendar().week.astype(int)

        # Payday proximity (simplified: 1st and 15th of month)
        X["days_to_payday"] = X["day_of_month"].apply(
            lambda d: min(abs(d - 1), abs(d - 15), abs(d - 30))
        )
        X["is_near_payday"] = (X["days_to_payday"] <= 2).astype(int)

        # Fourier features for weekly and annual seasonality
        # NOTE: Fourier terms encode non-linear seasonal patterns in a linear model
        # This is standard in time-series econometrics (e.g., Box-Jenkins)
        day_of_year = dates.dt.day_of_year
        for k in [1, 2, 3]:
            X[f"sin_annual_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
            X[f"cos_annual_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

        day_of_week_float = dates.dt.dayofweek.astype(float)
        X["sin_weekly"] = np.sin(2 * np.pi * day_of_week_float / 7)
        X["cos_weekly"] = np.cos(2 * np.pi * day_of_week_float / 7)

        return X


class DemandLagFeatures(BaseEstimator, TransformerMixin):
    """
    Create lagged demand and rolling statistics.

    Economics rationale: Habit formation (Becker & Murphy 1988) suggests
    current demand depends on past consumption. Rolling averages proxy for
    the 'stock of consumption capital'. These also serve as instruments
    in our causal identification strategy.
    """

    def __init__(
        self,
        target_col: str = "units_sold",
        sku_col: str = "product_id",
        lag_days: list[int] = None,
        rolling_windows: list[int] = None,
    ):
        self.target_col = target_col
        self.sku_col = sku_col
        self.lag_days = lag_days or [1, 7, 14, 28]
        self.rolling_windows = rolling_windows or [7, 14, 30]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy().sort_values(["date", self.sku_col])

        for lag in self.lag_days:
            X[f"demand_lag_{lag}d"] = (
                X.groupby(self.sku_col)[self.target_col].shift(lag)
            )

        for window in self.rolling_windows:
            X[f"demand_rolling_mean_{window}d"] = (
                X.groupby(self.sku_col)[self.target_col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
            )
            X[f"demand_rolling_std_{window}d"] = (
                X.groupby(self.sku_col)[self.target_col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=3).std())
            )

        return X


class CompetitorPriceFeatures(BaseEstimator, TransformerMixin):
    """
    Compute cross-price features relative to competitor benchmarks.

    Economics rationale: Cross-price elasticity — how our price relative to
    substitutes affects demand. If competitor_price_gap > 0, we're cheaper
    (demand boost for elastic goods). If < 0, we're more expensive (demand loss).
    """

    def __init__(self, price_col: str = "price", competitor_col: str = "competitor_price"):
        self.price_col = price_col
        self.competitor_col = competitor_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.competitor_col in X.columns:
            X["competitor_price_gap_pct"] = (
                (X[self.competitor_col] - X[self.price_col]) / X[self.price_col]
            )
            X["is_price_leader"] = (X[self.price_col] <= X[self.competitor_col]).astype(int)
            X["log_price_ratio"] = np.log(
                X[self.price_col] / X[self.competitor_col].replace(0, np.nan)
            )
        return X


# ─── Pipeline Builder ─────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str = "units_sold",
) -> Pipeline:
    """
    Build the full sklearn preprocessing pipeline.

    This pipeline:
    1. Winsorizes outliers (caps extreme values at 1st/99th percentile)
    2. Imputes missing values with median (numeric) or mode (categorical)
    3. Target-encodes high-cardinality categoricals (avoids one-hot explosion)
    4. Scales numeric features

    The pipeline is serialized with joblib and reused identically at inference.
    This is critical for production correctness — it must see the same
    transformations as training data.

    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        target_col: Target variable name (needed for TargetEncoder)

    Returns:
        Fitted sklearn Pipeline object
    """
    numeric_pipeline = Pipeline([
        ("winsorize", Winsorizer(capping_method="iqr", tail="both", fold=1.5)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", TargetEncoder(smoothing=10)),  # Smoothing prevents overfitting on rare categories
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    logger.info(
        f"Built preprocessing pipeline with {len(numeric_features)} numeric "
        f"and {len(categorical_features)} categorical features"
    )
    return pipeline


# ─── Feature Definitions ──────────────────────────────────────────────────────

# These lists define which features flow into the model.
# Separating them from code makes it easy to ablate features in experiments.
NUMERIC_FEATURES = [
    "log_price",
    "price_vs_category_median",
    "log_price_vs_category",
    "competitor_price_gap_pct",
    "log_price_ratio",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_near_payday",
    "days_to_payday",
    "sin_annual_1", "cos_annual_1",
    "sin_annual_2", "cos_annual_2",
    "sin_annual_3", "cos_annual_3",
    "sin_weekly", "cos_weekly",
    "demand_lag_1d",
    "demand_lag_7d",
    "demand_lag_14d",
    "demand_lag_28d",
    "demand_rolling_mean_7d",
    "demand_rolling_mean_14d",
    "demand_rolling_mean_30d",
    "demand_rolling_std_7d",
    "review_score",
    "days_since_launch",
    "inventory_level",
    "discount_depth",
    "is_on_promotion",
]

CATEGORICAL_FEATURES = [
    "department",
    "aisle_id",
]

HIGH_CARD_FEATURES = [
    "product_id",  # Encoded separately with higher smoothing
]

# Features used in DML as confounders W (controls) — NOT the treatment (price)
# Economics: these are the 'other factors' we condition on to isolate price effect
DML_CONTROL_FEATURES = [
    "day_of_week", "month", "is_weekend", "is_near_payday",
    "sin_annual_1", "cos_annual_1", "sin_weekly", "cos_weekly",
    "demand_lag_7d", "demand_rolling_mean_30d",
    "review_score", "inventory_level", "is_on_promotion",
    "department",
]

# Treatment variable for DML (what we're estimating the causal effect of)
DML_TREATMENT = "log_price"

# Outcome variable for DML
DML_OUTCOME = "log_units_sold"
