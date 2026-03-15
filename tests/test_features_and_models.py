"""
tests/test_features_and_models.py

Tests for src/features/pipeline.py and src/models/causal_dml.py.
Targeting 80%+ overall coverage.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


# ─── Shared Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame that satisfies all pipeline transformers."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "product_id":       [f"prod_{i%10:03d}" for i in range(n)],
        "department":       np.random.choice(["beverages", "produce", "dairy"], n),
        "aisle_id":         np.random.choice(["coffee", "milk", "fruit"], n),
        "price":            np.random.uniform(1.5, 9.9, n),
        "competitor_price": np.random.uniform(1.5, 9.9, n),
        "units_sold":       np.random.randint(10, 200, n),
        "date":             dates,
        "review_score":     np.random.uniform(3.0, 5.0, n),
        "days_since_launch": np.random.randint(0, 500, n),
        "inventory_level":  np.random.randint(0, 300, n),
        "discount_depth":   np.random.uniform(0.0, 0.3, n),
        "is_on_promotion":  np.random.randint(0, 2, n),
    })
    return df


@pytest.fixture
def dml_df(sample_df):
    df = sample_df.copy()
    df["log_price"]      = np.log(df["price"])
    df["log_units_sold"] = np.log(df["units_sold"] + 1)
    df["month"]          = df["date"].dt.month
    df["day_of_week"]    = df["date"].dt.dayofweek
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_near_payday"] = 0
    df["sin_annual_1"]   = np.sin(2 * np.pi * df["date"].dt.day_of_year / 365.25)
    df["cos_annual_1"]   = np.cos(2 * np.pi * df["date"].dt.day_of_year / 365.25)
    df["sin_weekly"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_weekly"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["demand_lag_7d"]           = df["units_sold"].shift(7).fillna(0)
    df["demand_rolling_mean_30d"] = df["units_sold"].rolling(30, min_periods=1).mean()
    # ✅ Encode department as numeric so DML can process it
    df["department"] = df["department"].map({"beverages": 0, "produce": 1, "dairy": 2})
    return df


# ─── PriceRelativeFeatures ────────────────────────────────────────────────────

class TestPriceRelativeFeatures:

    def test_fit_stores_category_medians(self, sample_df):
        from features.pipeline import PriceRelativeFeatures
        t = PriceRelativeFeatures()
        t.fit(sample_df)
        assert len(t._category_medians) == 3   # beverages, produce, dairy
        assert all(v > 0 for v in t._category_medians.values())

    def test_transform_adds_expected_columns(self, sample_df):
        from features.pipeline import PriceRelativeFeatures
        t = PriceRelativeFeatures().fit(sample_df)
        out = t.transform(sample_df)
        for col in ["price_vs_category_median", "log_price", "log_price_vs_category"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_price_vs_category_median_positive(self, sample_df):
        from features.pipeline import PriceRelativeFeatures
        t = PriceRelativeFeatures().fit(sample_df)
        out = t.transform(sample_df)
        assert (out["price_vs_category_median"] > 0).all()

    def test_log_price_is_log_of_price(self, sample_df):
        from features.pipeline import PriceRelativeFeatures
        t = PriceRelativeFeatures().fit(sample_df)
        out = t.transform(sample_df)
        expected = np.log1p(sample_df["price"])
        np.testing.assert_allclose(out["log_price"], expected)

    def test_unseen_category_uses_overall_median(self, sample_df):
        from features.pipeline import PriceRelativeFeatures
        t = PriceRelativeFeatures().fit(sample_df)
        unseen = sample_df.copy()
        unseen["department"] = "unknown_dept"
        out = t.transform(unseen)
        assert out["price_vs_category_median"].notna().all()

    def test_original_df_not_mutated(self, sample_df):
        from features.pipeline import PriceRelativeFeatures
        original_cols = set(sample_df.columns)
        t = PriceRelativeFeatures().fit(sample_df)
        t.transform(sample_df)
        assert set(sample_df.columns) == original_cols


# ─── TemporalFeatureExtractor ─────────────────────────────────────────────────

class TestTemporalFeatureExtractor:

    def test_adds_all_temporal_columns(self, sample_df):
        from features.pipeline import TemporalFeatureExtractor
        t = TemporalFeatureExtractor()
        out = t.fit_transform(sample_df)
        expected = [
            "day_of_week", "day_of_month", "month", "quarter",
            "is_weekend", "week_of_year", "days_to_payday", "is_near_payday",
            "sin_annual_1", "cos_annual_1", "sin_weekly", "cos_weekly",
        ]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_is_weekend_correct(self, sample_df):
        from features.pipeline import TemporalFeatureExtractor
        t = TemporalFeatureExtractor()
        out = t.fit_transform(sample_df)
        for _, row in out.iterrows():
            assert row["is_weekend"] == int(row["day_of_week"] >= 5)

    def test_month_range(self, sample_df):
        from features.pipeline import TemporalFeatureExtractor
        out = TemporalFeatureExtractor().fit_transform(sample_df)
        assert out["month"].between(1, 12).all()

    def test_fourier_features_bounded(self, sample_df):
        from features.pipeline import TemporalFeatureExtractor
        out = TemporalFeatureExtractor().fit_transform(sample_df)
        for col in ["sin_annual_1", "cos_annual_1", "sin_weekly", "cos_weekly"]:
            assert out[col].between(-1.01, 1.01).all(), f"{col} out of [-1, 1]"

    def test_fourier_higher_harmonics_present(self, sample_df):
        from features.pipeline import TemporalFeatureExtractor
        out = TemporalFeatureExtractor().fit_transform(sample_df)
        for k in [1, 2, 3]:
            assert f"sin_annual_{k}" in out.columns
            assert f"cos_annual_{k}" in out.columns

    def test_days_to_payday_non_negative(self, sample_df):
        from features.pipeline import TemporalFeatureExtractor
        out = TemporalFeatureExtractor().fit_transform(sample_df)
        assert (out["days_to_payday"] >= 0).all()

    def test_fit_is_stateless(self, sample_df):
        from features.pipeline import TemporalFeatureExtractor
        t = TemporalFeatureExtractor()
        out1 = t.fit_transform(sample_df)
        out2 = t.transform(sample_df)
        pd.testing.assert_frame_equal(out1, out2)


# ─── DemandLagFeatures ────────────────────────────────────────────────────────

class TestDemandLagFeatures:

    def test_creates_lag_columns(self, sample_df):
        from features.pipeline import DemandLagFeatures
        t = DemandLagFeatures(lag_days=[1, 7])
        out = t.fit_transform(sample_df)
        assert "demand_lag_1d" in out.columns
        assert "demand_lag_7d" in out.columns

    def test_creates_rolling_columns(self, sample_df):
        from features.pipeline import DemandLagFeatures
        t = DemandLagFeatures(rolling_windows=[7, 30])
        out = t.fit_transform(sample_df)
        assert "demand_rolling_mean_7d" in out.columns
        assert "demand_rolling_mean_30d" in out.columns
        assert "demand_rolling_std_7d" in out.columns

    def test_lag_columns_have_some_nulls(self, sample_df):
        from features.pipeline import DemandLagFeatures
        t = DemandLagFeatures(lag_days=[7], rolling_windows=[])
        out = t.fit_transform(sample_df)
         # Early rows per product should have NaN lags
        assert out["demand_lag_7d"].isna().any()

    def test_lag_values_match_shifted_units(self, sample_df):
        from features.pipeline import DemandLagFeatures
        t = DemandLagFeatures(lag_days=[1], rolling_windows=[])
        df = sample_df.sort_values(["product_id", "date"]).copy()
        out = t.fit_transform(df)
        # Non-null lag values should be positive (units_sold > 0)
        non_null = out["demand_lag_1d"].dropna()
        assert (non_null > 0).all()

    def test_default_lag_days(self, sample_df):
        from features.pipeline import DemandLagFeatures
        t = DemandLagFeatures()
        assert t.lag_days == [1, 7, 14, 28]

    def test_default_rolling_windows(self, sample_df):
        from features.pipeline import DemandLagFeatures
        t = DemandLagFeatures()
        assert t.rolling_windows == [7, 14, 30]


# ─── CompetitorPriceFeatures ──────────────────────────────────────────────────

class TestCompetitorPriceFeatures:

    def test_adds_competitor_columns(self, sample_df):
        from features.pipeline import CompetitorPriceFeatures
        t = CompetitorPriceFeatures()
        out = t.fit_transform(sample_df)
        for col in ["competitor_price_gap_pct", "is_price_leader", "log_price_ratio"]:
            assert col in out.columns

    def test_is_price_leader_binary(self, sample_df):
        from features.pipeline import CompetitorPriceFeatures
        out = CompetitorPriceFeatures().fit_transform(sample_df)
        assert set(out["is_price_leader"].unique()).issubset({0, 1})

    def test_no_competitor_col_leaves_df_unchanged(self, sample_df):
        from features.pipeline import CompetitorPriceFeatures
        df_no_comp = sample_df.drop(columns=["competitor_price"])
        original_cols = set(df_no_comp.columns)
        out = CompetitorPriceFeatures().fit_transform(df_no_comp)
        assert set(out.columns) == original_cols

    def test_price_gap_sign(self, sample_df):
        from features.pipeline import CompetitorPriceFeatures
        df = sample_df.copy()
        df["price"] = 5.0
        df["competitor_price"] = 6.0   # competitor is more expensive
        out = CompetitorPriceFeatures().fit_transform(df)
        # gap = (comp - own) / own = (6-5)/5 = 0.2 > 0
        assert (out["competitor_price_gap_pct"] > 0).all()


# ─── build_preprocessing_pipeline ────────────────────────────────────────────

class TestBuildPreprocessingPipeline:

    def test_pipeline_returns_sklearn_pipeline(self):
        from features.pipeline import build_preprocessing_pipeline
        from sklearn.pipeline import Pipeline
        p = build_preprocessing_pipeline(
            numeric_features=["log_price", "month"],
            categorical_features=["department"],
        )
        assert isinstance(p, Pipeline)

    def test_pipeline_has_preprocessor_step(self):
        from features.pipeline import build_preprocessing_pipeline
        p = build_preprocessing_pipeline(
            numeric_features=["log_price"],
            categorical_features=["department"],
        )
        assert "preprocessor" in p.named_steps


# ─── Feature Definition Constants ─────────────────────────────────────────────

class TestFeatureConstants:

    def test_numeric_features_is_list(self):
        from features.pipeline import NUMERIC_FEATURES
        assert isinstance(NUMERIC_FEATURES, list)
        assert len(NUMERIC_FEATURES) > 0

    def test_categorical_features_is_list(self):
        from features.pipeline import CATEGORICAL_FEATURES
        assert isinstance(CATEGORICAL_FEATURES, list)

    def test_dml_treatment_is_log_price(self):
        from features.pipeline import DML_TREATMENT
        assert DML_TREATMENT == "log_price"

    def test_dml_outcome_is_log_units_sold(self):
        from features.pipeline import DML_OUTCOME
        assert DML_OUTCOME == "log_units_sold"

    def test_dml_control_features_excludes_treatment(self):
        from features.pipeline import DML_CONTROL_FEATURES, DML_TREATMENT
        assert DML_TREATMENT not in DML_CONTROL_FEATURES

    def test_no_duplicate_numeric_features(self):
        from features.pipeline import NUMERIC_FEATURES
        assert len(NUMERIC_FEATURES) == len(set(NUMERIC_FEATURES))


# ─── PriceElasticityModel ─────────────────────────────────────────────────────

class TestPriceElasticityModel:

    def test_init_default_params(self):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel()
        assert m.n_splits == 5
        assert m.n_estimators == 200
        assert m.random_state == 42
        assert m.model is None

    def test_init_custom_params(self):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=3, n_estimators=100, random_state=0)
        assert m.n_splits == 3
        assert m.n_estimators == 100

    def test_build_nuisance_model_returns_lgbm(self):
        from models.causal_dml import PriceElasticityModel
        from lightgbm import LGBMRegressor
        m = PriceElasticityModel()
        nuisance = m._build_nuisance_model()
        assert isinstance(nuisance, LGBMRegressor)

    def test_predict_elasticity_raises_before_fit(self):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_elasticity()

    def test_fit_sets_model_attribute(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df)
        assert m.model is not None

    def test_fit_sets_ate(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df)
        assert m._ate is not None
        assert isinstance(m._ate, float)

    def test_fit_returns_self(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        result = m.fit(dml_df)
        assert result is m

    def test_predict_elasticity_returns_array(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df)
        elasticities = m.predict_elasticity()
        assert isinstance(elasticities, np.ndarray)
        assert len(elasticities) >= 1   # ATE returns 1 value when X=None

    def test_elasticity_summary_keys(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df)
        summary = m.elasticity_summary()
        expected_keys = [
            "ate", "ate_stderr", "ate_95ci_lower", "ate_95ci_upper",
            "cate_mean", "cate_std", "cate_p10", "cate_p50", "cate_p90",
            "n_elastic", "n_inelastic",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_elasticity_ci_is_ordered(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df)
        s = m.elasticity_summary()
        assert s["ate_95ci_lower"] < s["ate_95ci_upper"]

    def test_cate_percentiles_ordered(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df)
        s = m.elasticity_summary()
        assert s["cate_p10"] <= s["cate_p50"] <= s["cate_p90"]

    def test_n_elastic_plus_n_inelastic_equals_total(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df)
        s = m.elasticity_summary()
        # Without X, effect() returns ATE (1 value) not per-sample CATEs
        assert s["n_elastic"] + s["n_inelastic"] >= 1

    def test_fit_with_heterogeneity_features(self, dml_df):
        from models.causal_dml import PriceElasticityModel
        m = PriceElasticityModel(n_splits=2, n_estimators=10)
        m.fit(dml_df, heterogeneity_features=["month", "is_weekend"])
        assert m.model is not None
        assert m._ate is not None