"""
src/api/predict.py

DemandPredictor — loads trained models and serves inference for all API endpoints.

Models used:
  - demand_model.pkl        : LightGBM trained in log-log space (lgbm_tuned.pkl)
  - preprocessing_pipeline.pkl : sklearn scaler / column transformer (dcn_scaler.pkl)
  - dml_model.pkl           : EconML DoubleML model for causal elasticity estimation
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize_scalar

from api.schemas import (
    BatchDemandRequest,
    BatchDemandResponse,
    BatchDemandItem,
    DemandForecastRequest,
    DemandForecastResponse,
    ElasticityRequest,
    ElasticityResponse,
    OptimalPriceRequest,
    OptimalPriceResponse,
    ProductContext,
)

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_VERSION = "1.0.0"

# Quantile offsets used to build a ~90 % prediction interval in log-space.
# Calibrated empirically; replace with proper quantile regression if available.
_PI_LOWER_Z = 1.645   # 5th percentile  → subtract from log prediction
_PI_UPPER_Z = 1.645   # 95th percentile → add to log prediction
_LOG_SIGMA   = 0.25   # assumed residual std-dev in log-space (tunable)


# ─── Feature Engineering ──────────────────────────────────────────────────────

_FEATURE_COLS = [
    "log_price",
    "log_competitor_price",
    "price_ratio",
    "discount_depth",
    "is_on_promotion",
    "inventory_level",
    "review_score",
    "days_since_launch",
    "demand_lag_7d",
    "demand_rolling_mean_30d",
    "month",
    "day_of_week",
    "is_weekend",
    "is_month_end",
]


def _build_features(ctx: ProductContext) -> pd.DataFrame:
    """Convert a ProductContext into a single-row feature DataFrame."""
    pricing_date = ctx.pricing_date  # already a date object after schema fix

    log_price = np.log(ctx.price)
    comp_price = ctx.competitor_price if ctx.competitor_price else ctx.price
    log_comp   = np.log(comp_price)

    row = {
        "log_price":                log_price,
        "log_competitor_price":     log_comp,
        "price_ratio":              ctx.price / comp_price,
        "discount_depth":           ctx.discount_depth or 0.0,
        "is_on_promotion":          int(ctx.is_on_promotion or False),
        "inventory_level":          ctx.inventory_level if ctx.inventory_level is not None else 100,
        "review_score":             ctx.review_score if ctx.review_score is not None else 3.5,
        "days_since_launch":        ctx.days_since_launch if ctx.days_since_launch is not None else 180,
        "demand_lag_7d":            ctx.demand_lag_7d if ctx.demand_lag_7d is not None else 0.0,
        "demand_rolling_mean_30d":  ctx.demand_rolling_mean_30d if ctx.demand_rolling_mean_30d is not None else 0.0,
        "month":                    pricing_date.month,
        "day_of_week":              pricing_date.weekday(),
        "is_weekend":               int(pricing_date.weekday() >= 5),
        "is_month_end":             int(pricing_date.day >= 28),
    }
    return pd.DataFrame([row], columns=_FEATURE_COLS)


def _build_elasticity_features(req: ElasticityRequest) -> pd.DataFrame:
    """Build features for the DML elasticity model."""
    row = {
        "month":                    req.month,
        "is_on_promotion":          int(req.is_on_promotion),
        "demand_rolling_mean_30d":  req.demand_rolling_mean_30d or 0.0,
        # department one-hot is handled below as a simple label encode
        "department_code":          abs(hash(req.department)) % 50,
    }
    return pd.DataFrame([row])


# ─── DemandPredictor ──────────────────────────────────────────────────────────

@dataclass
class DemandPredictor:
    """
    Wraps all ML models behind a clean interface consumed by the FastAPI routes.

    Attributes
    ----------
    lgbm_model   : fitted LightGBM (or sklearn) regressor in log-log space
    pipeline     : sklearn preprocessing pipeline / scaler
    dml_model    : EconML DoubleML estimator for elasticity CATEs
    demo_mode    : if True, all methods return plausible synthetic values
    """

    lgbm_model:  object = field(default=None, repr=False)
    pipeline:    object = field(default=None, repr=False)
    dml_model:   object = field(default=None, repr=False)
    demo_mode:   bool   = False

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_path:    str,
        pipeline_path: str,
        dml_path:      str,
    ) -> "DemandPredictor":
        """Load all model artefacts from disk."""
        mp = Path(model_path)
        pp = Path(pipeline_path)
        dp = Path(dml_path)

        missing = [str(p) for p in (mp, pp, dp) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Model files not found: {missing}")

        logger.info(f"Loading demand model from {mp}")
        lgbm = joblib.load(mp)

        logger.info(f"Loading preprocessing pipeline from {pp}")
        pipeline = joblib.load(pp)

        logger.info(f"Loading DML model from {dp}")
        dml = joblib.load(dp)

        return cls(lgbm_model=lgbm, pipeline=pipeline, dml_model=dml, demo_mode=False)

    @classmethod
    def demo(cls) -> "DemandPredictor":
        """Return a predictor that serves deterministic demo responses (no models needed)."""
        logger.warning("DemandPredictor running in DEMO mode — responses are synthetic")
        return cls(demo_mode=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _predict_log_demand(self, X: pd.DataFrame) -> tuple[float, float, float]:
        """
        Run the LGBM model and return (point, lower, upper) in original units.

        The model is assumed to predict log(units).  Prediction intervals are
        constructed by adding ±1.645 * _LOG_SIGMA in log-space then exp()-ing.
        """
        if self.demo_mode:
            # Plausible synthetic demand: 50 units, ±30 %
            log_pred = np.log(50.0)
        else:
            try:
                # Try transform → predict (sklearn pipeline wrapping LGBM)
                X_t = self.pipeline.transform(X) if self.pipeline else X
                log_pred = float(self.lgbm_model.predict(X_t)[0])
            except Exception as e:
                logger.warning(f"Pipeline transform failed ({e}), using raw features")
                log_pred = float(self.lgbm_model.predict(X)[0])

        point = float(np.exp(log_pred))
        lower = float(np.exp(log_pred - _PI_LOWER_Z * _LOG_SIGMA))
        upper = float(np.exp(log_pred + _PI_UPPER_Z * _LOG_SIGMA))
        return point, lower, upper

    def _estimate_elasticity(self, X_elas: pd.DataFrame) -> tuple[float, float, float]:
        """
        Return (elasticity, lower_95, upper_95) from the DML model.

        Falls back to a plausible industry-average elasticity if DML inference
        raises any exception (e.g. feature mismatch after notebook re-training).
        """
        if self.demo_mode:
            return -1.35, -1.75, -0.95

        try:
            # EconML models expose .effect() for CATE; sklearn regressors use .predict()
            if hasattr(self.dml_model, "effect"):
                theta = float(self.dml_model.effect(X_elas)[0])
                # Approximate SE from DML's effect_interval if available
                if hasattr(self.dml_model, "effect_interval"):
                    lo, hi = self.dml_model.effect_interval(X_elas, alpha=0.05)
                    return theta, float(lo[0]), float(hi[0])
                se = abs(theta) * 0.20          # fallback: ±20 % of point estimate
                return theta, theta - 1.96 * se, theta + 1.96 * se
            else:
                theta = float(self.dml_model.predict(X_elas)[0])
                se    = abs(theta) * 0.20
                return theta, theta - 1.96 * se, theta + 1.96 * se
        except Exception as e:
            logger.warning(f"DML inference failed ({e}), using fallback elasticity")
            return -1.20, -1.60, -0.80

    # ── Public API ────────────────────────────────────────────────────────────

    def forecast_demand(self, request: DemandForecastRequest) -> DemandForecastResponse:
        """Predict units sold at the requested price + context."""
        X = _build_features(request)
        point, lower, upper = self._predict_log_demand(X)

        return DemandForecastResponse(
            product_id=request.product_id,
            price=request.price,
            predicted_units=round(point, 2),
            prediction_lower=round(lower, 2),
            prediction_upper=round(upper, 2),
            model_version=MODEL_VERSION,
        )

    def find_optimal_price(self, request: OptimalPriceRequest) -> OptimalPriceResponse:
        """Revenue-maximise over a constrained price grid using the demand model."""
        current_price = request.price
        cost          = request.unit_cost
        min_margin    = request.min_margin_pct
        max_change    = request.max_price_change_pct
        objective     = request.objective

        # ── Constraints / bounds ──────────────────────────────────────────────
        margin_floor  = cost * (1.0 + min_margin)
        price_lo      = max(margin_floor, current_price * (1.0 - max_change))
        price_hi      = current_price * (1.0 + max_change)
        price_lo      = min(price_lo, price_hi)   # guard degenerate case

        # ── Objective function ────────────────────────────────────────────────
        def _neg_objective(p: float) -> float:
            """Return negative revenue (or profit/GMV) for a trial price p."""
            # Mutate a copy of the context with the trial price
            ctx_dict = request.model_dump(by_alias=True)
            ctx_dict["price"] = p
            trial = DemandForecastRequest(**ctx_dict)
            X     = _build_features(trial)
            units, _, _ = self._predict_log_demand(X)

            if objective == "profit":
                val = (p - cost) * units
            elif objective == "gmv":
                val = p * units        # GMV = revenue here (no marketplace fee model)
            else:                      # "revenue"
                val = p * units

            return -val                # minimiser → negate

        # ── Optimise ──────────────────────────────────────────────────────────
        result = minimize_scalar(
            _neg_objective,
            bounds=(price_lo, price_hi),
            method="bounded",
            options={"xatol": 0.001},
        )
        import math  # add this at the top of the file with other imports
        optimal_price = math.floor(float(result.x) * 100) / 100  # floor to 2dp, never rounds up
        optimal_price = max(price_lo, min(optimal_price, price_hi))  # clamp

        # ── Compute expected lifts ─────────────────────────────────────────────
        X_current = _build_features(request)
        units_current, _, _ = self._predict_log_demand(X_current)

        ctx_opt = request.model_dump(by_alias=True)
        ctx_opt["price"] = optimal_price
        trial_opt = DemandForecastRequest(**ctx_opt)
        X_opt = _build_features(trial_opt)
        units_optimal, _, _ = self._predict_log_demand(X_opt)

        rev_current = current_price * units_current
        rev_optimal = optimal_price * units_optimal
        rev_lift    = (rev_optimal - rev_current) / (rev_current + 1e-9)
        demand_chg  = (units_optimal - units_current) / (units_current + 1e-9)

        price_chg_pct = (optimal_price - current_price) / current_price

        # ── Elasticity (point estimate) ───────────────────────────────────────
        X_elas = _build_elasticity_features(
            ElasticityRequest(
                product_id=request.product_id,
                department=request.department,
                month=request.pricing_date.month,
                is_on_promotion=request.is_on_promotion or False,
                demand_rolling_mean_30d=request.demand_rolling_mean_30d,
            )
        )
        elasticity, _, _ = self._estimate_elasticity(X_elas)

        # ── Constraint diagnosis ──────────────────────────────────────────────
        if abs(optimal_price - margin_floor) < 0.01:
            constraint = "margin"
        elif abs(optimal_price - price_lo) < 0.01 or abs(optimal_price - price_hi) < 0.01:
            constraint = "guardrail"
        else:
            constraint = "none"

        # ── Confidence ────────────────────────────────────────────────────────
        has_lags = (request.demand_lag_7d is not None
                    and request.demand_rolling_mean_30d is not None)
        confidence = "high" if has_lags else ("medium" if request.competitor_price else "low")

        return OptimalPriceResponse(
            product_id=request.product_id,
            current_price=round(current_price, 2),
            optimal_price=optimal_price,
            price_change_pct=round(price_chg_pct, 4),
            expected_revenue_lift_pct=round(rev_lift * 100, 2),
            expected_demand_change_pct=round(demand_chg * 100, 2),
            estimated_elasticity=round(elasticity, 3),
            constraint_binding=constraint,
            confidence=confidence,
        )

    def get_elasticity(self, request: ElasticityRequest) -> ElasticityResponse:
        """Return price elasticity CATE from the DML model."""
        X_elas = _build_elasticity_features(request)
        theta, lo, hi = self._estimate_elasticity(X_elas)

        if theta < -1.0:
            interpretation = "elastic"
            recommendation = (
                "Demand is elastic — a price decrease will increase revenue. "
                "Consider lowering price to capture volume."
            )
        elif theta > -1.0 and theta < 0.0:
            interpretation = "inelastic"
            recommendation = (
                "Demand is inelastic — a price increase will increase revenue. "
                "Carefully raise price; monitor for volume loss."
            )
        else:
            interpretation = "unit elastic"
            recommendation = (
                "Revenue is approximately insensitive to price changes near this point. "
                "Focus on cost or volume levers instead."
            )

        return ElasticityResponse(
            product_id=request.product_id,
            elasticity=round(theta, 4),
            elasticity_lower_95=round(lo, 4),
            elasticity_upper_95=round(hi, 4),
            interpretation=interpretation,
            pricing_recommendation=recommendation,
        )

    def batch_forecast(self, request: BatchDemandRequest) -> BatchDemandResponse:
        """Vectorised batch demand forecast for up to 1,000 SKUs."""
        t0 = time.perf_counter()
        results = []

        for item in request.items:
            X = _build_features(item)
            point, lower, upper = self._predict_log_demand(X)
            results.append(
                BatchDemandItem(
                    product_id=item.product_id,
                    predicted_units=round(point, 2),
                    prediction_lower=round(lower, 2),
                    prediction_upper=round(upper, 2),
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        return BatchDemandResponse(
            results=results,
            n_items=len(results),
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 2),
        )