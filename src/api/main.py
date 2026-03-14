"""
src/api/main.py

FastAPI application for demand forecasting and optimal price recommendations.

Endpoints:
  POST /demand-forecast  — predict units sold at a given price
  POST /optimal-price    — return revenue-maximizing price within guardrails
  POST /elasticity       — return price elasticity estimate for a SKU + context
  GET  /health           — health check
  GET  /metrics          — Prometheus metrics
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from api.schemas import (
    DemandForecastRequest,
    DemandForecastResponse,
    OptimalPriceRequest,
    OptimalPriceResponse,
    ElasticityRequest,
    ElasticityResponse,
    BatchDemandRequest,
    BatchDemandResponse,
)
from api.predict import DemandPredictor

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "pricing_api_requests_total",
    "Total API requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "pricing_api_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
PREDICTION_VALUE = Histogram(
    "demand_prediction_units",
    "Distribution of demand predictions",
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# ─── Application State ────────────────────────────────────────────────────────
predictor: DemandPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global predictor
    logger.info("Loading models...")
    try:
        predictor = DemandPredictor.load(
            model_path="models/production/demand_model.pkl",
            pipeline_path="models/production/preprocessing_pipeline.pkl",
            dml_path="models/production/dml_model.pkl",
        )
        logger.success("Models loaded successfully")
    except FileNotFoundError:
        logger.warning("Model files not found — running in demo mode")
        predictor = DemandPredictor.demo()
    yield
    logger.info("Shutting down API")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="E-Commerce Dynamic Pricing API",
    description="""
    Demand forecasting and optimal pricing recommendations using
    causal ML (Double Machine Learning) for unbiased price elasticity estimation.

    **Economics background**: The demand forecast uses a log-log specification
    so coefficients are interpretable as price elasticities. The DML endpoint
    corrects for endogeneity using the Frisch-Waugh-Lovell theorem.
    """,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware: Request Logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    logger.info(f"{request.method} {endpoint} → {response.status_code} ({duration*1000:.1f}ms)")
    return response


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check — used by Docker and Kubernetes liveness probes."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "version": app.version,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/demand-forecast", response_model=DemandForecastResponse)
async def demand_forecast(request: DemandForecastRequest):
    """
    Predict units sold given price and context features.

    Returns point estimate + 90% prediction interval.
    The model is a LightGBM trained in log-log space, so
    predictions are transformed back to units via exp().
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.forecast_demand(request)
        PREDICTION_VALUE.observe(result.predicted_units)
        return result
    except Exception as e:
        logger.error(f"Demand forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimal-price", response_model=OptimalPriceResponse)
async def optimal_price(request: OptimalPriceRequest):
    """
    Find the revenue-maximizing price subject to constraints.

    Uses scipy.optimize.minimize on the demand model's revenue function:
        Revenue(P) = P × Q(P) = P × exp(α + β·log(P) + controls)

    Subject to:
        - Margin constraint: P ≥ cost × (1 + min_margin)
        - Guardrail: |P - current_price| / current_price ≤ max_change_pct
        - Non-negativity: P > 0

    Economics note: The unconstrained optimum is at dR/dP = 0, which gives
    P* = MC × ε/(ε+1) — the standard monopoly pricing formula.
    Our optimizer finds this numerically, allowing the demand model to be
    arbitrarily complex (non-constant elasticity).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.find_optimal_price(request)
        return result
    except Exception as e:
        logger.error(f"Price optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/elasticity", response_model=ElasticityResponse)
async def get_elasticity(request: ElasticityRequest):
    """
    Return the estimated price elasticity of demand for this SKU + context.

    Returns the CATE from the DML model: θ(x) = ∂E[log Q | x] / ∂log P

    Interpretation:
      - elasticity = -1.5 → a 10% price increase reduces demand by 15% (elastic)
      - elasticity = -0.3 → a 10% price increase reduces demand by 3% (inelastic)
      - elasticity > 0    → WARNING: positive elasticity (possible data issue or Giffen good)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return predictor.get_elasticity(request)
    except Exception as e:
        logger.error(f"Elasticity estimation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/demand-forecast/batch", response_model=BatchDemandResponse)
async def batch_demand_forecast(request: BatchDemandRequest):
    """
    Batch demand forecasting for up to 1,000 SKUs simultaneously.
    Used for portfolio-level pricing optimization.
    """
    if len(request.items) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds maximum of 1,000 items"
        )
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return predictor.batch_forecast(request)
    except Exception as e:
        logger.error(f"Batch forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
