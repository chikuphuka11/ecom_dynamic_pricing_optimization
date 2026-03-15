"""
tests/test_api.py — Tests for the FastAPI pricing endpoints.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from datetime import date

from api.main import app

client = TestClient(app)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_product():
    return {
        "product_id": "prod_00123",
        "department": "beverages",
        "aisle_id": "coffee",
        "price": 4.99,
        "date": "2024-11-01",
        "competitor_price": 5.29,
        "inventory_level": 150,
        "is_on_promotion": False,
        "discount_depth": 0.0,
        "review_score": 4.2,
        "days_since_launch": 365,
        "demand_lag_7d": 45.2,
        "demand_rolling_mean_30d": 42.8,
    }


# ─── Health ───────────────────────────────────────────────────────────────────

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


# ─── Demand Forecast ──────────────────────────────────────────────────────────

def test_demand_forecast_returns_200(sample_product):
    response = client.post("/demand-forecast", json=sample_product)
    assert response.status_code == 200


def test_demand_forecast_response_structure(sample_product):
    response = client.post("/demand-forecast", json=sample_product)
    data = response.json()
    assert "predicted_units" in data
    assert "prediction_lower" in data
    assert "prediction_upper" in data
    assert "model_version" in data


def test_demand_forecast_prediction_interval_valid(sample_product):
    """Prediction interval must be ordered: lower ≤ prediction ≤ upper."""
    response = client.post("/demand-forecast", json=sample_product)
    data = response.json()
    assert data["prediction_lower"] <= data["predicted_units"] <= data["prediction_upper"]


def test_demand_forecast_non_negative_prediction(sample_product):
    """Demand can never be negative — units must be ≥ 0."""
    response = client.post("/demand-forecast", json=sample_product)
    data = response.json()
    assert data["predicted_units"] >= 0
    assert data["prediction_lower"] >= 0


def test_demand_forecast_invalid_price(sample_product):
    """Negative price should return 422 Unprocessable Entity."""
    sample_product["price"] = -1.0
    response = client.post("/demand-forecast", json=sample_product)
    assert response.status_code == 422


# ─── Optimal Price ────────────────────────────────────────────────────────────

def test_optimal_price_returns_200(sample_product):
    payload = {**sample_product, "unit_cost": 2.10, "min_margin_pct": 0.10}
    response = client.post("/optimal-price", json=payload)
    assert response.status_code == 200


def test_optimal_price_respects_guardrail(sample_product):
    """Optimal price must not change more than max_price_change_pct."""
    payload = {
        **sample_product,
        "unit_cost": 2.10,
        "min_margin_pct": 0.10,
        "max_price_change_pct": 0.20,  # ±20% guardrail
    }
    response = client.post("/optimal-price", json=payload)
    data = response.json()
    assert abs(data["price_change_pct"]) <= 0.20 + 1e-6  # tolerance for floating point


def test_optimal_price_respects_margin(sample_product):
    """Optimal price must maintain the minimum margin."""
    unit_cost = 2.10
    min_margin = 0.10
    payload = {
        **sample_product,
        "unit_cost": unit_cost,
        "min_margin_pct": min_margin,
    }
    response = client.post("/optimal-price", json=payload)
    data = response.json()
    optimal = data["optimal_price"]
    actual_margin = (optimal - unit_cost) / optimal
    assert actual_margin >= min_margin - 1e-6


# ─── Batch Forecast ───────────────────────────────────────────────────────────

def test_batch_forecast_returns_correct_count(sample_product):
    payload = {"items": [sample_product] * 5}
    response = client.post("/demand-forecast/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_items"] == 5
    assert len(data["results"]) == 5


def test_batch_forecast_rejects_oversized_request(sample_product):
    """Batches > 1000 items should be rejected."""
    payload = {"items": [sample_product] * 1001}
    response = client.post("/demand-forecast/batch", json=payload)
    assert response.status_code == 422
