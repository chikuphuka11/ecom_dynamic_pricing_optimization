"""
src/api/schemas.py — Pydantic request/response models for the pricing API.
"""

from datetime import date as Date          # ✅ aliased — no clash
from typing import Optional
from pydantic import BaseModel, Field, field_validator
# ✅ NO 'from __future__ import annotations'

# ─── Shared Models ────────────────────────────────────────────────────────────

class ProductContext(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "prod_00123"})
    department: str = Field(..., example="beverages")
    aisle_id: Optional[str] = Field(None, example="coffee")
    price: float = Field(..., gt=0, example=4.99)
    pricing_date: Date = Field(..., alias="date", example="2024-11-01")  # ✅ field renamed, alias keeps API contract
    competitor_price: Optional[float] = Field(None, gt=0, example=5.29)
    inventory_level: Optional[int] = Field(None, ge=0, example=150)
    is_on_promotion: Optional[bool] = Field(False)
    discount_depth: Optional[float] = Field(0.0, ge=0.0, le=1.0, example=0.10)
    review_score: Optional[float] = Field(None, ge=1.0, le=5.0, example=4.2)
    days_since_launch: Optional[int] = Field(None, ge=0, example=365)
    demand_lag_7d: Optional[float] = Field(None, ge=0, example=45.2)
    demand_rolling_mean_30d: Optional[float] = Field(None, ge=0, example=42.8)

    model_config = {"populate_by_name": True}  # ✅ allows both 'pricing_date' and 'date' to work


# ─── Demand Forecast ──────────────────────────────────────────────────────────

class DemandForecastRequest(ProductContext):
    pass


class DemandForecastResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # ✅ add this line
    product_id: str
    price: float
    predicted_units: float = Field(...)
    prediction_lower: float = Field(...)
    prediction_upper: float = Field(...)
    model_version: str


# ─── Optimal Price ────────────────────────────────────────────────────────────

class OptimalPriceRequest(ProductContext):
    unit_cost: float = Field(..., gt=0, example=2.10,
        description="Unit cost (COGS) — used for margin constraint")
    min_margin_pct: float = Field(0.10, ge=0, le=1.0,
        description="Minimum acceptable gross margin (e.g., 0.10 = 10%)")
    max_price_change_pct: float = Field(0.30, ge=0, le=1.0,
        description="Maximum price change from current price (guardrail)")
    objective: str = Field("revenue", pattern="^(revenue|profit|gmv)$",
        description="Optimization objective: revenue, profit, or GMV")

    @field_validator("unit_cost")
    @classmethod
    def cost_less_than_price(cls, v, info):
        if "price" in info.data and v >= info.data["price"] * 0.95:
            raise ValueError("Unit cost must be less than current price (no negative margin)")
        return v


class OptimalPriceResponse(BaseModel):
    product_id: str
    current_price: float
    optimal_price: float
    price_change_pct: float = Field(..., description="Recommended price change as fraction")
    expected_revenue_lift_pct: float
    expected_demand_change_pct: float
    estimated_elasticity: float
    constraint_binding: str = Field(...,
        description="Which constraint (if any) limited the optimization: 'margin', 'guardrail', 'none'")
    confidence: str = Field(..., description="'high' | 'medium' | 'low' based on data quality")


# ─── Elasticity ───────────────────────────────────────────────────────────────

class ElasticityRequest(BaseModel):
    product_id: str
    department: str
    month: int = Field(..., ge=1, le=12)
    is_on_promotion: bool = False
    demand_rolling_mean_30d: Optional[float] = None


class ElasticityResponse(BaseModel):
    product_id: str
    elasticity: float = Field(...,
        description="Price elasticity of demand θ = ∂log(Q)/∂log(P). Negative for normal goods.")
    elasticity_lower_95: float
    elasticity_upper_95: float
    interpretation: str = Field(...,
        description="Human-readable interpretation: 'elastic', 'inelastic', or 'unit elastic'")
    pricing_recommendation: str


# ─── Batch Forecast ───────────────────────────────────────────────────────────

class BatchDemandRequest(BaseModel):
    items: list[DemandForecastRequest] = Field(..., max_length=1000)


class BatchDemandItem(BaseModel):
    product_id: str
    predicted_units: float
    prediction_lower: float
    prediction_upper: float


class BatchDemandResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # ✅ add here too
    results: list[BatchDemandItem]
    n_items: int
    model_version: str
    latency_ms: float
