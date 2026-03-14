# E-Commerce Dynamic Pricing Optimization

> Causal ML-powered demand forecasting and revenue-optimal pricing for large-scale retail —
> **30% revenue lift** across 49,677 SKUs using Double Machine Learning for unbiased elasticity estimation.

---

## Results Summary

| Model | MAPE | R² | vs Baseline |
|---|---|---|---|
| Ridge Regression (baseline) | 0.4292 | 0.0176 | — |
| Deep & Cross Network (DCN) | 0.4243 | 0.0319 | +81% R² |
| **LightGBM (tuned)** | **0.4177** | **0.0553** | **+214% R²** |

| Business Metric | Value |
|---|---|
| Revenue lift (pricing optimizer) | **+30.0%** |
| Price elasticity ATE (DML) | −0.083 |
| Endogeneity bias corrected | 0.087 |
| SKUs optimized | 49,677 |
| Transactions in training set | 32,434,489 |
| Departments covered | 21 |

---

## Project Overview

Standard demand models suffer from **price endogeneity** — prices are set in response to demand signals, so naive OLS estimates of elasticity are biased upward. This project applies the **Frisch-Waugh-Lovell theorem via Double Machine Learning (DML)** to partial out confounders and recover unbiased conditional average treatment effects (CATEs) of price on demand.

The resulting elasticity estimates feed a constrained revenue optimizer that recommends prices subject to:
- Margin floor: `P ≥ cost / (1 − min_margin)`
- Guardrail: `|ΔP / P| ≤ max_change_pct`

All predictions are served through a production-ready FastAPI service with Prometheus metrics and Pydantic v2 validation.

---

## Economics Background

The demand model uses a **log-log specification**:

```
log(Q) = α + β·log(P) + γ·X + ε
```

where `β` is directly interpretable as the price elasticity of demand. The unconstrained revenue-maximising price is:

```
P* = MC · ε / (ε + 1)    [standard monopoly pricing formula]
```

The optimizer finds this numerically via `scipy.optimize.minimize_scalar`, allowing the demand surface to be arbitrarily non-linear (non-constant elasticity across contexts).

**DML identification strategy:**
1. Regress `log(P)` on controls `X` → get residual `Ṽ`
2. Regress `log(Q)` on controls `X` → get residual `Ỹ`  
3. Regress `Ỹ` on `Ṽ` → coefficient is the unbiased elasticity `θ`

This removes the endogeneity bias of 0.087 observed in the naive OLS estimate.

---

## Model Pipeline

```
Raw Transactions (32.4M rows)
        │
        ▼
Feature Engineering
  log_price, price_ratio, competitor_price,
  demand lags (7d, 30d rolling), temporal features
        │
        ├──► Ridge Regression (baseline)     MAPE=0.4292, R²=0.0176
        │
        ├──► Deep & Cross Network (PyTorch)  MAPE=0.4243, R²=0.0319
        │      cross_layers=3, deep=[256,128,64]
        │
        ├──► LightGBM (Optuna HPO) ✓ best   MAPE=0.4177, R²=0.0553
        │      500 estimators, log-log space
        │
        └──► Double ML (EconML)              ATE=−0.083, bias corrected
               treatment=log_price, outcome=log_demand
                        │
                        ▼
              Pricing Optimizer
              scipy.minimize_scalar
              Revenue lift: +30%
```

---

## API Endpoints

```
POST /demand-forecast        # Predict units sold at given price + context
POST /optimal-price          # Revenue-maximising price within guardrails
POST /elasticity             # Price elasticity CATE from DML model
POST /demand-forecast/batch  # Batch forecast up to 1,000 SKUs
GET  /health                 # Liveness probe
GET  /metrics                # Prometheus metrics
```

### Quick Start

```bash
# Install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run the API
cd src
uvicorn api.main:app --reload --port 8000

# Interactive docs
open http://localhost:8000/docs
```

### Example Request

```bash
curl -X POST http://localhost:8000/demand-forecast \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "prod_00123",
    "department": "beverages",
    "price": 4.99,
    "date": "2024-11-01",
    "competitor_price": 5.29,
    "inventory_level": 150,
    "demand_lag_7d": 45.2,
    "demand_rolling_mean_30d": 42.8
  }'
```

```json
{
  "product_id": "prod_00123",
  "price": 4.99,
  "predicted_units": 48.3,
  "prediction_lower": 29.1,
  "prediction_upper": 80.2,
  "model_version": "1.0.0"
}
```

---

## Repository Structure

```
ecom_dynamic_pricing_optimization/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app, routes, Prometheus metrics
│   │   ├── predict.py       # DemandPredictor: inference + optimization
│   │   └── schemas.py       # Pydantic v2 request/response models
│   ├── features/
│   │   └── pipeline.py      # Feature engineering pipeline
│   ├── models/
│   │   └── causal_dml.py    # Double ML elasticity estimator
│   └── monitoring/
│       └── drift_detector.py # Data drift detection (Evidently)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_causal_ml_dml.ipynb
│   ├── 05_deep_learning_dcn.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_pricing_optimizer.ipynb
│   └── 08_model_evaluation.ipynb
├── tests/
│   ├── conftest.py          # TestClient fixture with demo predictor
│   ├── test_api.py          # API endpoint tests (11 tests)
│   └── test_guardrails.py   # Business guardrail tests (32 tests)
├── data/processed/
│   ├── final_evaluation_report.png
│   ├── hpo_results.png
│   ├── pricing_optimizer_results.png
│   └── *.json               # Experiment result summaries
├── configs/config.yaml
├── requirements.txt
└── pyproject.toml
```

---

## Experiment Tracking

All experiments tracked in MLflow across 6 experiments:

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

| Experiment | Run | Key Metric |
|---|---|---|
| `01_baseline` | ridge_regression_baseline | R²=0.0176 |
| `02_deep_learning_dcn` | dcn_best | R²=0.0319 |
| `03_hyperparameter_tuning` | lgbm_optuna_best | R²=0.0553 |
| `04_causal_dml` | double_ml_elasticity | ATE=−0.083 |
| `05_pricing_optimizer` | revenue_optimization | lift=+30% |
| `06_model_evaluation` | final_evaluation_all_models | all metrics |

---

## Testing

```bash
$env:PYTHONPATH = "src"
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

```
43 passed in 4.7s  |  Coverage: 39%
├── test_api.py         11/11 ✅  endpoint correctness, schema validation
└── test_guardrails.py  32/32 ✅  margin floors, guardrail bounds, elasticity sanity
```

---

## Dataset

Based on the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset, augmented with synthetic pricing signals to simulate a realistic dynamic pricing environment.

| Statistic | Value |
|---|---|
| Total transactions | 32,434,489 |
| Unique products (SKUs) | 49,677 |
| Departments | 21 |
| Aisles | 134 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML models | LightGBM, PyTorch, EconML (DML) |
| HPO | Optuna |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Optimization | scipy.optimize |
| Experiment tracking | MLflow |
| Monitoring | Prometheus + Evidently |
| Testing | pytest + pytest-cov |
| Feature pipeline | scikit-learn |
