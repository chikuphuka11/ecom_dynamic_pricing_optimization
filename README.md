# 🛒 E-Commerce Dynamic Pricing Optimization
### End-to-End MLOps Project | Causal ML + Price Elasticity + FastAPI Deployment

> **Background note:** This project is intentionally designed to leverage my BA in Economics & Business training.
> Price elasticity, endogeneity correction, and causal inference are Economics concepts first —
> the ML is the tooling that makes them scalable.

---

## 📐 Project Architecture

```
ecom_pricing/
├── data/
│   ├── raw/               # Original, immutable data (versioned with DVC)
│   ├── processed/         # Cleaned, feature-engineered datasets
│   └── external/          # Competitor prices, macroeconomic data
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb    # Feature construction & pipelines
│   ├── 03_baseline_models.ipynb        # OLS, log-log elasticity models
│   ├── 04_causal_ml_dml.ipynb          # Double Machine Learning (EconML)
│   ├── 05_deep_learning_dcn.ipynb      # Deep & Cross Network (PyTorch)
│   ├── 06_hyperparameter_tuning.ipynb  # Optuna + multi-objective HPO
│   ├── 07_pricing_optimizer.ipynb      # Downstream price optimization
│   └── 08_model_evaluation.ipynb       # SHAP, calibration, A/B simulation
│
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Scikit-learn preprocessing pipelines
│   │   ├── engineering.py     # Domain feature creation functions
│   │   └── feature_store.py   # Feast feature store integration
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── elasticity_ols.py  # Log-log OLS baseline
│   │   ├── causal_dml.py      # Double ML via EconML
│   │   ├── dcn_model.py       # Deep & Cross Network (PyTorch)
│   │   └── optimizer.py       # Constrained price optimizer
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI application
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── predict.py         # Inference logic
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── drift_detector.py  # Evidently AI drift reports
│       └── metrics_logger.py  # Prometheus metrics
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_guardrails.py     # Pricing guardrail validation
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── configs/
│   ├── config.yaml            # Project-wide configuration
│   ├── model_config.yaml      # Model hyperparameters
│   └── monitoring_config.yaml # Drift thresholds
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml          # GitHub Actions pipeline
│
├── .vscode/
│   ├── settings.json          # VS Code workspace settings
│   ├── extensions.json        # Recommended extensions
│   └── launch.json            # Debug configurations
│
├── docs/
│   └── economics_primer.md    # Elasticity & causal ML concepts
│
├── .env.example               # Environment variable template
├── .gitignore
├── dvc.yaml                   # DVC pipeline definition
├── pyproject.toml             # Project metadata & tool config
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── setup.py                   # Package installation
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.11+
- VS Code with recommended extensions (see `.vscode/extensions.json`)
- Docker Desktop
- Git

### 1. Clone & Environment Setup
```bash
# Clone your repo
git clone https://github.com/chikuphuka11/ecom_dynamic_pricing_optimisation.git
cd ecom-dynamic-pricing

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .                 # Install project as editable package
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (Kaggle, MLflow tracking URI, etc.)
```

### 3. Pull Data with DVC
```bash
dvc pull                         # Pulls data from configured remote
# OR to download fresh from Kaggle:
python scripts/download_data.py
```

### 4. Run Notebooks in Order
```
notebooks/01_eda.ipynb  →  02  →  03  →  04  →  05  →  06  →  07  →  08
```

### 5. Start MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 6. Launch API locally
```bash
uvicorn src.api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## 🧠 Economics Background — Why This Project Suits You

| Economics Concept | How It Appears in This Project |
|---|---|
| **Price Elasticity of Demand** | The regression target — we model ∂log(Q)/∂log(P) |
| **Endogeneity** | Prices are set based on demand → OLS is biased → we use Double ML |
| **Instrumental Variables** | DML's first stage is analogous to 2SLS — partials out confounders |
| **Marginal Revenue = Marginal Cost** | The optimizer finds this point subject to constraints |
| **Consumer Surplus** | Monitored as a fairness metric — we don't extract all surplus |
| **Market Structure** | Competitor pricing features capture oligopolistic interdependence |
| **Revealed Preference** | Transaction data reveals willingness-to-pay empirically |

---

## 📊 Key Results (Target Benchmarks)

| Metric | Baseline (Static Pricing) | This Model |
|---|---|---|
| MAPE on demand forecast | — | < 12% |
| Revenue lift (A/B test) | 0% | +3–5% |
| Price update frequency | Manual (weekly) | Every 15 min |
| SKUs covered | Top 1,000 only | All active SKUs |
| Overstock reduction | — | ~8% |

---

## 📚 References

- Chernozhukov et al. (2018) — *Double/Debiased Machine Learning* (the DML paper)
- Wang & Rossi (2019) — *Causal Inference for Pricing Analytics*
- EconML Documentation — https://econml.azurewebsites.net/
- Kaggle Instacart Dataset — https://www.kaggle.com/c/instacart-market-basket-analysis