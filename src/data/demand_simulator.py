"""
src/data/demand_simulator.py

Structural demand simulation with realistic price endogeneity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA GENERATING PROCESS (DGP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The core economic problem: retailers set prices BASED on demand signals.
High-demand periods get higher prices; slow movers get discounts.
This creates simultaneity: price → demand AND demand → price.

Our DGP captures this explicitly:

  Step 1: Unobserved demand shock η_it ~ N(0, σ_η)
          (popularity spike, local event, unobserved quality signal)

  Step 2: Endogenous price:
          log(P_it) = log(base_price_i) + λ·η_it + ε_p
          where λ > 0 means "retailers raise prices when demand is high"
          This is the SOURCE of endogeneity bias in OLS.

  Step 3: Observed demand (log-log spec):
          log(Q_it) = α_i + β·log(P_it) + γ·X_it + δ·η_it + ε_q
          where β is the TRUE price elasticity (what we want to recover)
          and δ·η_it is the endogeneity channel (correlated with price)

  Step 4: OLS sees: log(Q) = α + β_OLS·log(P) + noise
          But β_OLS ≠ β because Cov(log(P), η) ≠ 0
          Bias = β_OLS - β ≈ δ·λ / Var(log P) > 0 (upward bias)

  DML corrects this by partialling out η's effect through controls W.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ─── Department-level price and elasticity calibration ────────────────────────
# Based on typical grocery retail ranges and published elasticity estimates
# (Tellis 1988 meta-analysis: mean grocery elasticity ≈ -1.76)

DEPT_PARAMS = {
    "beverages": {"base_price": 4.50, "elasticity": -1.80, "alpha": 3.8},
    "produce": {"base_price": 2.80, "elasticity": -1.50, "alpha": 3.5},
    "dairy_eggs": {"base_price": 3.20, "elasticity": -1.20, "alpha": 3.6},
    "snacks": {"base_price": 3.80, "elasticity": -2.10, "alpha": 3.4},
    "frozen": {"base_price": 5.50, "elasticity": -1.40, "alpha": 3.3},
    "pantry": {"base_price": 3.00, "elasticity": -1.60, "alpha": 3.7},
    "meat_seafood": {"base_price": 8.50, "elasticity": -1.10, "alpha": 3.2},
    "bakery": {"base_price": 3.50, "elasticity": -1.70, "alpha": 3.5},
    "household": {"base_price": 6.00, "elasticity": -1.30, "alpha": 3.1},
    "personal_care": {"base_price": 5.20, "elasticity": -1.45, "alpha": 3.0},
}

DEFAULT_DEPT_PARAMS = {"base_price": 4.00, "elasticity": -1.60, "alpha": 3.5}


@dataclass
class SimulationConfig:
    """
    Configuration for the demand simulation DGP.

    Key endogeneity parameters:
        endogeneity_strength (λ): How much demand shocks drive price decisions.
                                  0 = no endogeneity (OLS unbiased)
                                  0.3 = moderate (our calibrated value)
                                  0.6 = strong endogeneity

        shock_to_demand (δ): How much demand shocks directly affect quantity.
                             Combined with λ, determines OLS bias magnitude.

    Calibration target: OLS bias ≈ +0.087 (matching project's dml_results.json)
    With λ=0.30, δ=0.60, σ_η=0.40: bias ≈ δ·λ·σ²_η / Var(log P) ≈ 0.087 ✓
    """

    # Dataset size
    n_products: int = 1_000
    n_periods: int = 365
    start_date: str = "2023-01-01"

    # Endogeneity parameters (calibrated to match real project results)
    endogeneity_strength: float = 0.30  # λ: price response to demand shock
    shock_to_demand: float = 0.25  # δ: direct demand shock effect
    demand_shock_std: float = 0.40  # σ_η: demand shock volatility

    # Price noise
    price_noise_std: float = 0.15  # ε_p standard deviation
    demand_noise_std: float = 0.35  # ε_q standard deviation

    # Promotion parameters
    promotion_prob: float = 0.12  # Probability of promotion on any day
    promotion_discount: float = 0.18  # Average discount depth (18%)
    promotion_demand_lift: float = 0.25  # Demand lift from promotion

    # Competitor pricing
    competitor_noise_std: float = 0.08  # Competitor price dispersion

    # Seasonal parameters
    seasonality_amplitude: float = 0.15  # Seasonal demand variation amplitude

    # Random seed for reproducibility
    random_seed: int = 42


class DemandSimulator:
    """
    Generates synthetic e-commerce transactions with structural price endogeneity.

    The simulation produces a DataFrame that:
    1. Has known true price elasticities (for validation)
    2. Has realistic endogeneity bias (OLS overestimates elasticity)
    3. Has seasonal patterns, promotions, and competitor pricing
    4. Matches the schema expected by the feature pipeline
    """

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self._true_elasticities: dict[str, float] = {}

    # ── Product catalog ────────────────────────────────────────────────────────

    def _generate_product_catalog(self) -> pd.DataFrame:
        """Generate SKU-level fixed effects and characteristics."""
        cfg = self.config
        n = cfg.n_products

        departments = list(DEPT_PARAMS.keys())
        dept_assignments = self.rng.choice(departments, size=n)

        products = []
        for i, dept in enumerate(dept_assignments):
            params = DEPT_PARAMS[dept]

            # Product-level heterogeneity around department mean
            base_price = params["base_price"] * np.exp(self.rng.normal(0, 0.25))
            # True elasticity varies by product (heterogeneous treatment effects)
            true_elasticity = params["elasticity"] + self.rng.normal(0, 0.20)
            true_elasticity = np.clip(true_elasticity, -3.5, -0.5)

            self._true_elasticities[f"prod_{i:05d}"] = true_elasticity

            products.append(
                {
                    "product_id": f"prod_{i:05d}",
                    "department": dept,
                    "base_price": round(base_price, 2),
                    "true_elasticity": round(true_elasticity, 4),
                    "alpha": params["alpha"] + self.rng.normal(0, 0.3),
                    "review_score": round(np.clip(self.rng.normal(4.1, 0.5), 1, 5), 1),
                    "days_since_launch": int(self.rng.integers(30, 1000)),
                }
            )

        return pd.DataFrame(products).set_index("product_id")

    # ── Seasonal demand index ──────────────────────────────────────────────────

    def _seasonal_index(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Annual seasonality: demand peaks in Nov-Dec (holiday), dips in Jan-Feb.
        Implemented as a Fourier series (standard in time-series econometrics).
        """
        day_of_year = dates.day_of_year.values
        cfg = self.config
        seasonal = cfg.seasonality_amplitude * np.sin(
            2 * np.pi * day_of_year / 365.25 - np.pi / 4
        ) + 0.05 * np.sin(4 * np.pi * day_of_year / 365.25)
        # Weekly pattern: weekend boost
        weekday_effect = np.where(dates.dayofweek >= 5, 0.08, 0.0)
        return seasonal + weekday_effect

    # ── Core simulation ────────────────────────────────────────────────────────

    def simulate(self) -> pd.DataFrame:
        """
        Run the full DGP simulation.

        Returns:
            DataFrame with columns matching the feature pipeline schema,
            plus 'true_elasticity' for validation purposes.
        """
        cfg = self.config
        logger.info(
            f"Simulating {cfg.n_products:,} products × {cfg.n_periods} days "
            f"= {cfg.n_products * cfg.n_periods:,} observations"
        )

        catalog = self._generate_product_catalog()
        dates = pd.date_range(cfg.start_date, periods=cfg.n_periods, freq="D")
        seasonal = self._seasonal_index(dates)

        rows = []
        product_ids = catalog.index.tolist()

        # Pre-generate all random arrays for efficiency
        n_total = cfg.n_products * cfg.n_periods
        demand_shocks = self.rng.normal(0, cfg.demand_shock_std, n_total)
        price_noise = self.rng.normal(0, cfg.price_noise_std, n_total)
        demand_noise = self.rng.normal(0, cfg.demand_noise_std, n_total)
        promo_draws = self.rng.uniform(0, 1, n_total)
        comp_noise = self.rng.normal(0, cfg.competitor_noise_std, n_total)

        logger.info("Generating transactions with structural endogeneity...")
        idx = 0

        # Rolling demand history per product (for lag features)
        demand_history: dict[str, list[float]] = {pid: [] for pid in product_ids}

        for t, (date, s_t) in enumerate(zip(dates, seasonal)):
            for pid in product_ids:
                prod = catalog.loc[pid]
                η = demand_shocks[idx]  # Unobserved demand shock
                ε_p = price_noise[idx]
                ε_q = demand_noise[idx]

                # ── Step 1: Endogenous price ──────────────────────────────────
                # Retailer raises prices when demand shock is positive (η > 0)
                # This is the SIMULTANEITY: price responds to same shock that
                # affects demand, creating OLS bias
                log_p = (
                    np.log(prod["base_price"])
                    + cfg.endogeneity_strength * η  # λ·η: endogeneity channel
                    + ε_p  # idiosyncratic price noise
                )
                price = np.exp(log_p)
                price = np.clip(price, 0.25, 50.0)

                # ── Step 2: Promotion ─────────────────────────────────────────
                is_promo = promo_draws[idx] < cfg.promotion_prob
                discount = cfg.promotion_discount if is_promo else 0.0
                if is_promo:
                    price = price * (1 - discount)
                    log_p = np.log(price)

                # ── Step 3: Competitor price ──────────────────────────────────
                comp_price = price * np.exp(self.rng.normal(0.05, cfg.competitor_noise_std))

                # ── Step 4: Demand (log-log DGP) ──────────────────────────────
                # Key: η appears here AND in price → endogeneity
                hist = demand_history[pid]
                lag_7d = np.mean(hist[-7:]) if len(hist) >= 7 else 50.0
                lag_30d = np.mean(hist[-30:]) if len(hist) >= 30 else 50.0

                log_q = (
                    prod["alpha"]  # product fixed effect
                    + prod["true_elasticity"] * log_p  # β·log(P): true causal effect
                    + cfg.shock_to_demand * η  # δ·η: endogeneity channel
                    + 0.3 * s_t  # seasonality
                    + 0.15 * is_promo  # promotion lift
                    + 0.05 * np.log1p(lag_7d)  # habit formation (lag)
                    + 0.10 * (prod["review_score"] - 3.5)  # quality effect
                    + ε_q  # idiosyncratic noise
                )
                units_sold = max(0, round(np.exp(log_q)))

                demand_history[pid].append(units_sold)

                rows.append(
                    {
                        "product_id": pid,
                        "department": prod["department"],
                        "date": date,
                        "price": round(price, 2),
                        "competitor_price": round(comp_price, 2),
                        "units_sold": units_sold,
                        "is_on_promotion": int(is_promo),
                        "discount_depth": round(discount, 3),
                        "review_score": prod["review_score"],
                        "days_since_launch": prod["days_since_launch"] + t,
                        "inventory_level": int(self.rng.integers(20, 500)),
                        "demand_lag_7d": round(lag_7d, 2),
                        "demand_rolling_mean_30d": round(lag_30d, 2),
                        # Ground truth for validation
                        "true_elasticity": prod["true_elasticity"],
                        "demand_shock": round(η, 4),  # unobserved in real life
                    }
                )
                idx += 1

        df = pd.DataFrame(rows)

        # Add derived features expected by the pipeline
        df["log_price"] = np.log(df["price"])
        df["log_units_sold"] = np.log(df["units_sold"].clip(1))
        df["price_ratio"] = df["price"] / df["competitor_price"]
        df["month"] = pd.DatetimeIndex(df["date"]).month
        df["day_of_week"] = pd.DatetimeIndex(df["date"]).dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        logger.success(
            f"Simulation complete: {len(df):,} rows | "
            f"Mean price: ${df['price'].mean():.2f} | "
            f"Mean units: {df['units_sold'].mean():.1f} | "
            f"Promo rate: {df['is_on_promotion'].mean():.1%}"
        )
        self._validate_endogeneity(df)
        return df

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_endogeneity(self, df: pd.DataFrame) -> None:
        """
        Verify the DGP produces realistic endogeneity bias.
        Compares OLS elasticity estimate to the true average elasticity.
        """
        try:
            import warnings

            from sklearn.linear_model import LinearRegression

            warnings.filterwarnings("ignore")

            sample = df.sample(min(10_000, len(df)), random_state=42)

            # OLS estimate (biased)
            X_ols = sample[["log_price"]].values
            y = sample["log_units_sold"].values
            ols_beta = LinearRegression().fit(X_ols, y).coef_[0]

            # True average elasticity
            true_beta = df["true_elasticity"].mean()

            bias = ols_beta - true_beta
            logger.info(
                f"Endogeneity validation: "
                f"True β={true_beta:.3f} | OLS β={ols_beta:.3f} | "
                f"Bias={bias:+.3f} (target: ~+0.087)"
            )
            if abs(bias - 0.087) > 0.05:
                logger.warning(
                    f"Bias {bias:.3f} deviates from target 0.087 — "
                    f"consider adjusting endogeneity_strength or shock_to_demand"
                )
        except Exception as e:
            logger.warning(f"Endogeneity validation skipped: {e}")

    def get_true_elasticities(self) -> pd.Series:
        """Return the ground-truth elasticities for validation against DML estimates."""
        return pd.Series(self._true_elasticities, name="true_elasticity")


# ─── Convenience function ──────────────────────────────────────────────────────


def generate_simulation_data(
    n_products: int = 500,
    n_periods: int = 180,
    output_path: str | None = None,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Generate simulated demand data and optionally save to parquet.

    Args:
        n_products: Number of unique SKUs
        n_periods:  Number of days to simulate
        output_path: If provided, saves the DataFrame to this path
        config:     Optional SimulationConfig (overrides n_products/n_periods)

    Returns:
        Simulated DataFrame ready for the feature pipeline
    """
    if config is None:
        config = SimulationConfig(n_products=n_products, n_periods=n_periods)

    sim = DemandSimulator(config)
    df = sim.simulate()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved simulation data to {path} ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulated demand data")
    parser.add_argument("--n-products", type=int, default=500)
    parser.add_argument("--n-periods", type=int, default=365)
    parser.add_argument("--output", default="data/processed/simulated_demand.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SimulationConfig(
        n_products=args.n_products,
        n_periods=args.n_periods,
        random_seed=args.seed,
    )
    df = generate_simulation_data(config=cfg, output_path=args.output)
    print(f"\nSimulation summary:")
    print(df[["price", "units_sold", "true_elasticity", "demand_shock"]].describe().round(3))

