"""
src/models/causal_dml.py

Double Machine Learning (DML) for unbiased price elasticity estimation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ECONOMICS PRIMER — WHY OLS FAILS FOR PRICING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In standard demand estimation, we want: log(Q) = α + β·log(P) + ε
where β is the price elasticity of demand.

The problem: OLS gives biased estimates of β because price (P) is
ENDOGENOUS — it's set by us based on expected demand. When we raise
prices on popular items (to capture surplus), or cut prices to clear
slow-moving inventory, the error term ε correlates with P.

This is the classic SIMULTANEITY BIAS you study in econometrics.
E[ε | P] ≠ 0 → OLS E[β̂] ≠ β (inconsistent estimator)

SOLUTION: Double Machine Learning (Chernozhukov et al., 2018)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DML is essentially a high-dimensional generalization of the
Frisch-Waugh-Lovell (FWL) theorem + 2SLS intuition:

Step 1 (Nuisance regression 1): Regress Y on controls W using ML
        Ỹ = Y - E[Y | W]   ← demand residual (net of controls)

Step 2 (Nuisance regression 2): Regress T on controls W using ML
        T̃ = T - E[T | W]   ← price residual (net of controls)

Step 3 (Final regression): Regress Ỹ on T̃
        Ỹ = θ · T̃ + ε
        θ̂ = Cov(Ỹ, T̃) / Var(T̃) ← unbiased elasticity estimate

The FWL theorem guarantees: θ̂ from step 3 equals the coefficient
from OLS[Y ~ T + W] when relationships are linear.
DML extends this to nonlinear/high-dimensional W using cross-fitting
(k-fold split) to avoid overfitting bias from the nuisance models.

Analogy to 2SLS: T̃ = T - E[T|W] plays the role of the instrument —
variation in price UNEXPLAINED by demand conditions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from econml.dml import LinearDML, NonParamDML, CausalForestDML
from econml.inference import BootstrapInference
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score
import shap

from features.pipeline import (
    DML_CONTROL_FEATURES,
    DML_TREATMENT,
    DML_OUTCOME,
)


class PriceElasticityModel:
    """
    Wrapper around EconML's LinearDML for price elasticity estimation.

    This class implements the Double ML estimator with:
    - LightGBM nuisance models (fast, handles nonlinearity & interactions)
    - Cross-fitting with k=5 folds (reduces regularization bias)
    - Heterogeneous treatment effects (HTE): elasticity varies by category,
      season, promotion status — this is the key insight over simple OLS

    After fitting, self.effect(X) returns the CATE (Conditional Average
    Treatment Effect), i.e., the price elasticity conditional on features X.
    This answers: "How elastic are customers for THIS product on THIS day?"
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: LinearDML | None = None
        self._ate: float | None = None
        self._ate_stderr: float | None = None

    def _build_nuisance_model(self) -> LGBMRegressor:
        """LightGBM for both nuisance regressions (fast, regularized)."""
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self,
        df: pd.DataFrame,
        heterogeneity_features: list[str] | None = None,
    ) -> PriceElasticityModel:
        """
        Fit the Double ML estimator.

        Args:
            df: DataFrame with all features, treatment, and outcome
            heterogeneity_features: Features that moderate the elasticity.
                                    If None, estimates ATE (constant elasticity).
                                    If provided, estimates CATE (heterogeneous).

        Returns:
            self (fitted model)
        """
        logger.info("Preparing DML inputs...")

        # Outcome: log(units_sold) — log-log spec gives direct elasticity interpretation
        Y = df[DML_OUTCOME].values

        # Treatment: log(price) — β in log(Q) = α + β·log(P) + ... is the elasticity
        T = df[DML_TREATMENT].values

        # Controls (confounders W): demand shifters that also correlate with price
        # Economics: these are factors that affect BOTH supply decisions (pricing)
        # and demand — e.g., promotions affect price AND quantity simultaneously
        W = df[DML_CONTROL_FEATURES].values

        # Optional heterogeneity features X (interact with treatment)
        # This tests: does price sensitivity differ by product category?
        X = df[heterogeneity_features].values if heterogeneity_features else None

        model_y = self._build_nuisance_model()   # E[Y | W, X] — demand given controls
        model_t = self._build_nuisance_model()   # E[T | W, X] — price given controls

        # LinearDML: linear final stage (interpretable, with inference)
        # NonParamDML: nonlinear final stage (more flexible but less interpretable)
        self.model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            cv=self.n_splits,
            discrete_treatment=False,
            random_state=self.random_state,
        )

        logger.info(
            f"Fitting DML with {self.n_splits}-fold cross-fitting, "
            f"n={len(Y):,} observations..."
        )
        self.model.fit(Y, T, X=X, W=W,
                       inference=BootstrapInference(n_bootstrap_samples=100))

        # ATE summary (Average Treatment Effect — the average elasticity)
        ate_result = self.model.ate_inference(X=X)
        self._ate = float(ate_result.mean_point)
        self._ate_stderr = float(ate_result.stderr_mean)

        logger.success(
            f"DML fitted. ATE (average elasticity) = {self._ate:.3f} "
            f"(stderr = {self._ate_stderr:.3f}). "
            f"95% CI: [{self._ate - 1.96*self._ate_stderr:.3f}, "
            f"{self._ate + 1.96*self._ate_stderr:.3f}]"
        )
        if self._ate > 0:
            logger.warning(
                "Positive elasticity estimate — check for Giffen good or data issue!"
            )
        elif abs(self._ate) < 0.5:
            logger.info("Inelastic demand (|ε| < 0.5) — pricing power is HIGH.")
        elif abs(self._ate) > 1.5:
            logger.info("Elastic demand (|ε| > 1.5) — pricing power is LOW.")

        return self

    def predict_elasticity(
        self, X: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Predict CATE (Conditional Average Treatment Effect = local elasticity).

        For a given set of context features X, returns the estimated elasticity:
        θ(x) = ∂E[log Q | x] / ∂log P

        Args:
            X: Context features for heterogeneous elasticity (n_samples × n_features)

        Returns:
            Array of elasticity estimates, one per sample
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self.model.effect(X)

    def elasticity_summary(self, X: np.ndarray | None = None) -> dict:
        """Return summary statistics of estimated elasticities."""
        elasticities = self.predict_elasticity(X)
        return {
            "ate": self._ate,
            "ate_stderr": self._ate_stderr,
            "ate_95ci_lower": self._ate - 1.96 * self._ate_stderr,
            "ate_95ci_upper": self._ate + 1.96 * self._ate_stderr,
            "cate_mean": float(np.mean(elasticities)),
            "cate_std": float(np.std(elasticities)),
            "cate_p10": float(np.percentile(elasticities, 10)),
            "cate_p50": float(np.percentile(elasticities, 50)),
            "cate_p90": float(np.percentile(elasticities, 90)),
            "n_elastic": int(np.sum(elasticities < -1)),    # |ε| > 1
            "n_inelastic": int(np.sum(elasticities >= -1)), # |ε| < 1
        }

    def log_to_mlflow(self, run_name: str = "dml_elasticity"):
        """Log model and metrics to MLflow."""
        with mlflow.start_run(run_name=run_name):
            summary = self.elasticity_summary()
            mlflow.log_params({
                "n_splits": self.n_splits,
                "n_estimators": self.n_estimators,
                "model_type": "LinearDML",
                "treatment": DML_TREATMENT,
                "outcome": DML_OUTCOME,
            })
            mlflow.log_metrics({
                "ate": summary["ate"],
                "ate_stderr": summary["ate_stderr"],
                "cate_p50_elasticity": summary["cate_p50"],
                "pct_elastic_skus": summary["n_elastic"] / (
                    summary["n_elastic"] + summary["n_inelastic"]
                ),
            })
            mlflow.sklearn.log_model(self.model, "dml_model")
            logger.info(f"Logged DML model to MLflow run: {run_name}")


def train_dml_model(
    data_path: str,
    heterogeneity_features: list[str] | None = None,
    experiment_name: str = "ecom-dynamic-pricing",
) -> PriceElasticityModel:
    """
    End-to-end DML training function (called from CLI or notebook).

    Args:
        data_path: Path to processed Parquet file with features
        heterogeneity_features: Optional features for CATE estimation
        experiment_name: MLflow experiment name

    Returns:
        Fitted PriceElasticityModel
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    mlflow.set_experiment(experiment_name)

    model = PriceElasticityModel(n_splits=5)
    model.fit(df, heterogeneity_features=heterogeneity_features)
    model.log_to_mlflow()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DML price elasticity model")
    parser.add_argument("--data", default="data/processed/train.parquet")
    parser.add_argument("--experiment", default="ecom-dynamic-pricing")
    args = parser.parse_args()

    model = train_dml_model(
        data_path=args.data,
        heterogeneity_features=["department", "month", "is_on_promotion"],
        experiment_name=args.experiment,
    )

    summary = model.elasticity_summary()
    print("\n" + "="*60)
    print("DML PRICE ELASTICITY RESULTS")
    print("="*60)
    print(f"Average Treatment Effect (ATE): {summary['ate']:.3f}")
    print(f"95% Confidence Interval: [{summary['ate_95ci_lower']:.3f}, {summary['ate_95ci_upper']:.3f}]")
    print(f"Median CATE: {summary['cate_p50']:.3f}")
    print(f"CATE std: {summary['cate_std']:.3f}")
    print(f"Elastic SKUs (|ε|>1): {summary['n_elastic']:,}")
    print(f"Inelastic SKUs: {summary['n_inelastic']:,}")
    print("="*60)
