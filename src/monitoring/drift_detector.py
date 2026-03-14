"""
src/monitoring/drift_detector.py

Data drift and model performance monitoring using Evidently AI.

Why drift monitoring matters (Economics angle):
────────────────────────────────────────────────
A demand model trained on pre-COVID data will completely fail post-COVID
because the underlying preferences, substitution patterns, and income
effects all shifted. This is "concept drift" — the mapping from features
to demand has changed, not just the input distribution.

We monitor two types of drift:

1. DATA DRIFT (covariate shift): The distribution of input features P(X) changes.
   Example: competitor_price_gap_pct shifts because a new competitor entered.
   Detection: PSI (Population Stability Index) on each feature.
   Threshold: PSI > 0.20 → retrain.

2. CONCEPT DRIFT (target shift): The relationship P(Y|X) changes.
   Example: price sensitivity increased because disposable income fell.
   Detection: Monitor model MAPE on rolling windows of new actuals.
   Threshold: MAPE > 25% → retrain.

PSI Interpretation (industry standard from credit risk modeling):
   PSI < 0.10: No significant change
   0.10–0.20:  Moderate change — investigate
   PSI > 0.20: Significant change — retrain required
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    RegressionQualityMetric,
    RegressionErrorPlot,
    RegressionPredictedVsActualScatter,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAllFeaturesValueDrift,
    TestValueMeanInNSigmas,
    TestColumnValueDrift,
)
from loguru import logger
import yaml


class DriftDetector:
    """
    Monitors data drift and model performance degradation.

    Usage:
        detector = DriftDetector.from_config("configs/monitoring_config.yaml")
        report = detector.run_drift_report(reference_df, current_df)
        if detector.should_retrain(report):
            trigger_retraining_pipeline()
    """

    def __init__(
        self,
        feature_columns: list[str],
        target_col: str = "units_sold",
        prediction_col: str = "predicted_units",
        psi_warning_threshold: float = 0.10,
        psi_alert_threshold: float = 0.20,
        mape_warning_threshold: float = 0.15,
        mape_alert_threshold: float = 0.25,
        reports_dir: str = "monitoring/reports",
    ):
        self.feature_columns = feature_columns
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.psi_warning = psi_warning_threshold
        self.psi_alert = psi_alert_threshold
        self.mape_warning = mape_warning_threshold
        self.mape_alert = mape_alert_threshold
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Column mapping tells Evidently which columns are features, target, prediction
        self.column_mapping = ColumnMapping(
            target=target_col,
            prediction=prediction_col,
            numerical_features=[c for c in feature_columns if c not in
                                  ["department", "aisle_id", "product_id"]],
            categorical_features=["department", "aisle_id"],
        )

    @classmethod
    def from_config(cls, config_path: str) -> DriftDetector:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        m = config["monitoring"]
        f = config["features"]
        return cls(
            feature_columns=f.get("numeric_features", []) + f.get("categorical_features", []),
            target_col=f["target_col"],
            psi_warning_threshold=m["psi_warning_threshold"],
            psi_alert_threshold=m["psi_alert_threshold"],
            mape_warning_threshold=m["mape_warning_threshold"],
            mape_alert_threshold=m["mape_alert_threshold"],
            reports_dir=m["reports_dir"],
        )

    def run_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        save: bool = True,
    ) -> dict:
        """
        Generate a full drift report comparing reference and current data.

        Args:
            reference_df: Training data distribution (the "baseline")
            current_df:   Recent production data (what the model is seeing now)
            save:         Whether to save HTML and JSON reports to disk

        Returns:
            Dictionary with drift scores and alert flags
        """
        logger.info(
            f"Running drift report: reference={len(reference_df):,} rows, "
            f"current={len(current_df):,} rows"
        )

        # Data drift report
        data_report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
            ColumnDriftMetric(column_name="log_price"),
            ColumnDriftMetric(column_name="competitor_price_gap_pct"),
            ColumnDriftMetric(column_name="demand_rolling_mean_30d"),
        ])
        data_report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=self.column_mapping,
        )

        # Model performance report (requires actuals in current_df)
        has_actuals = self.target_col in current_df.columns and \
                      self.prediction_col in current_df.columns

        if has_actuals:
            perf_report = Report(metrics=[
                RegressionQualityMetric(),
                RegressionErrorPlot(),
                RegressionPredictedVsActualScatter(),
            ])
            perf_report.run(
                reference_data=reference_df,
                current_data=current_df,
                column_mapping=self.column_mapping,
            )

        # Extract summary metrics
        data_dict = data_report.as_dict()
        dataset_drift = data_dict["metrics"][0]["result"]

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_rows": len(reference_df),
            "current_rows": len(current_df),
            "n_drifted_features": dataset_drift.get("number_of_drifted_columns", 0),
            "n_total_features": dataset_drift.get("number_of_columns", len(self.feature_columns)),
            "dataset_drift_detected": dataset_drift.get("dataset_drift", False),
            "share_of_drifted_features": dataset_drift.get("share_of_drifted_columns", 0.0),
        }

        # Per-feature drift scores
        feature_drift_scores = {}
        for metric in data_dict["metrics"][1:]:
            if "column_name" in metric.get("result", {}):
                col = metric["result"]["column_name"]
                score = metric["result"].get("drift_score", 0.0)
                feature_drift_scores[col] = score

        results["feature_drift_scores"] = feature_drift_scores

        # Determine alert level
        share_drifted = results["share_of_drifted_features"]
        if share_drifted > 0.5 or results["dataset_drift_detected"]:
            results["alert_level"] = "CRITICAL"
            results["action"] = "immediate_retrain"
        elif share_drifted > 0.25:
            results["alert_level"] = "WARNING"
            results["action"] = "schedule_retrain"
        else:
            results["alert_level"] = "OK"
            results["action"] = "none"

        # Log results
        logger.info(
            f"Drift report complete — "
            f"Alert: {results['alert_level']} | "
            f"Drifted features: {results['n_drifted_features']}/{results['n_total_features']} | "
            f"Action: {results['action']}"
        )

        if results["alert_level"] == "CRITICAL":
            logger.warning(
                "🚨 CRITICAL DRIFT DETECTED — model performance likely degraded. "
                "Trigger retraining pipeline immediately."
            )
        elif results["alert_level"] == "WARNING":
            logger.warning(
                "⚠️  Feature drift detected — schedule retraining within 24 hours."
            )

        # Save reports
        if save:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            html_path = self.reports_dir / f"drift_report_{ts}.html"
            json_path = self.reports_dir / f"drift_report_{ts}.json"

            data_report.save_html(str(html_path))
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Reports saved: {html_path}, {json_path}")
            results["report_path"] = str(html_path)

        return results

    def should_retrain(self, drift_results: dict) -> bool:
        """Return True if drift results indicate retraining is needed."""
        return drift_results["alert_level"] in ("CRITICAL", "WARNING")

    def compute_rolling_mape(
        self,
        predictions_df: pd.DataFrame,
        window_days: int = 7,
    ) -> pd.Series:
        """
        Compute rolling MAPE over recent predictions.

        Args:
            predictions_df: DataFrame with 'date', 'actual', 'predicted' columns
            window_days:     Rolling window size in days

        Returns:
            Series of rolling MAPE values indexed by date
        """
        df = predictions_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["abs_pct_error"] = abs(df["actual"] - df["predicted"]) / (df["actual"].clip(lower=1))

        rolling_mape = df.set_index("date")["abs_pct_error"].rolling(
            f"{window_days}D", min_periods=100
        ).mean()

        # Log alerts
        latest_mape = float(rolling_mape.iloc[-1]) if len(rolling_mape) > 0 else 0.0
        if latest_mape > self.mape_alert:
            logger.error(
                f"🚨 MAPE ALERT: Rolling {window_days}d MAPE = {latest_mape:.1%} "
                f"(threshold = {self.mape_alert:.1%}). Triggering retrain."
            )
        elif latest_mape > self.mape_warning:
            logger.warning(
                f"⚠️  MAPE WARNING: Rolling {window_days}d MAPE = {latest_mape:.1%} "
                f"(threshold = {self.mape_warning:.1%})."
            )
        else:
            logger.info(f"✅ MAPE OK: Rolling {window_days}d MAPE = {latest_mape:.1%}")

        return rolling_mape


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run drift detection report")
    parser.add_argument("--reference", required=True, help="Path to reference Parquet")
    parser.add_argument("--current", required=True, help="Path to current Parquet")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    detector = DriftDetector.from_config(args.config)
    reference_df = pd.read_parquet(args.reference)
    current_df = pd.read_parquet(args.current)

    results = detector.run_drift_report(reference_df, current_df)

    print("\n" + "="*50)
    print("DRIFT DETECTION RESULTS")
    print("="*50)
    print(f"Alert Level:      {results['alert_level']}")
    print(f"Drifted Features: {results['n_drifted_features']} / {results['n_total_features']}")
    print(f"Action Required:  {results['action']}")
    print("="*50)

    if detector.should_retrain(results):
        print("\n⚡ Retraining pipeline would be triggered here.")
        # In production: subprocess.run(["python", "scripts/retrain.py"])
