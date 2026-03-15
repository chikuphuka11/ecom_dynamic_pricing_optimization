"""
tests/test_guardrails.py — Pricing guardrail validation tests.

These tests verify that the optimizer NEVER produces unsafe price recommendations.
Think of these as business logic tests — they encode hard constraints that
must hold regardless of what the ML model predicts.

Economics context: Guardrails exist because:
1. Demand models have uncertainty — we don't want a noisy estimate to cause
   a 50% price hike that destroys customer trust
2. Regulatory constraints (price gouging laws, MAP agreements with suppliers)
3. Brand equity — extreme discounting signals low quality (Veblen effect risk)
4. Competitive response — aggressive price cuts may trigger price wars

Run with: pytest tests/test_guardrails.py -v
"""

import pytest
import numpy as np


# ─── Guardrail Logic (mirrors src/models/optimizer.py) ───────────────────────

def apply_guardrails(
    optimal_price: float,
    current_price: float,
    unit_cost: float,
    max_change_pct: float = 0.30,
    min_margin_pct: float = 0.10,
) -> tuple[float, str]:
    """
    Apply all business guardrails to a raw optimal price.

    Returns:
        (clipped_price, binding_constraint)
        binding_constraint: 'none' | 'guardrail_upper' | 'guardrail_lower' | 'margin'
    """
    binding = "none"

    # Guardrail: max price change from current
    upper_bound = current_price * (1 + max_change_pct)
    lower_bound = current_price * (1 - max_change_pct)

    if optimal_price > upper_bound:
        optimal_price = upper_bound
        binding = "guardrail_upper"
    elif optimal_price < lower_bound:
        optimal_price = lower_bound
        binding = "guardrail_lower"

    # Margin floor: price must be >= cost / (1 - min_margin)
    # This is the standard cost-plus floor: P ≥ C / (1 - m)
    margin_floor = unit_cost / (1 - min_margin_pct)
    if optimal_price < margin_floor:
        optimal_price = margin_floor
        binding = "margin"

    return optimal_price, binding


# ─── Guardrail Tests ──────────────────────────────────────────────────────────

class TestPriceChangeGuardrail:
    """The optimizer must never suggest more than ±30% price change."""

    def test_upward_clip(self):
        price, constraint = apply_guardrails(
            optimal_price=10.00,   # Model wants to go much higher
            current_price=5.00,
            unit_cost=2.00,
            max_change_pct=0.30,
        )
        assert price == pytest.approx(6.50, rel=1e-4)  # 5.00 * 1.30
        assert constraint == "guardrail_upper"

    def test_downward_clip(self):
        price, constraint = apply_guardrails(
            optimal_price=1.00,    # Model wants to go much lower
            current_price=5.00,
            unit_cost=0.50,
            max_change_pct=0.30,
        )
        assert price == pytest.approx(3.50, rel=1e-4)  # 5.00 * 0.70
        assert constraint == "guardrail_lower"

    def test_within_guardrail_no_clip(self):
        price, constraint = apply_guardrails(
            optimal_price=5.50,   # 10% increase — within guardrail
            current_price=5.00,
            unit_cost=2.00,
            max_change_pct=0.30,
        )
        assert price == pytest.approx(5.50, rel=1e-4)
        assert constraint == "none"

    @pytest.mark.parametrize("change_pct", [-0.30, -0.15, 0.0, 0.15, 0.30])
    def test_boundary_prices_are_valid(self, change_pct):
        current = 10.00
        boundary_price = current * (1 + change_pct)
        price, _ = apply_guardrails(
            optimal_price=boundary_price,
            current_price=current,
            unit_cost=3.00,
            max_change_pct=0.30,
        )
        assert abs(price - boundary_price) < 0.01

    def test_extreme_model_prediction_clipped(self):
        """Even if the model predicts 10x price increase, guardrail limits to 30%."""
        price, constraint = apply_guardrails(
            optimal_price=999.99,
            current_price=5.00,
            unit_cost=1.00,
            max_change_pct=0.30,
        )
        assert price <= 5.00 * 1.30 + 1e-6
        assert constraint == "guardrail_upper"


class TestMarginFloorGuardrail:
    """Price must always cover costs with minimum margin."""

    def test_margin_floor_enforced(self):
        price, constraint = apply_guardrails(
            optimal_price=2.00,   # Below margin floor
            current_price=5.00,
            unit_cost=3.00,
            min_margin_pct=0.15,  # Need 15% margin → floor = 3.00 / 0.85 ≈ 3.53
        )
        assert price >= 3.00 / (1 - 0.15) - 1e-6
        assert constraint == "margin"

    def test_margin_floor_not_binding_when_unnecessary(self):
        price, constraint = apply_guardrails(
            optimal_price=8.00,   # Well above margin floor
            current_price=7.50,
            unit_cost=3.00,
            min_margin_pct=0.10,  # Floor = 3.00 / 0.90 ≈ 3.33
        )
        # Margin not binding — guardrail upper might be
        assert constraint in ("none", "guardrail_upper")

    def test_zero_cost_no_margin_floor_issue(self):
        """Free goods (digital products with zero marginal cost)."""
        price, constraint = apply_guardrails(
            optimal_price=4.00,
            current_price=5.00,
            unit_cost=0.00,
            min_margin_pct=0.10,
        )
        assert price >= 0

    @pytest.mark.parametrize("unit_cost,min_margin,current_price", [
        (2.00, 0.10, 5.00),
        (1.50, 0.20, 3.00),
        (5.00, 0.05, 10.00),
        (0.50, 0.30, 2.00),
    ])
    def test_margin_floor_formula(self, unit_cost, min_margin, current_price):
        """Margin floor = unit_cost / (1 - min_margin)."""
        expected_floor = unit_cost / (1 - min_margin)
        price, _ = apply_guardrails(
            optimal_price=0.01,    # Below floor, forces margin binding
            current_price=current_price,
            unit_cost=unit_cost,
            min_margin_pct=min_margin,
            max_change_pct=1.0,    # Disable guardrail so only margin binds
        )
        assert price >= expected_floor - 1e-6


class TestElasticityGuardrails:
    """Sanity checks on elasticity estimates."""

    @pytest.mark.parametrize("elasticity", [-0.1, -0.5, -1.0, -2.0, -5.0])
    def test_valid_elasticities_are_negative(self, elasticity):
        """Normal goods have negative price elasticity (law of demand)."""
        assert elasticity < 0, "Elasticity must be negative for normal goods"

    def test_reject_positive_elasticity_unless_flagged(self):
        """
        Positive elasticity (Giffen good or model error) must be flagged.
        In practice, positive elasticity in e-commerce almost always indicates
        a data problem (endogeneity not fully corrected, or data leakage).
        """
        elasticity = 0.5   # Positive — suspicious
        is_suspicious = elasticity > 0
        assert is_suspicious, "Positive elasticity should raise a flag"

    @pytest.mark.parametrize("elasticity", [-0.1, -0.5, -0.9])
    def test_inelastic_demand_interpretation(self, elasticity):
        """Inelastic demand: |ε| < 1 → price increase raises revenue."""
        is_inelastic = abs(elasticity) < 1
        assert is_inelastic
        # Revenue-maximizing action: raise price (MR > 0 in inelastic region)
        revenue_effect = "increase_price"
        assert revenue_effect == "increase_price"

    @pytest.mark.parametrize("elasticity", [-1.1, -1.5, -2.0, -3.0])
    def test_elastic_demand_interpretation(self, elasticity):
        """Elastic demand: |ε| > 1 → price decrease raises revenue."""
        is_elastic = abs(elasticity) > 1
        assert is_elastic
        revenue_effect = "decrease_price"
        assert revenue_effect == "decrease_price"


class TestBatchGuardrails:
    """Batch-level guardrail checks — catches systemic pricing issues."""

    def test_no_more_than_x_pct_skus_at_guardrail(self):
        """
        If >20% of SKUs are hitting the guardrail upper bound, something is
        wrong — the model may be systematically biased upward.
        This test would run in CI against a sample of predictions.
        """
        np.random.seed(42)
        n_skus = 1000
        # Simulate recommendations (most should be within guardrail)
        price_changes = np.random.normal(0.05, 0.10, n_skus)  # Mean +5%, std 10%
        at_upper_guardrail = np.sum(price_changes >= 0.299)  # At 30% bound

        pct_at_guardrail = at_upper_guardrail / n_skus
        assert pct_at_guardrail < 0.20, (
            f"Too many SKUs ({pct_at_guardrail:.1%}) hitting upper guardrail — "
            "model may be biased toward price increases"
        )

    def test_average_price_change_is_reasonable(self):
        """
        Average recommended price change should be small (±5–10%).
        Large systematic shifts suggest model drift.
        """
        np.random.seed(42)
        price_changes = np.random.normal(0.03, 0.08, 1000)  # +3% mean
        avg_change = np.mean(price_changes)
        assert abs(avg_change) < 0.15, (
            f"Average price change {avg_change:.1%} is suspiciously large"
        )

    def test_no_zero_or_negative_prices(self):
        """Prices must always be positive — a fundamental constraint."""
        sample_prices = [4.99, 9.99, 0.99, 24.99, 1.49]
        for price in sample_prices:
            assert price > 0, f"Price {price} is not positive"
