# Economics Primer for This Project
## Bridging Your BA Economics to Production ML

This document maps your undergraduate economics concepts directly to the code
in this project. You already know the theory — this explains where it lives.

---

## 1. Price Elasticity of Demand → `src/models/causal_dml.py`

**What you know:** ε = (ΔQ/Q) / (ΔP/P) = ∂log Q / ∂log P

**How it appears in code:** We estimate β in the log-log specification:
```
log(units_sold) = α + β·log(price) + γ·controls + ε
```
β is directly interpretable as the price elasticity. This is the `DML_OUTCOME`
and `DML_TREATMENT` in the feature pipeline.

**Typical values to expect:**
- Groceries: -0.5 to -1.2 (inelastic to unit elastic)
- Electronics: -1.5 to -3.0 (elastic)
- Luxury goods: varies significantly by brand equity

---

## 2. Endogeneity & Simultaneous Equations → DML's whole purpose

**What you know from Econometrics:** If Cov(P, ε) ≠ 0, OLS is biased.
In demand estimation, this happens because:
- We raise prices on popular items (price → demand, but also popularity → price)
- We discount slow-moving inventory (inventory → price → demand)
- Promotions affect both price AND demand simultaneously

**How this is solved in code:**
```python
# LinearDML partials out the confounders W from BOTH Y and T:
# Ỹ = Y - E[Y|W]   (demand residual)
# T̃ = T - E[T|W]   (price residual = price variation UNEXPLAINED by W)
# Then: Ỹ = θ·T̃ + ε  gives unbiased θ
```

This is exactly the Frisch-Waugh-Lovell (FWL) theorem applied with
ML nuisance models instead of OLS. It's 2SLS without needing an external
instrument — the "instrument" is the variation in price not explained by demand conditions.

---

## 3. Marginal Revenue = Marginal Cost → `src/models/optimizer.py`

**What you know:** Profit-maximizing price satisfies MR = MC.
For a monopolist: P* = MC × ε/(ε+1)  (Lerner index condition)

**How this appears in code:** The pricing optimizer solves:
```python
def revenue(price):
    log_q = demand_model.predict(price, context_features)
    return -(price * np.exp(log_q))  # Negative because we minimize

result = scipy.optimize.minimize_scalar(
    revenue,
    bounds=(min_price, max_price),
    method='bounded'
)
```

The closed-form Lerner solution assumes constant elasticity. The optimizer
handles non-constant elasticity (realistic) and multiple constraints (margin floor,
price guardrails) that the closed form can't accommodate.

---

## 4. Cross-Price Elasticity → `src/features/pipeline.py` (CompetitorPriceFeatures)

**What you know:** ε_xy = ∂log(Qx) / ∂log(Py)
- Positive cross-elasticity → substitutes (our coffee vs. competitor's coffee)
- Negative cross-elasticity → complements (coffee ↑ price → creamer demand ↓)

**In code:**
```python
X["competitor_price_gap_pct"] = (competitor_price - our_price) / our_price
X["log_price_ratio"] = log(our_price / competitor_price)
```
These features capture the cross-price effect in the demand model.
A positive coefficient on `competitor_price_gap_pct` confirms substitutability.

---

## 5. Consumer Surplus & Market Power → Monitoring

**What you know:** CS = area above price, below demand curve.
Price optimization extracts producer surplus at the expense of consumer surplus.

**Why it matters in production:**
The monitoring module tracks:
- Whether prices are consistently at the guardrail upper bound
  (suggests the model is always trying to extract maximum surplus)
- Price dispersion across customer segments
  (flags potential price discrimination concerns)
- Competitor response patterns
  (Stackelberg leader-follower dynamics in repeat pricing games)

---

## 6. Revealed Preference → The entire dataset

**What you know:** Consumers reveal their preferences through purchases.
WTP (willingness to pay) is revealed by the prices at which they do/don't buy.

**In code:** Every row in our training data is a revealed preference observation:
- `(price, units_sold, date, product_id)` = an observed point on the demand curve
- Different prices across stores/time periods give us the variation needed to
  trace out the demand curve (this is why price variation is essential for
  elasticity estimation — no variation = no identification)

---

## 7. Information Asymmetry & Adverse Selection → Data Drift Monitoring

**What you know:** Hidden information creates market failures.

**Subtle version in ML:**
- We observe prices and quantities, but NOT consumer intent/substitution behavior
- When competitors enter/exit, our model's training distribution changes
  (the mapping from price to quantity shifts)
- This is operationalized as DATA DRIFT — detected by Evidently AI
- When drift exceeds the PSI threshold (0.20), we retrain: we're "re-learning"
  the new market structure

---

## Key Papers to Read (connects your Econ to this codebase)

| Paper | Relevance |
|-------|-----------|
| Chernozhukov et al. (2018) "Double/Debiased Machine Learning" | The DML algorithm — an econometrics paper, not just ML |
| Berry (1994) "Estimating Discrete-Choice Models of Product Differentiation" | How to handle product differentiation in demand estimation |
| Hitsch, Hortaçsu & Ariely (2010) "Matching and Sorting in Online Dating" | How to use revealed preference in platform contexts |
| Einav & Levin (2014) "Economics in the Age of Big Data" | Bridges economics and modern data methods — recommended first read |
