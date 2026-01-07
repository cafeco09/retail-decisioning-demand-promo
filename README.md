# Retail Decisioning: Demand Modelling + Price/Promo Policy Simulation

Store-segment retail decisioning built on **store × product × week** data (dunnhumby *Breakfast at the Frat*).
This repo trains a demand model and then simulates **constrained** pricing/promotion actions to recommend what to do next week.

**Important:** recommendations are produced via **predictive simulation** (not causal uplift). Promotions in the dataset are observational, not randomised.

---

## What this project does

### 1) Demand model (forecasting)
Predicts **next-week unit sales (UNITS)** for each store–product pair using:
- **Decision variables:** `PRICE`, `BASE_PRICE`, discount depth, and promo support (`FEATURE`, `DISPLAY`, `TPR_ONLY`)
- **Behavioural time-series signals:** lags and rolling averages of past units (habit / replenishment patterns)
- **Seasonality:** week/month effects
- **Store context:** store segment (Value/Mainstream/Upscale) + geography
- **Product context:** category/sub-category + manufacturer + size

The model is saved as a single pipeline (preprocessing + model) so scoring and simulation use identical transformations.

### 2) Decision engine (policy simulation)
For each store–product–week context in the evaluation period, we:
- generate a small set of **candidate actions** (price multipliers + historically observed promo combinations)
- apply constraints (e.g., maximum discount cap)
- score each action using the demand model
- recommend the action that maximises **predicted revenue = predicted_units × candidate_price**

---

## What this project does *not* claim

- **No causal impact claims:** this is not an uplift model and does not prove promotions cause incremental sales.
- **No dataset redistribution:** this repo does not include the dataset; it shares code only.
## Dashboard

![Model metrics](reports/figures/01_model_metrics.png)
![Promo rate by policy](reports/figures/02_promo_rate_by_policy.png)
![Discount distribution by policy](reports/figures/03_discount_dist_by_policy.png)

---

## Data (not included)

Place the Excel workbook here:

```text
data/raw/dunnhumby - Breakfast at the Frat.xlsx

