#!/usr/bin/env python3
# Constrained policy simulator (predictive, not causal).

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import yaml

DATA = Path("data/processed")
REPORTS = Path("reports")
CFG = Path("configs/policy.yaml")
REPORTS.mkdir(exist_ok=True)

def main():
    cfg = yaml.safe_load(CFG.read_text())
    price_mults = cfg.get("price_multipliers", [1.0, 0.95, 0.90, 0.85])
    max_disc = float(cfg.get("max_discount_pct", 0.30))
    sample_rows = int(cfg.get("sample_rows", 20000))

    df = pd.read_parquet(DATA / "model_frame.parquet").sort_values("WEEK_END_DATE")

    # Decision horizon: last 20% of weeks
    cut = df["WEEK_END_DATE"].quantile(0.8)
    hist = df[df["WEEK_END_DATE"] <= cut]
    holdout = df[df["WEEK_END_DATE"] > cut].copy()

    # Only allow promo combinations that actually occurred historically
    promo_combos = (
        hist[["promo_feature", "promo_display", "promo_tpr_only"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    pipe = joblib.load(DATA / "demand_model.joblib")

    numeric = [
        "PRICE",
        "BASE_PRICE",
        "discount_pct",
        "is_promo",
        "promo_feature",
        "promo_display",
        "promo_tpr_only",
        "units_lag_1",
        "units_lag_4",
        "units_roll_4",
        "price_lag_1",
        "discount_lag_1",
        "visits_lag_1",
        "hhs_lag_1",
        "year",
        "weekofyear",
        "month",
    ]
    categorical = [
        "STORE_NUM",
        "UPC",
        "SEG_VALUE_NAME",
        "ADDRESS_STATE_PROV_CODE",
        "MSA_CODE",
        "CATEGORY",
        "SUB_CATEGORY",
        "MANUFACTURER",
        "PRODUCT_SIZE",
    ]
    cols = numeric + categorical

    recs = []
    sample = holdout.sample(min(sample_rows, len(holdout)), random_state=42)

    for _, base in sample.iterrows():
        best = None

        for pm in price_mults:
            cand_price = float(base["BASE_PRICE"]) * float(pm)
            disc = (
                (float(base["BASE_PRICE"]) - cand_price) / float(base["BASE_PRICE"])
                if float(base["BASE_PRICE"])
                else 0.0
            )
            if disc > max_disc:
                continue

            for _, combo in promo_combos.iterrows():
                cand = base.copy()
                cand["PRICE"] = cand_price
                cand["discount_pct"] = disc
                cand["promo_feature"] = int(combo["promo_feature"])
                cand["promo_display"] = int(combo["promo_display"])
                cand["promo_tpr_only"] = int(combo["promo_tpr_only"])
                cand["is_promo"] = int(
                    (cand["promo_feature"] == 1)
                    or (cand["promo_display"] == 1)
                    or (cand["promo_tpr_only"] == 1)
                )

                X = pd.DataFrame([cand[cols].to_dict()])
                pred_units = float(np.expm1(pipe.predict(X)[0]))
                pred_units = max(0.0, pred_units)
                revenue = pred_units * cand_price

                if (best is None) or (revenue > best["revenue_pred"]):
                    best = {
                        "WEEK_END_DATE": cand["WEEK_END_DATE"],
                        "STORE_NUM": int(cand["STORE_NUM"]),
                        "UPC": int(cand["UPC"]),
                        "seg": cand.get("SEG_VALUE_NAME", None),
                        "state": cand.get("ADDRESS_STATE_PROV_CODE", None),
                        "price_rec": cand_price,
                        "discount_rec": disc,
                        "promo_feature": cand["promo_feature"],
                        "promo_display": cand["promo_display"],
                        "promo_tpr_only": cand["promo_tpr_only"],
                        "units_pred": pred_units,
                        "revenue_pred": revenue,
                    }

        if best:
            recs.append(best)

    out = pd.DataFrame(recs)
    out.to_csv(REPORTS / "recommended_actions_sample.csv", index=False)
    print(f"Wrote recommendations: {len(out):,} rows -> {REPORTS/'recommended_actions_sample.csv'}")

if __name__ == "__main__":
    main()
