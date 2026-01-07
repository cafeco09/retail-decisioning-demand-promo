#!/usr/bin/env python3
# Fast constrained retail policy simulator (predictive, not causal).
#
# What it does:
# - Generates candidate actions (price multipliers × promo combos)
# - Scores actions in batches using the trained demand model pipeline
# - Outputs two policies:
#   1) revenue_only: maximise predicted revenue
#   2) penalised: maximise revenue minus penalties for discounting and promo usage
#
# Notes:
# - This is NOT causal uplift; promos are observational.
# - Penalties approximate real-world scarcity/cost of promos + margin pressure.

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
    if not CFG.exists():
        raise FileNotFoundError("Missing configs/policy.yaml")

    cfg = yaml.safe_load(CFG.read_text())

    price_mults = list(cfg.get("price_multipliers", [1.0, 0.95, 0.90, 0.85]))
    max_disc = float(cfg.get("max_discount_pct", 0.30))
    sample_rows = int(cfg.get("sample_rows", 200))
    batch_size = int(cfg.get("batch_size", 5000))
    promo_cap = int(cfg.get("promo_combo_cap", 25))

    # Penalties (set to 0.0 to disable)
    lam_d = float(cfg.get("lambda_discount", 0.0))
    lam_p = float(cfg.get("lambda_promo", 0.0))

    df = pd.read_parquet(DATA / "model_frame.parquet").sort_values("WEEK_END_DATE")

    # Decision horizon: last 20% weeks
    cut = df["WEEK_END_DATE"].quantile(0.8)
    hist = df[df["WEEK_END_DATE"] <= cut]
    holdout = df[df["WEEK_END_DATE"] > cut].copy()

    # Promo combos by frequency (not random) + force at least one promo-on if present
    combo_cols = ["promo_feature", "promo_display", "promo_tpr_only"]
    combo_counts = (
        hist[combo_cols]
        .value_counts()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    promo_combos = combo_counts[combo_cols].head(promo_cap).reset_index(drop=True)

    promo_on = combo_counts[combo_counts[combo_cols].sum(axis=1) > 0]
    if len(promo_on) > 0:
        first_on = promo_on[combo_cols].head(1)
        promo_combos = (
            pd.concat([promo_combos, first_on], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    pipe = joblib.load(DATA / "demand_model.joblib")

    numeric = [
        "PRICE", "BASE_PRICE", "discount_pct",
        "is_promo", "promo_feature", "promo_display", "promo_tpr_only",
        "units_lag_1", "units_lag_4", "units_roll_4",
        "price_lag_1", "discount_lag_1", "visits_lag_1", "hhs_lag_1",
        "year", "weekofyear", "month",
    ]
    categorical = [
        "STORE_NUM", "UPC",
        "SEG_VALUE_NAME", "ADDRESS_STATE_PROV_CODE", "MSA_CODE",
        "CATEGORY", "SUB_CATEGORY", "MANUFACTURER", "PRODUCT_SIZE",
    ]
    cols = numeric + categorical

    # Base rows to recommend on
    base = holdout.sample(min(sample_rows, len(holdout)), random_state=42).reset_index(drop=True)
    key_cols = ["WEEK_END_DATE", "STORE_NUM", "UPC"]
    base_keys = base[key_cols].copy()
    base_keys["base_id"] = np.arange(len(base_keys))

    # ---- build price candidates (vectorised) ----
    cand_list = []
    for pm in price_mults:
        c = base.copy()
        c["price_mult"] = float(pm)
        c["PRICE"] = c["BASE_PRICE"].astype(float) * float(pm)

        c["discount_pct"] = np.where(
            c["BASE_PRICE"].astype(float) > 0,
            (c["BASE_PRICE"].astype(float) - c["PRICE"].astype(float)) / c["BASE_PRICE"].astype(float),
            0.0,
        )

        # Enforce discount constraint
        c = c[c["discount_pct"] <= max_disc].copy()
        cand_list.append(c)

    if not cand_list:
        raise RuntimeError("No candidates after applying max_discount_pct constraint.")

    base_cand = pd.concat(cand_list, ignore_index=True)

    # ---- cross join with promo combos ----
    base_cand["_k"] = 1
    promo_combos = promo_combos.copy()
    promo_combos["_k"] = 1

    cand = base_cand.merge(promo_combos, on="_k", suffixes=("", "_p")).drop(columns=["_k"])

    # Apply promo values from combos
    cand["promo_feature"] = cand["promo_feature_p"].astype(int)
    cand["promo_display"] = cand["promo_display_p"].astype(int)
    cand["promo_tpr_only"] = cand["promo_tpr_only_p"].astype(int)
    cand.drop(columns=["promo_feature_p", "promo_display_p", "promo_tpr_only_p"], inplace=True)

    cand["is_promo"] = (
        (cand["promo_feature"] == 1) | (cand["promo_display"] == 1) | (cand["promo_tpr_only"] == 1)
    ).astype("int8")

    # Attach base_id for grouping best action
    cand = cand.merge(base_keys, on=key_cols, how="left")

    # ---- score in batches ----
    preds = np.empty(len(cand), dtype=float)
    for start in range(0, len(cand), batch_size):
        end = min(start + batch_size, len(cand))
        Xb = cand.loc[start:end - 1, cols]
        preds[start:end] = np.expm1(pipe.predict(Xb)).clip(0, None)

    cand["units_pred"] = preds
    cand["revenue_pred"] = cand["units_pred"] * cand["PRICE"].astype(float)

    # Penalised objective to approximate promo scarcity and margin pressure
    cand["score_penalised"] = (
        cand["revenue_pred"]
        - lam_d * cand["discount_pct"].astype(float)
        - lam_p * cand["is_promo"].astype(float)
    )

    # ---- choose best action per base_id under two policies ----
    best_rev = (
        cand.sort_values(["base_id", "revenue_pred"], ascending=[True, False])
        .groupby("base_id", as_index=False)
        .head(1)
        .copy()
    )
    best_rev["policy"] = "revenue_only"

    best_pen = (
        cand.sort_values(["base_id", "score_penalised"], ascending=[True, False])
        .groupby("base_id", as_index=False)
        .head(1)
        .copy()
    )
    best_pen["policy"] = "penalised"

    best = pd.concat([best_rev, best_pen], ignore_index=True)

    out = best[[
        "policy",
        "WEEK_END_DATE", "STORE_NUM", "UPC",
        "SEG_VALUE_NAME", "ADDRESS_STATE_PROV_CODE",
        "PRICE", "discount_pct", "price_mult",
        "promo_feature", "promo_display", "promo_tpr_only",
        "is_promo",
        "units_pred", "revenue_pred", "score_penalised",
    ]].rename(columns={
        "SEG_VALUE_NAME": "seg",
        "ADDRESS_STATE_PROV_CODE": "state",
        "PRICE": "price_rec",
        "discount_pct": "discount_rec",
    })

    out_path = REPORTS / "recommended_actions_sample.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote recommendations: {len(out):,} rows -> {out_path}")
    print(
        f"Candidates scored: {len(cand):,} "
        f"(sample_rows={len(base)}, price_mults={len(price_mults)}, promo_combos={len(promo_combos)})"
    )
    print(f"Penalties: lambda_discount={lam_d}, lambda_promo={lam_p}")

if __name__ == "__main__":
    main()
