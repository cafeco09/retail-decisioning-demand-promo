#!/usr/bin/env python3
# Fast constrained policy simulator (predictive, not causal).
# Vectorises scoring by batching candidate actions.

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
    batch_size = int(cfg.get("batch_size", 5000))  # optional, safe default

    df = pd.read_parquet(DATA / "model_frame.parquet").sort_values("WEEK_END_DATE")

    # Decision horizon: last 20% weeks
    cut = df["WEEK_END_DATE"].quantile(0.8)
    hist = df[df["WEEK_END_DATE"] <= cut]
    holdout = df[df["WEEK_END_DATE"] > cut].copy()

    # Limit promo combos to observed patterns, but cap cardinality for speed
    promo_combos = (
        hist[["promo_feature", "promo_display", "promo_tpr_only"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # cap combos (optional safety)
    promo_cap = int(cfg.get("promo_combo_cap", 25))
    if len(promo_combos) > promo_cap:
        promo_combos = promo_combos.sample(promo_cap, random_state=42).reset_index(drop=True)

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

    base = holdout.sample(min(sample_rows, len(holdout)), random_state=42).reset_index(drop=True)

    # Build candidate action table (vectorised)
    candidates = []
    for pm in price_mults:
        cand = base.copy()
        cand["PRICE"] = cand["BASE_PRICE"].astype(float) * float(pm)
        cand["discount_pct"] = np.where(
            cand["BASE_PRICE"].astype(float) > 0,
            (cand["BASE_PRICE"].astype(float) - cand["PRICE"].astype(float)) / cand["BASE_PRICE"].astype(float),
            0.0
        )
        cand = cand[cand["discount_pct"] <= max_disc].copy()
        cand["price_mult"] = float(pm)
        candidates.append(cand)

    if not candidates:
        raise RuntimeError("No candidates after applying max_discount constraint.")

    base_cand = pd.concat(candidates, ignore_index=True)

    # Cross join with promo combos (bounded by promo_cap)
    base_cand["_k"] = 1
    promo_combos["_k"] = 1
    cand = base_cand.merge(promo_combos, on="_k", suffixes=("", "_p")).drop(columns=["_k"])

    # Apply promo columns from combos
    cand["promo_feature"] = cand["promo_feature_p"].astype(int)
    cand["promo_display"] = cand["promo_display_p"].astype(int)
    cand["promo_tpr_only"] = cand["promo_tpr_only_p"].astype(int)
    cand.drop(columns=["promo_feature_p", "promo_display_p", "promo_tpr_only_p"], inplace=True)

    cand["is_promo"] = (
        (cand["promo_feature"] == 1) | (cand["promo_display"] == 1) | (cand["promo_tpr_only"] == 1)
    ).astype("int8")

    # Score in batches
    preds = np.empty(len(cand), dtype=float)
    for start in range(0, len(cand), batch_size):
        end = min(start + batch_size, len(cand))
        Xb = cand.loc[start:end-1, cols]
        preds[start:end] = np.expm1(pipe.predict(Xb)).clip(0, None)

    cand["units_pred"] = preds
    cand["revenue_pred"] = cand["units_pred"] * cand["PRICE"].astype(float)

    # Pick best action per original row (use index of base row via an id)
    cand["base_id"] = cand.index // (len(promo_combos) if len(promo_combos) else 1)  # fallback
    # Better: reconstruct base_id from merge order
    # base rows repeat in blocks of (price_mult candidates * promo_combos). We keep a stable id from base.
    # We'll add it properly:
    # (recreate quickly)
    # NOTE: base_id is added below more robustly.
    # --- robust base_id ---
    # base had rows 0..n-1; base_cand preserves them but after concat.
    # We'll derive from 'WEEK_END_DATE, STORE_NUM, UPC' keys + row number.
    key_cols = ["WEEK_END_DATE", "STORE_NUM", "UPC"]
    base_keys = base[key_cols].copy()
    base_keys["base_id"] = np.arange(len(base_keys))
    cand = cand.merge(base_keys, on=key_cols, how="left")

    best = (
        cand.sort_values(["base_id", "revenue_pred"], ascending=[True, False])
        .groupby("base_id", as_index=False)
        .head(1)
    )

    out = best[[
        "WEEK_END_DATE", "STORE_NUM", "UPC",
        "SEG_VALUE_NAME", "ADDRESS_STATE_PROV_CODE",
        "PRICE", "discount_pct", "price_mult",
        "promo_feature", "promo_display", "promo_tpr_only",
        "units_pred", "revenue_pred"
    ]].rename(columns={
        "SEG_VALUE_NAME": "seg",
        "ADDRESS_STATE_PROV_CODE": "state",
        "PRICE": "price_rec",
        "discount_pct": "discount_rec",
    })

    out_path = REPORTS / "recommended_actions_sample.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote recommendations: {len(out):,} rows -> {out_path}")
    print(f"Candidates scored: {len(cand):,} (sample_rows={len(base)}, price_mults={len(price_mults)}, promo_combos={len(promo_combos)})")

if __name__ == "__main__":
    main()
