#!/usr/bin/env python3
# Build modelling frame for next-week demand forecasting.

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data/processed")

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["WEEK_END_DATE"]
    df["year"] = dt.dt.year.astype("int16")
    df["weekofyear"] = dt.dt.isocalendar().week.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    return df

def add_price_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    df["discount_pct"] = np.where(
        df["BASE_PRICE"] > 0,
        (df["BASE_PRICE"] - df["PRICE"]) / df["BASE_PRICE"],
        0.0,
    )
    df["discount_pct"] = df["discount_pct"].clip(-1, 1)
    df["is_promo"] = (
        (df["FEATURE"] == 1) | (df["DISPLAY"] == 1) | (df["TPR_ONLY"] == 1)
    ).astype("int8")
    df["promo_feature"] = df["FEATURE"].astype("int8")
    df["promo_display"] = df["DISPLAY"].astype("int8")
    df["promo_tpr_only"] = df["TPR_ONLY"].astype("int8")
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["STORE_NUM", "UPC", "WEEK_END_DATE"])
    g = df.groupby(["STORE_NUM", "UPC"], sort=False)

    # Demand lags
    df["units_lag_1"] = g["UNITS"].shift(1)
    df["units_lag_4"] = g["UNITS"].shift(4)
    df["units_roll_4"] = g["UNITS"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )

    # Context lags
    df["price_lag_1"] = g["PRICE"].shift(1)
    df["discount_lag_1"] = g["discount_pct"].shift(1)
    df["visits_lag_1"] = g["VISITS"].shift(1)
    df["hhs_lag_1"] = g["HHS"].shift(1)

    for c in [
        "units_lag_1",
        "units_lag_4",
        "units_roll_4",
        "price_lag_1",
        "discount_lag_1",
        "visits_lag_1",
        "hhs_lag_1",
    ]:
        df[c] = df[c].fillna(0.0)

    # QC-only metrics (not features by default)
    df["units_per_visit"] = np.where(df["VISITS"] > 0, df["UNITS"] / df["VISITS"], 0.0)
    df["visits_per_hh"] = np.where(df["HHS"] > 0, df["VISITS"] / df["HHS"], 0.0)

    return df

def main():
    tx = pd.read_parquet(DATA / "transactions.parquet")
    stores = pd.read_parquet(DATA / "stores.parquet")
    prods = pd.read_parquet(DATA / "products.parquet")

    df = tx.merge(stores, left_on="STORE_NUM", right_on="STORE_ID", how="left")
    df = df.merge(prods, on="UPC", how="left")

    df = add_time_features(df)
    df = add_price_promo_features(df)
    df = add_lag_features(df)

    out = DATA / "model_frame.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote model frame: {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()
