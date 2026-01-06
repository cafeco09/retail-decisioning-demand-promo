#!/usr/bin/env python3
# Train demand model: predict UNITS (log1p) for store-product-week.

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import joblib

DATA = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

def smape(y_true, y_pred, eps=1e-6) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def main():
    df = pd.read_parquet(DATA / "model_frame.parquet").sort_values("WEEK_END_DATE")
    y = np.log1p(df["UNITS"].values)

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

    X = df[numeric + categorical].copy()

    # Out-of-time split: last 20% of weeks as test
    cut = df["WEEK_END_DATE"].quantile(0.8)
    train_idx = df["WEEK_END_DATE"] <= cut
    test_idx = ~train_idx

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="regression",
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X[train_idx], y[train_idx])

    pred_log = pipe.predict(X[test_idx])
    pred = np.expm1(pred_log).clip(0, None)
    y_true = df.loc[test_idx, "UNITS"].values

    mae = mean_absolute_error(y_true, pred)
    s = smape(y_true, pred)

    metrics = pd.DataFrame(
        [{"mae": float(mae), "smape": float(s), "test_rows": int(test_idx.sum())}]
    )
    metrics.to_csv(REPORTS / "metrics.csv", index=False)
    print(metrics.to_string(index=False))

    joblib.dump(pipe, DATA / "demand_model.joblib")
    print(f"Saved model pipeline -> {DATA/'demand_model.joblib'}")

if __name__ == "__main__":
    main()
