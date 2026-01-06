#!/usr/bin/env python3
# Extract dunnhumby Excel workbook into parquet for fast I/O.

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def dedupe_store_lookup(store: pd.DataFrame) -> pd.DataFrame:
    # If a STORE_ID has multiple segment labels, mark as AMBIGUOUS and keep one row.
    dup = store.groupby("STORE_ID")["SEG_VALUE_NAME"].nunique()
    ambiguous_ids = set(dup[dup > 1].index.tolist())
    store_dedup = store.sort_values(["STORE_ID"]).drop_duplicates(subset=["STORE_ID"], keep="first").copy()
    store_dedup.loc[store_dedup["STORE_ID"].isin(ambiguous_ids), "SEG_VALUE_NAME"] = "AMBIGUOUS"
    return store_dedup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to the Excel file")
    ap.add_argument("--outdir", default="data/processed", help="Output directory")
    ap.add_argument("--tx_rows", type=int, default=None, help="Optional limit for faster dev")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    store = pd.read_excel(xlsx, sheet_name="dh Store Lookup", header=1)
    prod  = pd.read_excel(xlsx, sheet_name="dh Products Lookup", header=1)
    store = dedupe_store_lookup(store)

    store.to_parquet(outdir / "stores.parquet", index=False)
    prod.to_parquet(outdir / "products.parquet", index=False)

    tx = pd.read_excel(xlsx, sheet_name="dh Transaction Data", header=1, nrows=args.tx_rows)
    tx["WEEK_END_DATE"] = pd.to_datetime(tx["WEEK_END_DATE"])
    tx["STORE_NUM"] = tx["STORE_NUM"].astype("int32")
    tx["UPC"] = tx["UPC"].astype("int64")
    tx.to_parquet(outdir / "transactions.parquet", index=False)

    print(f"Wrote transactions: {len(tx):,} rows")
    print(f"Wrote stores: {store['STORE_ID'].nunique():,} unique stores")
    print(f"Wrote products: {prod['UPC'].nunique():,} unique UPCs")

if __name__ == "__main__":
    main()
