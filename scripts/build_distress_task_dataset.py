#!/usr/bin/env python3
"""
Build financial-distress prediction dataset (full A-share coverage).

Label definition:
  label_loss_next_year = 1 if annual net_profit in year t+1 < 0, else 0.

Uses B002000000 (净利润) from CSMAR income statement, which was NOT in the
original features_financial.csv (that only had B001100000=营业收入 and
B001101000=主营业务收入). We extract it from the pre-cached net_profit_raw.csv.

Time split (same convention as rating task):
  train: year <= 2018
  val:   2019 <= year <= 2021
  test:  2022 <= year <= 2024

Outputs:
  processed/stage1/distress_task/distress_panel_labeled.csv
  processed/stage1/distress_task/train.csv
  processed/stage1/distress_task/val.csv
  processed/stage1/distress_task/test.csv
  processed/stage1/distress_task/summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_ROOT / "processed" / "stage1"
OUT_DIR = STAGE1_DIR / "distress_task"
NET_PROFIT_RAW = STAGE1_DIR / "net_profit_raw.csv"


def normalize_stock_code(v: object) -> str:
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        s = str(int(float(s)))
    return s.zfill(6)


def build_annual_net_profit() -> pd.DataFrame:
    """Build company-year annual net profit from quarterly CSMAR data."""
    raw = pd.read_csv(NET_PROFIT_RAW, dtype={"Stkcd": str})
    raw["Stkcd"] = raw["Stkcd"].apply(normalize_stock_code)
    raw["Accper"] = pd.to_datetime(raw["Accper"], errors="coerce")
    raw["net_profit"] = pd.to_numeric(raw["B002000000"], errors="coerce")
    raw = raw.dropna(subset=["Stkcd", "Accper", "net_profit"])
    raw["year"] = raw["Accper"].dt.year
    raw["month"] = raw["Accper"].dt.month

    # Keep only annual reports (month 12) or latest report per year
    annual = raw[raw["month"] == 12].copy()
    if len(annual) == 0:
        # Fallback: use latest report per company-year
        annual = raw.sort_values("Accper").drop_duplicates(["Stkcd", "year"], keep="last")
    else:
        annual = annual.sort_values("Accper").drop_duplicates(["Stkcd", "year"], keep="last")

    return annual[["Stkcd", "year", "net_profit"]].copy()


def main() -> None:
    print("Building annual net profit...")
    annual_profit = build_annual_net_profit()
    print(f"  Annual records: {len(annual_profit)}, companies: {annual_profit['Stkcd'].nunique()}")
    print(f"  Loss (profit<0): {(annual_profit['net_profit'] < 0).sum()} "
          f"({(annual_profit['net_profit'] < 0).mean():.1%})")

    print("Loading panel...")
    panel = pd.read_csv(STAGE1_DIR / "panel_company_year.csv", dtype={"Stkcd": str})
    panel["Stkcd"] = panel["Stkcd"].astype(str).str.zfill(6)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype(int)

    # Merge current-year net profit
    panel = panel.merge(annual_profit, on=["Stkcd", "year"], how="left")

    # Build label: net_profit_next_year < 0
    next_year_profit = annual_profit.copy()
    next_year_profit["year"] = next_year_profit["year"] - 1  # shift: year t+1 profit becomes label for year t
    next_year_profit = next_year_profit.rename(columns={"net_profit": "net_profit_next_year"})

    panel = panel.merge(next_year_profit[["Stkcd", "year", "net_profit_next_year"]], on=["Stkcd", "year"], how="left")
    panel["label_loss_next_year"] = (panel["net_profit_next_year"] < 0).astype(float)
    panel.loc[panel["net_profit_next_year"].isna(), "label_loss_next_year"] = np.nan

    # Add derived features
    eps = 1e-8
    panel["roa"] = np.where(panel["total_assets"].abs() > eps, panel["net_profit"] / panel["total_assets"], np.nan)
    panel["equity_ratio"] = np.where(panel["total_assets"].abs() > eps, panel["total_equity"] / panel["total_assets"], np.nan)
    panel["cashflow_to_assets"] = np.where(panel["total_assets"].abs() > eps, panel["cashflow_operating"] / panel["total_assets"], np.nan)
    panel["is_loss_current_year"] = (panel["net_profit"] < 0).astype(float)
    panel.loc[panel["net_profit"].isna(), "is_loss_current_year"] = np.nan

    # Filter to labeled rows with financial data
    labeled = panel.dropna(subset=["label_loss_next_year"]).copy()
    labeled["label_loss_next_year"] = labeled["label_loss_next_year"].astype(int)
    labeled = labeled[labeled["total_assets"].notna()].copy()

    print(f"\nLabeled panel: {len(labeled)} rows, {labeled['Stkcd'].nunique()} companies")
    print(f"Positive rate (loss next year): {labeled['label_loss_next_year'].mean():.1%}")

    # Time split
    train = labeled[labeled["year"] <= 2018]
    val = labeled[(labeled["year"] >= 2019) & (labeled["year"] <= 2021)]
    test = labeled[(labeled["year"] >= 2022) & (labeled["year"] <= 2024)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUT_DIR / "distress_panel_labeled.csv", index=False, encoding="utf-8-sig")
    train.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(OUT_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    summary = {
        "total_samples": int(len(labeled)),
        "unique_companies": int(labeled["Stkcd"].nunique()),
        "year_range": [int(labeled["year"].min()), int(labeled["year"].max())],
        "positive_rate_overall": float(labeled["label_loss_next_year"].mean()),
        "train": {
            "rows": int(len(train)),
            "companies": int(train["Stkcd"].nunique()),
            "positive_rate": float(train["label_loss_next_year"].mean()) if len(train) else 0,
            "year_range": [int(train["year"].min()), int(train["year"].max())] if len(train) else [],
        },
        "val": {
            "rows": int(len(val)),
            "companies": int(val["Stkcd"].nunique()),
            "positive_rate": float(val["label_loss_next_year"].mean()) if len(val) else 0,
            "year_range": [int(val["year"].min()), int(val["year"].max())] if len(val) else [],
        },
        "test": {
            "rows": int(len(test)),
            "companies": int(test["Stkcd"].nunique()),
            "positive_rate": float(test["label_loss_next_year"].mean()) if len(test) else 0,
            "year_range": [int(test["year"].min()), int(test["year"].max())] if len(test) else [],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nDone. Output:", OUT_DIR)


if __name__ == "__main__":
    main()
