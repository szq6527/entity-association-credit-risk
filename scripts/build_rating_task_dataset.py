#!/usr/bin/env python3
"""
Build labeled dataset for credit-rating prediction tasks from panel_company_year.csv.

Outputs (under processed/stage1/rating_task/):
  rating_panel_labeled.csv
  train.csv
  val.csv
  test.csv
  summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_ROOT / "processed" / "stage1" / "panel_company_year.csv"
OUT_DIR = PROJECT_ROOT / "processed" / "stage1" / "rating_task"


RATING_TO_SCORE = {
    "D": 1,
    "C": 2,
    "CC": 3,
    "CCC": 4,
    "B-": 5,
    "B": 6,
    "B+": 7,
    "BB-": 8,
    "BB": 9,
    "BB+": 10,
    "BBB-": 11,
    "BBB": 12,
    "BBB+": 13,
    "A-": 14,
    "A": 15,
    "A+": 16,
    "AA-": 17,
    "AA": 18,
    "AA+": 19,
    "AAA-": 20,
    "AAA": 21,
    "AAA+": 22,
}

# Longest-first pattern to avoid partial mis-match.
RATING_PATTERN = re.compile(
    r"(AAA\+|AAA-|AAA|AA\+|AA-|AA|A\+|A-|A|BBB\+|BBB-|BBB|BB\+|BB-|BB|B\+|B-|B|CCC|CC|C|D)"
)

PROSPECT_TO_CODE = {"负面": -1, "稳定": 0, "正面": 1, "待决": 2}


def normalize_rating(v: object) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).upper().strip().replace(" ", "")
    # A-1 is short-term rating; exclude from long-term transition labels.
    if "A-1" in s:
        return None
    m = RATING_PATTERN.search(s)
    if not m:
        return None
    return m.group(1)


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["year"] >= 2010) & (df["year"] <= 2018)].copy()
    val = df[(df["year"] >= 2019) & (df["year"] <= 2021)].copy()
    test = df[(df["year"] >= 2022) & (df["year"] <= 2024)].copy()
    return train, val, test


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, dtype={"Stkcd": str})
    panel["Stkcd"] = panel["Stkcd"].astype(str).str.zfill(6)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype(int)

    panel["rating_norm"] = panel["rating_latest_longterm"].map(normalize_rating)
    panel["rating_score"] = panel["rating_norm"].map(RATING_TO_SCORE)
    panel["prospect_code"] = panel["rating_latest_prospect"].map(PROSPECT_TO_CODE)

    nxt = panel[["Stkcd", "year", "rating_norm", "rating_score"]].copy()
    nxt["year"] = nxt["year"] - 1
    nxt = nxt.rename(
        columns={
            "rating_norm": "label_rating_next_year",
            "rating_score": "label_rating_score_next_year",
        }
    )

    ds = panel.merge(nxt, on=["Stkcd", "year"], how="left")
    ds["label_downgrade_next_year"] = np.where(
        ds["label_rating_score_next_year"].notna() & ds["rating_score"].notna(),
        (ds["label_rating_score_next_year"] < ds["rating_score"]).astype(int),
        np.nan,
    )

    # Keep rows that have current-year rating and next-year rating labels.
    # year <= 2024 ensures t+1 exists inside panel horizon.
    ds = ds[(ds["year"] >= 2010) & (ds["year"] <= 2024)].copy()
    ds = ds[ds["rating_score"].notna() & ds["label_rating_score_next_year"].notna()].copy()

    # Drop label-placeholder columns to avoid confusion.
    for c in ["label_default_next_year", "label_st_next_year"]:
        if c in ds.columns:
            ds.drop(columns=[c], inplace=True)

    ds = ds.sort_values(["year", "Stkcd"]).reset_index(drop=True)
    ds.to_csv(OUT_DIR / "rating_panel_labeled.csv", index=False, encoding="utf-8-sig")

    train, val, test = split_by_year(ds)
    train.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(OUT_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows_total": int(len(ds)),
        "companies_total": int(ds["Stkcd"].nunique()),
        "year_min": int(ds["year"].min()) if len(ds) else None,
        "year_max": int(ds["year"].max()) if len(ds) else None,
        "downgrade_rate": float(ds["label_downgrade_next_year"].mean()) if len(ds) else None,
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
        "train_downgrade_rate": float(train["label_downgrade_next_year"].mean()) if len(train) else None,
        "val_downgrade_rate": float(val["label_downgrade_next_year"].mean()) if len(val) else None,
        "test_downgrade_rate": float(test["label_downgrade_next_year"].mean()) if len(test) else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

