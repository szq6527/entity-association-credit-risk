#!/usr/bin/env python3
"""
Build listed-company yearly training panel from processed/stage1 outputs.

Output:
  processed/stage1/panel_company_year.csv
  processed/stage1/panel_company_year_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_ROOT / "processed" / "stage1"


def year_from_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.year


def build_base_panel(nodes: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    list_year = year_from_date(nodes["Listdt"])
    delist_year = year_from_date(nodes["DelistedDate"]).fillna(end_year).astype(int)

    for r, ly, dy in zip(nodes.to_dict("records"), list_year, delist_year):
        stkcd = str(r["Stkcd"]).zfill(6)
        if pd.isna(ly):
            continue
        y0 = max(int(ly), start_year)
        y1 = min(int(dy), end_year)
        if y0 > y1:
            continue
        for y in range(y0, y1 + 1):
            rows.append(
                {
                    "Stkcd": stkcd,
                    "year": y,
                    "Markettype": r.get("Markettype"),
                    "Nnindcd": r.get("Nnindcd"),
                    "Statco": r.get("Statco"),
                    "Regcap": r.get("Regcap"),
                }
            )
    return pd.DataFrame(rows)


def build_financial_yearly(fin: pd.DataFrame) -> pd.DataFrame:
    fin = fin.copy()
    fin["Stkcd"] = fin["Stkcd"].astype(str).str.zfill(6)
    fin["acc_date"] = pd.to_datetime(fin["Accper"], errors="coerce")
    fin["year"] = fin["acc_date"].dt.year
    fin = fin.dropna(subset=["Stkcd", "year", "acc_date"])
    fin["year"] = fin["year"].astype(int)

    value_cols = [
        "A001101000",
        "A001109000",
        "A001111000",
        "B001100000",
        "B001101000",
        "C001001000",
    ]
    for c in value_cols:
        if c in fin.columns:
            fin[c] = pd.to_numeric(fin[c], errors="coerce")

    fin = fin.sort_values(["Stkcd", "year", "acc_date"]).drop_duplicates(["Stkcd", "year"], keep="last")
    fin = fin[
        [
            "Stkcd",
            "year",
            "A001101000",
            "A001109000",
            "A001111000",
            "B001100000",
            "B001101000",
            "C001001000",
        ]
    ].copy()
    fin.rename(
        columns={
            "A001101000": "total_assets",
            "A001109000": "total_liabilities",
            "A001111000": "total_equity",
            "B001100000": "revenue_total",
            "B001101000": "revenue_main",
            "C001001000": "cashflow_operating",
        },
        inplace=True,
    )
    fin["asset_liability_ratio"] = np.where(
        fin["total_assets"] > 0,
        fin["total_liabilities"] / fin["total_assets"],
        np.nan,
    )
    return fin


def build_rating_yearly(rating: pd.DataFrame) -> pd.DataFrame:
    rating = rating.copy()
    rating["Stkcd"] = rating["Stkcd"].astype(str).str.zfill(6)
    rating["event_date"] = pd.to_datetime(rating["event_date"], errors="coerce")
    rating["year"] = rating["event_date"].dt.year
    rating = rating.dropna(subset=["Stkcd", "event_date", "year"])
    rating["year"] = rating["year"].astype(int)

    cnt = (
        rating.groupby(["Stkcd", "year"], as_index=False)
        .agg(
            rating_event_cnt=("Stkcd", "size"),
            rating_agency_nunique=("RatingInstitution", "nunique"),
            rating_longterm_nonnull_cnt=("LongTermRating", lambda s: s.notna().sum()),
        )
    )

    latest = (
        rating.sort_values(["Stkcd", "year", "event_date"])
        .drop_duplicates(["Stkcd", "year"], keep="last")[
            ["Stkcd", "year", "LongTermRating", "RatingProspect"]
        ]
        .rename(
            columns={
                "LongTermRating": "rating_latest_longterm",
                "RatingProspect": "rating_latest_prospect",
            }
        )
    )
    out = cnt.merge(latest, on=["Stkcd", "year"], how="left")
    return out


def main() -> None:
    nodes = pd.read_csv(STAGE1_DIR / "nodes_company.csv", dtype={"Stkcd": str})
    fin = pd.read_csv(STAGE1_DIR / "features_financial.csv", dtype={"Stkcd": str})
    rating = pd.read_csv(STAGE1_DIR / "events_rating.csv", dtype={"Stkcd": str})
    guar = pd.read_csv(STAGE1_DIR / "features_guarantee_yearly.csv", dtype={"src_stkcd": str})

    panel_start_year = 2010
    panel_end_year = 2025

    base = build_base_panel(nodes, panel_start_year, panel_end_year)
    fin_y = build_financial_yearly(fin)
    rating_y = build_rating_yearly(rating)

    guar = guar.copy()
    guar["Stkcd"] = guar["src_stkcd"].astype(str).str.zfill(6)
    guar["year"] = pd.to_numeric(guar["year"], errors="coerce").astype("Int64")
    guar = guar[(guar["year"] >= panel_start_year) & (guar["year"] <= panel_end_year)].copy()
    guar["year"] = guar["year"].astype(int)
    guar = guar.drop(columns=["src_stkcd"])

    panel = (
        base.merge(fin_y, on=["Stkcd", "year"], how="left")
        .merge(guar, on=["Stkcd", "year"], how="left")
        .merge(rating_y, on=["Stkcd", "year"], how="left")
    )

    # No observed label table yet; reserve explicit target slots.
    panel["label_default_next_year"] = pd.NA
    panel["label_st_next_year"] = pd.NA

    # Missing guarantee activity means 0 exposure in that year.
    zero_cols = [
        "guar_event_cnt",
        "guar_amt_sum",
        "guar_amt_mean",
        "guar_listed_target_cnt",
        "guar_nonlisted_target_cnt",
        "guar_listed_target_ratio",
        "rating_event_cnt",
        "rating_agency_nunique",
        "rating_longterm_nonnull_cnt",
    ]
    for c in zero_cols:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)

    panel["has_financial"] = panel["total_assets"].notna().astype(int)
    panel["has_rating"] = (panel["rating_event_cnt"] > 0).astype(int)

    panel = panel.sort_values(["Stkcd", "year"]).reset_index(drop=True)
    panel.to_csv(STAGE1_DIR / "panel_company_year.csv", index=False, encoding="utf-8-sig")

    summary = {
        "panel_rows": int(len(panel)),
        "panel_unique_stkcd": int(panel["Stkcd"].nunique()),
        "panel_year_min": int(panel["year"].min()),
        "panel_year_max": int(panel["year"].max()),
        "financial_coverage_ratio": float(panel["has_financial"].mean()),
        "rating_coverage_ratio": float(panel["has_rating"].mean()),
        "guarantee_exposure_ratio": float((panel["guar_event_cnt"] > 0).mean()),
    }
    (STAGE1_DIR / "panel_company_year_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

