#!/usr/bin/env python3
"""
Prepare stage-1 datasets for enterprise credit risk modeling.

Outputs:
  processed/stage1/nodes_company.csv
  processed/stage1/features_financial.csv
  processed/stage1/events_rating.csv
  processed/stage1/edges_guarantee.csv
  processed/stage1/summary.json

Usage:
  python3 scripts/prepare_data_stage1.py
  python3 scripts/prepare_data_stage1.py --include-daily
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def ensure_local_deps(project_root: Path) -> None:
    local_dep = project_root / "third_party" / "python"
    if local_dep.exists():
        sys.path.insert(0, str(local_dep))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ensure_local_deps(PROJECT_ROOT)

import pandas as pd  # noqa: E402


A_SHARE_MARKETS = {1, 4, 16, 32, 64}


def read_csmar_excel(path: Path, usecols: Iterable[str]) -> pd.DataFrame:
    # Most CSMAR exports have two metadata rows under the header row.
    return pd.read_excel(
        path,
        usecols=list(usecols),
        skiprows=[1, 2],
        dtype=object,
        engine="openpyxl",
    )


def normalize_stock_code(v: object) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if re.fullmatch(r"\d+(\.0+)?", s):
        s = str(int(float(s)))
    s = s.strip()
    if s.isdigit() and len(s) <= 6:
        return s.zfill(6)
    return s if s else None


def normalize_name(v: object) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    s = re.sub(r"\s+", "", s)
    return s if s else None


def to_date_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def date_span(series: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    d = pd.to_datetime(series, errors="coerce").dropna()
    if d.empty:
        return None, None
    return d.min().strftime("%Y-%m-%d"), d.max().strftime("%Y-%m-%d")


def build_name_to_code(universe: pd.DataFrame, copro_latest: pd.DataFrame) -> Dict[str, str]:
    name_pairs = []
    for _, r in universe.iterrows():
        name_pairs.append((normalize_name(r.get("Conme")), r.get("Stkcd")))
        name_pairs.append((normalize_name(r.get("Stknme")), r.get("Stkcd")))
    for _, r in copro_latest.iterrows():
        name_pairs.append((normalize_name(r.get("Conme")), r.get("Stkcd")))
        name_pairs.append((normalize_name(r.get("Stknme")), r.get("Stkcd")))

    rev: Dict[str, set] = {}
    for name, code in name_pairs:
        if not name or not code:
            continue
        rev.setdefault(name, set()).add(code)

    out = {}
    for k, v in rev.items():
        if len(v) == 1:
            out[k] = next(iter(v))
    return out


def process_company_nodes(data_root: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    trd_path = data_root / "股票市场" / "TRDNEW_Co.xlsx"
    copro_path = data_root / "上市公司基本情况文件180625851(仅供北京大学使用)" / "HLD_Copro.xlsx"

    trd = read_csmar_excel(
        trd_path,
        ["Stkcd", "Listdt", "DelistedDate", "Markettype", "Statco", "Nnindcd", "Conme", "Stknme"],
    )
    trd["Stkcd"] = trd["Stkcd"].map(normalize_stock_code)
    trd["Markettype"] = pd.to_numeric(trd["Markettype"], errors="coerce")
    trd["Listdt"] = to_date_str(trd["Listdt"])
    trd["DelistedDate"] = to_date_str(trd["DelistedDate"])
    trd = trd[trd["Stkcd"].notna()].copy()
    trd = trd[trd["Markettype"].isin(A_SHARE_MARKETS)].copy()
    trd = trd.sort_values(["Stkcd", "Listdt"]).drop_duplicates("Stkcd", keep="last")

    copro = read_csmar_excel(
        copro_path,
        ["Stkcd", "Reptdt", "Nnindcd", "Regcap", "Conme", "Stknme"],
    )
    copro["Stkcd"] = copro["Stkcd"].map(normalize_stock_code)
    copro["Reptdt"] = to_date_str(copro["Reptdt"])
    copro["Regcap"] = to_float(copro["Regcap"])
    copro = copro[copro["Stkcd"].notna()].copy()
    copro_latest = copro.sort_values(["Stkcd", "Reptdt"]).drop_duplicates("Stkcd", keep="last")

    nodes = trd.merge(
        copro_latest[["Stkcd", "Reptdt", "Regcap"]],
        on="Stkcd",
        how="left",
        suffixes=("", "_copro"),
    )
    nodes.rename(columns={"Reptdt": "latest_profile_date"}, inplace=True)

    out_file = out_dir / "nodes_company.csv"
    nodes.to_csv(out_file, index=False, encoding="utf-8-sig")

    list_min, list_max = date_span(nodes["Listdt"])
    summary = {
        "rows": int(len(nodes)),
        "unique_stkcd": int(nodes["Stkcd"].nunique()),
        "list_date_min": list_min,
        "list_date_max": list_max,
        "regcap_non_null_ratio": float(nodes["Regcap"].notna().mean()),
    }
    return nodes, copro_latest, summary


def process_financial_features(data_root: Path, out_dir: Path, universe_codes: set) -> Dict[str, object]:
    combas = read_csmar_excel(
        data_root / "财务情况" / "FS_Combas.xlsx",
        ["Stkcd", "Accper", "DeclareDate", "A001101000", "A001109000", "A001111000"],
    )
    comins = read_csmar_excel(
        data_root / "利润表201210555(仅供北京大学使用)" / "FS_Comins.xlsx",
        ["Stkcd", "Accper", "DeclareDate", "B001100000", "B001101000"],
    )
    comscfd = read_csmar_excel(
        data_root / "现金流量表(直接法)201221487(仅供北京大学使用)" / "FS_Comscfd.xlsx",
        ["Stkcd", "Accper", "DeclareDate", "C001001000"],
    )

    def clean(df: pd.DataFrame, value_cols: Iterable[str]) -> pd.DataFrame:
        df = df.copy()
        df["Stkcd"] = df["Stkcd"].map(normalize_stock_code)
        df["Accper"] = to_date_str(df["Accper"])
        df["DeclareDate"] = to_date_str(df["DeclareDate"])
        for c in value_cols:
            df[c] = to_float(df[c])
        df = df[df["Stkcd"].isin(universe_codes)].copy()
        df = df.dropna(subset=["Stkcd", "Accper"])
        return df

    combas = clean(combas, ["A001101000", "A001109000", "A001111000"])
    comins = clean(comins, ["B001100000", "B001101000"])
    comscfd = clean(comscfd, ["C001001000"])

    fin = combas.merge(
        comins[["Stkcd", "Accper", "B001100000", "B001101000"]],
        on=["Stkcd", "Accper"],
        how="outer",
    ).merge(
        comscfd[["Stkcd", "Accper", "C001001000"]],
        on=["Stkcd", "Accper"],
        how="outer",
    )
    fin = fin.sort_values(["Stkcd", "Accper"]).drop_duplicates(["Stkcd", "Accper"], keep="last")
    fin.to_csv(out_dir / "features_financial.csv", index=False, encoding="utf-8-sig")

    accper_min, accper_max = date_span(fin["Accper"])
    return {
        "rows": int(len(fin)),
        "unique_stkcd": int(fin["Stkcd"].nunique()),
        "accper_min": accper_min,
        "accper_max": accper_max,
        "coverage_vs_universe": float(fin["Stkcd"].nunique() / max(1, len(universe_codes))),
    }


def process_rating_events(data_root: Path, out_dir: Path, universe_codes: set) -> Dict[str, object]:
    rating = read_csmar_excel(
        data_root / "上市公司信用评级情况表160746111(仅供北京大学使用)" / "DEBT_BOND_RATING.xlsx",
        ["Symbol", "DeclareDate", "RatingDate", "LongTermRating", "RatingProspect", "RatingInstitution"],
    )
    rating["Stkcd"] = rating["Symbol"].map(normalize_stock_code)
    rating["DeclareDate"] = to_date_str(rating["DeclareDate"])
    rating["RatingDate"] = to_date_str(rating["RatingDate"])
    rating["event_date"] = rating["RatingDate"].fillna(rating["DeclareDate"])
    rating = rating[rating["Stkcd"].isin(universe_codes)].copy()
    rating = rating.dropna(subset=["Stkcd", "event_date"])
    rating = rating[
        ["Stkcd", "event_date", "LongTermRating", "RatingProspect", "RatingInstitution", "DeclareDate", "RatingDate"]
    ]
    rating.to_csv(out_dir / "events_rating.csv", index=False, encoding="utf-8-sig")

    dt_min, dt_max = date_span(rating["event_date"])
    return {
        "rows": int(len(rating)),
        "unique_stkcd": int(rating["Stkcd"].nunique()),
        "event_date_min": dt_min,
        "event_date_max": dt_max,
        "long_term_rating_non_null_ratio": float(rating["LongTermRating"].notna().mean()),
    }


def process_guarantee_edges(
    data_root: Path,
    out_dir: Path,
    universe_codes: set,
    name_to_code: Dict[str, str],
) -> Dict[str, object]:
    g = read_csmar_excel(
        data_root / "上市公司对外担保情况表175710486(仅供北京大学使用)" / "STK_Guarantee_Main.xlsx",
        ["Symbol", "Guarantee", "DeclareDate", "StartDate", "EndDate", "ActualGuaranteeAmount", "GuaranteeTypeID", "CurrencyCode"],
    )
    g["src_stkcd"] = g["Symbol"].map(normalize_stock_code)
    g["dst_name"] = g["Guarantee"].map(normalize_name)
    g["DeclareDate"] = to_date_str(g["DeclareDate"])
    g["StartDate"] = to_date_str(g["StartDate"])
    g["EndDate"] = to_date_str(g["EndDate"])
    g["event_date"] = g["StartDate"].fillna(g["DeclareDate"])
    g["ActualGuaranteeAmount"] = to_float(g["ActualGuaranteeAmount"])
    g = g[g["src_stkcd"].isin(universe_codes)].copy()
    g = g.dropna(subset=["src_stkcd", "dst_name", "event_date"])

    g["dst_stkcd"] = g["dst_name"].map(name_to_code)
    g["is_listed_target"] = g["dst_stkcd"].notna()

    edge_cols = [
        "src_stkcd",
        "dst_stkcd",
        "dst_name",
        "event_date",
        "EndDate",
        "ActualGuaranteeAmount",
        "GuaranteeTypeID",
        "CurrencyCode",
        "is_listed_target",
    ]
    edges = g[edge_cols].copy()
    edges.to_csv(out_dir / "edges_guarantee.csv", index=False, encoding="utf-8-sig")

    # Listed-company to listed-company edge subset (sparse but clean).
    edges_ll = edges[edges["dst_stkcd"].notna()].copy()
    edges_ll["year"] = pd.to_datetime(edges_ll["event_date"], errors="coerce").dt.year
    edges_ll.to_csv(out_dir / "edges_guarantee_listed_to_listed.csv", index=False, encoding="utf-8-sig")

    # Source-side yearly guarantee exposure features.
    edges["year"] = pd.to_datetime(edges["event_date"], errors="coerce").dt.year
    gy = (
        edges.dropna(subset=["src_stkcd", "year"])
        .groupby(["src_stkcd", "year"], as_index=False)
        .agg(
            guar_event_cnt=("src_stkcd", "size"),
            guar_amt_sum=("ActualGuaranteeAmount", "sum"),
            guar_amt_mean=("ActualGuaranteeAmount", "mean"),
            guar_listed_target_cnt=("is_listed_target", "sum"),
            guar_nonlisted_target_cnt=("is_listed_target", lambda x: (~x).sum()),
        )
    )
    gy["guar_amt_sum"] = gy["guar_amt_sum"].fillna(0.0)
    gy["guar_amt_mean"] = gy["guar_amt_mean"].fillna(0.0)
    gy["guar_listed_target_ratio"] = gy["guar_listed_target_cnt"] / gy["guar_event_cnt"].where(gy["guar_event_cnt"] > 0, 1)
    gy.rename(columns={"src_stkcd": "src_stkcd"}, inplace=True)
    gy.to_csv(out_dir / "features_guarantee_yearly.csv", index=False, encoding="utf-8-sig")

    dt_min, dt_max = date_span(edges["event_date"])
    return {
        "rows": int(len(edges)),
        "unique_src_stkcd": int(edges["src_stkcd"].nunique()),
        "listed_target_match_ratio": float(edges["is_listed_target"].mean()),
        "listed_to_listed_rows": int(len(edges_ll)),
        "yearly_feature_rows": int(len(gy)),
        "event_date_min": dt_min,
        "event_date_max": dt_max,
    }


def process_daily_returns(
    data_root: Path,
    out_dir: Path,
    universe_codes: set,
) -> Dict[str, object]:
    daily_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("日个股回报率文件")])
    daily_files = []
    for d in daily_dirs:
        daily_files.extend(sorted(d.glob("TRD_Dalyr*.xlsx")))

    out_file = out_dir / "market_daily.csv"
    if out_file.exists():
        out_file.unlink()

    total_rows = 0
    all_stkcd = set()
    min_date = None
    max_date = None

    for i, path in enumerate(daily_files):
        df = read_csmar_excel(
            path,
            ["Stkcd", "Trddt", "Dretwd", "Dretnd", "Clsprc", "Trdsta", "Markettype"],
        )
        df["Stkcd"] = df["Stkcd"].map(normalize_stock_code)
        df["Trddt"] = to_date_str(df["Trddt"])
        df["Dretwd"] = to_float(df["Dretwd"])
        df["Dretnd"] = to_float(df["Dretnd"])
        df["Clsprc"] = to_float(df["Clsprc"])
        df["Markettype"] = pd.to_numeric(df["Markettype"], errors="coerce")
        df = df[df["Stkcd"].isin(universe_codes)].copy()
        df = df[df["Trddt"].notna()]

        total_rows += len(df)
        all_stkcd.update(df["Stkcd"].dropna().unique().tolist())
        dt_min, dt_max = date_span(df["Trddt"])
        if dt_min and (min_date is None or dt_min < min_date):
            min_date = dt_min
        if dt_max and (max_date is None or dt_max > max_date):
            max_date = dt_max

        df.to_csv(out_file, mode="a", header=(i == 0), index=False, encoding="utf-8-sig")

    return {
        "rows": int(total_rows),
        "unique_stkcd": int(len(all_stkcd)),
        "trddt_min": min_date,
        "trddt_max": max_date,
        "files_processed": len(daily_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stage-1 datasets for credit risk graph modeling.")
    parser.add_argument("--data-root", default="数据", help="Root directory of raw datasets.")
    parser.add_argument("--out-dir", default="processed/stage1", help="Output directory.")
    parser.add_argument("--skip-financial", action="store_true", help="Skip building features_financial.csv.")
    parser.add_argument("--skip-rating", action="store_true", help="Skip building events_rating.csv.")
    parser.add_argument("--skip-guarantee", action="store_true", help="Skip building edges_guarantee.csv.")
    parser.add_argument("--include-daily", action="store_true", help="Also build market_daily.csv (large, slow).")
    args = parser.parse_args()

    data_root = (PROJECT_ROOT / args.data_root).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, copro_latest, nodes_summary = process_company_nodes(data_root, out_dir)
    universe_codes = set(nodes["Stkcd"].dropna().unique().tolist())
    name_to_code = build_name_to_code(nodes, copro_latest)

    summary = {"nodes_company": nodes_summary}

    if not args.skip_financial:
        summary["features_financial"] = process_financial_features(data_root, out_dir, universe_codes)
    if not args.skip_rating:
        summary["events_rating"] = process_rating_events(data_root, out_dir, universe_codes)
    if not args.skip_guarantee:
        summary["edges_guarantee"] = process_guarantee_edges(data_root, out_dir, universe_codes, name_to_code)

    if args.include_daily:
        summary["market_daily"] = process_daily_returns(data_root, out_dir, universe_codes)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
