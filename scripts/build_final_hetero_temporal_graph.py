#!/usr/bin/env python3
"""
Build final heterogeneous temporal graph snapshots for listed-company risk modeling.

Design:
- Node type: company (listed firms only).
- Temporal granularity: year.
- Relations:
  1) guarantee: listed -> listed (directed, weighted by guarantee amount)
  2) equity_assoc: company <-> listed counterpart from shareholding chain mapping (directed symmetric)
  3) co_controller: companies sharing same actual controller in the same year (directed symmetric)

Outputs:
  processed/final_hetero_temporal_graph/
    node_mapping.csv
    feature_schema.json
    metadata.json
    snapshots/{year}/node_features.csv
    snapshots/{year}/edges_guarantee.csv
    snapshots/{year}/edges_equity_assoc.csv
    snapshots/{year}/edges_co_controller.csv
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEP = PROJECT_ROOT / "third_party" / "python"
if LOCAL_DEP.exists():
    sys.path.insert(0, str(LOCAL_DEP))

import numpy as np
import pandas as pd


DATA_ROOT = PROJECT_ROOT / "数据"
STAGE1_DIR = PROJECT_ROOT / "processed" / "stage1"
OUT_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"

YEAR_MIN = 2010
YEAR_MAX = 2024


def norm_code(v: object) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if re.fullmatch(r"\d+(\.0+)?", s):
        s = str(int(float(s)))
    if s.isdigit() and len(s) <= 6:
        return s.zfill(6)
    return s or None


def norm_name(v: object) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = re.sub(r"\s+", "", s)
    if not s or s.lower() in {"nan", "none", "0"}:
        return None
    return s


def to_year(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.year


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_xlsx(path: Path, usecols: Iterable[str]) -> pd.DataFrame:
    return pd.read_excel(path, usecols=list(usecols), skiprows=[1, 2], dtype=object, engine="openpyxl")


def build_name_code_map(nodes: pd.DataFrame) -> Dict[str, str]:
    copro = read_xlsx(
        DATA_ROOT / "上市公司基本情况文件180625851(仅供北京大学使用)" / "HLD_Copro.xlsx",
        ["Stkcd", "Conme", "Stknme"],
    )
    copro["Stkcd"] = copro["Stkcd"].map(norm_code)

    all_names = pd.concat(
        [
            nodes[["Stkcd", "Conme", "Stknme"]].copy(),
            copro[["Stkcd", "Conme", "Stknme"]].copy(),
        ],
        ignore_index=True,
    )
    rev: Dict[str, set] = {}
    for _, r in all_names.iterrows():
        c = norm_code(r.get("Stkcd"))
        if not c:
            continue
        for k in ("Conme", "Stknme"):
            n = norm_name(r.get(k))
            if not n:
                continue
            rev.setdefault(n, set()).add(c)
    # Keep only unique name->code mappings to avoid false joins.
    return {n: next(iter(cset)) for n, cset in rev.items() if len(cset) == 1}


def build_guarantee_edges(node_set: Set[str]) -> pd.DataFrame:
    g = pd.read_csv(STAGE1_DIR / "edges_guarantee_listed_to_listed.csv", dtype={"src_stkcd": str, "dst_stkcd": str})
    g["src_stkcd"] = g["src_stkcd"].map(norm_code)
    g["dst_stkcd"] = g["dst_stkcd"].map(norm_code)
    g["year"] = pd.to_numeric(g["year"], errors="coerce").astype("Int64")
    g["amount_raw"] = to_num(g["ActualGuaranteeAmount"]).fillna(0.0)
    g = g[g["src_stkcd"].isin(node_set) & g["dst_stkcd"].isin(node_set)]
    g = g[g["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    g = g[g["src_stkcd"] != g["dst_stkcd"]].copy()
    g["year"] = g["year"].astype(int)

    agg = (
        g.groupby(["year", "src_stkcd", "dst_stkcd"], as_index=False)
        .agg(guarantee_amount_sum=("amount_raw", "sum"), guarantee_event_cnt=("src_stkcd", "size"))
    )
    agg["weight"] = np.log1p(agg["guarantee_amount_sum"].clip(lower=0))
    # Keep positive structural signal even when amount is missing/0.
    agg.loc[agg["weight"] <= 0, "weight"] = 1.0
    return agg


def build_equity_assoc_edges(node_set: Set[str], name_to_code: Dict[str, str]) -> pd.DataFrame:
    sh = read_xlsx(
        DATA_ROOT / "股权关系文件200853795(仅供北京大学使用)" / "HLD_Shrrelchain.xlsx",
        ["Stkcd", "Reptdt", "S0603a", "S0604a", "S0606a"],
    )
    sh["Stkcd"] = sh["Stkcd"].map(norm_code)
    sh = sh[sh["Stkcd"].isin(node_set)].copy()
    sh["year"] = to_year(sh["Reptdt"])
    sh = sh[sh["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    sh["year"] = sh["year"].astype(int)
    sh["ratio"] = to_num(sh["S0606a"])
    sh["ratio"] = sh["ratio"].fillna(1.0)
    sh["ratio"] = sh["ratio"].clip(lower=0)
    sh["weight"] = sh["ratio"] / 100.0
    sh.loc[sh["weight"] <= 0, "weight"] = 1.0

    sh["c1"] = sh["S0603a"].map(norm_name).map(name_to_code)
    sh["c2"] = sh["S0604a"].map(norm_name).map(name_to_code)

    rows = []
    for _, r in sh.iterrows():
        focal = r["Stkcd"]
        year = int(r["year"])
        w = float(r["weight"])
        c1 = r["c1"]
        c2 = r["c2"]
        if c1 and c1 in node_set and c1 != focal:
            rows.append((year, focal, c1, w))
            rows.append((year, c1, focal, w))
        if c2 and c2 in node_set and c2 != focal:
            rows.append((year, focal, c2, w))
            rows.append((year, c2, focal, w))
    if not rows:
        return pd.DataFrame(columns=["year", "src_stkcd", "dst_stkcd", "weight", "equity_link_cnt"])

    ed = pd.DataFrame(rows, columns=["year", "src_stkcd", "dst_stkcd", "w"])
    ed = ed[ed["src_stkcd"] != ed["dst_stkcd"]].copy()
    ed = (
        ed.groupby(["year", "src_stkcd", "dst_stkcd"], as_index=False)
        .agg(weight=("w", "sum"), equity_link_cnt=("w", "size"))
    )
    return ed


def valid_controller_name(s: Optional[str]) -> bool:
    if not s:
        return False
    bad_tokens = ["无控股股东", "无实际控制人", "无实际控制", "不存在实际控制人", "无"]
    return not any(tok in s for tok in bad_tokens)


def build_co_controller_edges(node_set: Set[str]) -> pd.DataFrame:
    ctrl = read_xlsx(
        DATA_ROOT / "上市公司控制人文件200916280(仅供北京大学使用)" / "HLD_Contrshr.xlsx",
        ["Stkcd", "Reptdt", "S0701b"],
    )
    ctrl["Stkcd"] = ctrl["Stkcd"].map(norm_code)
    ctrl = ctrl[ctrl["Stkcd"].isin(node_set)].copy()
    ctrl["year"] = to_year(ctrl["Reptdt"])
    ctrl = ctrl[ctrl["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    ctrl["year"] = ctrl["year"].astype(int)
    ctrl["controller"] = ctrl["S0701b"].map(norm_name)
    ctrl = ctrl[ctrl["controller"].map(valid_controller_name)].copy()
    ctrl = ctrl[["year", "controller", "Stkcd"]].drop_duplicates()

    rows = []
    for (year, controller), sub in ctrl.groupby(["year", "controller"]):
        comps = sorted(sub["Stkcd"].unique().tolist())
        if len(comps) < 2:
            continue
        # undirected clique as bi-directional directed edges
        for a, b in combinations(comps, 2):
            rows.append((year, a, b, 1.0))
            rows.append((year, b, a, 1.0))

    if not rows:
        return pd.DataFrame(columns=["year", "src_stkcd", "dst_stkcd", "weight", "shared_controller_cnt"])

    ed = pd.DataFrame(rows, columns=["year", "src_stkcd", "dst_stkcd", "w"])
    ed = ed[ed["src_stkcd"] != ed["dst_stkcd"]].copy()
    ed = (
        ed.groupby(["year", "src_stkcd", "dst_stkcd"], as_index=False)
        .agg(weight=("w", "sum"), shared_controller_cnt=("w", "size"))
    )
    return ed


def build_node_features(nodes: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_csv(STAGE1_DIR / "panel_company_year.csv", dtype={"Stkcd": str})
    panel["Stkcd"] = panel["Stkcd"].map(norm_code)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel[panel["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    panel["year"] = panel["year"].astype(int)
    panel = panel[panel["Stkcd"].isin(set(nodes["Stkcd"]))].copy()

    # Keep core static + dynamic fields for graph encoder.
    keep_cols = [
        "Stkcd",
        "year",
        "Markettype",
        "Nnindcd",
        "Statco",
        "Regcap",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "revenue_total",
        "revenue_main",
        "cashflow_operating",
        "asset_liability_ratio",
        "guar_event_cnt",
        "guar_amt_sum",
        "guar_amt_mean",
        "guar_listed_target_ratio",
        "rating_event_cnt",
        "rating_agency_nunique",
        "rating_latest_longterm",
        "rating_latest_prospect",
        "has_financial",
        "has_rating",
    ]
    panel = panel[keep_cols].copy()
    return panel


def write_snapshot(
    year: int,
    nodes: pd.DataFrame,
    node_features: pd.DataFrame,
    guar: pd.DataFrame,
    equity: pd.DataFrame,
    ctrl: pd.DataFrame,
    code_to_id: Dict[str, int],
) -> Dict[str, int]:
    ydir = OUT_DIR / "snapshots" / str(year)
    ydir.mkdir(parents=True, exist_ok=True)

    nf = node_features[node_features["year"] == year].copy()
    # Full node inventory each year with active flag.
    full = nodes[["Stkcd"]].copy()
    full["year"] = year
    full = full.merge(nf, on=["Stkcd", "year"], how="left")
    full["node_id"] = full["Stkcd"].map(code_to_id)
    full["is_active"] = full["Markettype"].notna().astype(float)
    full.to_csv(ydir / "node_features.csv", index=False, encoding="utf-8-sig")

    def write_edge(df: pd.DataFrame, name: str) -> int:
        e = df[df["year"] == year].copy()
        if e.empty:
            out = pd.DataFrame(columns=["src_id", "dst_id", "weight"])
            out.to_csv(ydir / f"edges_{name}.csv", index=False, encoding="utf-8-sig")
            return 0
        e["src_id"] = e["src_stkcd"].map(code_to_id)
        e["dst_id"] = e["dst_stkcd"].map(code_to_id)
        keep = [c for c in ["src_id", "dst_id", "weight", "guarantee_amount_sum", "guarantee_event_cnt", "equity_link_cnt", "shared_controller_cnt"] if c in e.columns]
        e = e[keep].dropna(subset=["src_id", "dst_id"]).copy()
        e["src_id"] = e["src_id"].astype(int)
        e["dst_id"] = e["dst_id"].astype(int)
        e.to_csv(ydir / f"edges_{name}.csv", index=False, encoding="utf-8-sig")
        return int(len(e))

    n_guar = write_edge(guar, "guarantee")
    n_equity = write_edge(equity, "equity_assoc")
    n_ctrl = write_edge(ctrl, "co_controller")

    return {
        "year": year,
        "num_nodes": int(len(full)),
        "num_active_nodes": int(full["is_active"].sum()),
        "num_edges_guarantee": n_guar,
        "num_edges_equity_assoc": n_equity,
        "num_edges_co_controller": n_ctrl,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(STAGE1_DIR / "nodes_company.csv", dtype={"Stkcd": str})
    nodes["Stkcd"] = nodes["Stkcd"].map(norm_code)
    nodes = nodes[nodes["Stkcd"].notna()].copy()
    nodes = nodes.sort_values("Stkcd").drop_duplicates("Stkcd")

    node_set = set(nodes["Stkcd"].tolist())
    code_to_id = {c: i for i, c in enumerate(nodes["Stkcd"].tolist())}

    name_to_code = build_name_code_map(nodes)
    guar = build_guarantee_edges(node_set)
    equity = build_equity_assoc_edges(node_set, name_to_code)
    ctrl = build_co_controller_edges(node_set)
    node_features = build_node_features(nodes)

    nodes.assign(node_id=nodes["Stkcd"].map(code_to_id))[["node_id", "Stkcd"]].to_csv(
        OUT_DIR / "node_mapping.csv", index=False, encoding="utf-8-sig"
    )

    feature_schema = {
        "node_feature_columns": [c for c in node_features.columns if c not in {"Stkcd", "year"}],
        "relation_types": ["guarantee", "equity_assoc", "co_controller"],
        "time_granularity": "year",
        "year_range": [YEAR_MIN, YEAR_MAX],
    }
    (OUT_DIR / "feature_schema.json").write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    yearly_stats = []
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        yearly_stats.append(write_snapshot(y, nodes, node_features, guar, equity, ctrl, code_to_id))

    metadata = {
        "num_nodes_total": int(len(nodes)),
        "num_years": YEAR_MAX - YEAR_MIN + 1,
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
        "edge_totals": {
            "guarantee": int(len(guar)),
            "equity_assoc": int(len(equity)),
            "co_controller": int(len(ctrl)),
        },
        "yearly": yearly_stats,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
