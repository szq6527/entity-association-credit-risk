#!/usr/bin/env python3
"""
Build final heterogeneous temporal graph snapshots for listed-company risk modeling.

Design:
- Node type: company (listed firms only).
- Temporal granularity: year.
- Relations:
  1) guarantee: listed -> listed (directed, weighted by guarantee amount)
  2) shared_nonlisted_guarantee: companies sharing non-listed guaranteed targets in the same year
  3) equity_assoc: company <-> listed counterpart from shareholding chain mapping (directed symmetric)
  4) equity_change: company links from equity change events (directed symmetric after mapping counterpart names)
  5) co_controller: companies sharing same actual controller in the same year (directed symmetric)
  6) market_corr: stock return correlation edge from daily return panel (directed by top-k neighbors)

Outputs:
  processed/final_hetero_temporal_graph/
    node_mapping.csv
    feature_schema.json
    metadata.json
    snapshots/{year}/node_features.csv
    snapshots/{year}/edges_guarantee.csv
    snapshots/{year}/edges_shared_nonlisted_guarantee.csv
    snapshots/{year}/edges_equity_assoc.csv
    snapshots/{year}/edges_equity_change.csv
    snapshots/{year}/edges_co_controller.csv
    snapshots/{year}/edges_market_corr.csv
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEP = PROJECT_ROOT / "third_party" / "python"
if LOCAL_DEP.exists():
    # Keep system scientific stack (numpy/pandas) ahead of vendored deps.
    sys.path.append(str(LOCAL_DEP))

import numpy as np
import pandas as pd


DATA_ROOT = PROJECT_ROOT / "数据"
STAGE1_DIR = PROJECT_ROOT / "processed" / "stage1"
OUT_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"

YEAR_MIN = 2010
YEAR_MAX = 2024
MARKET_CORR_MIN_ABS = 0.35
MARKET_CORR_TOPK = 10
MARKET_CORR_MIN_DAYS = 80
MARKET_CORR_MAX_ABS_RET = 0.30
MARKET_CORR_WINSOR_Q = 0.001


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


def split_entity_names(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [x for x in re.split(r"[;；,，、/|]+", v) if x]


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


def build_shared_nonlisted_guarantee_edges(node_set: Set[str]) -> pd.DataFrame:
    p = STAGE1_DIR / "edges_guarantee.csv"
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "year",
                "src_stkcd",
                "dst_stkcd",
                "weight",
                "shared_nonlisted_target_cnt",
                "shared_nonlisted_target_amt_sum",
            ]
        )

    g = pd.read_csv(p, dtype={"src_stkcd": str, "dst_stkcd": str, "dst_name": str}, low_memory=False)
    g["src_stkcd"] = g["src_stkcd"].map(norm_code)
    g["dst_stkcd"] = g["dst_stkcd"].map(norm_code)
    g["dst_name"] = g["dst_name"].map(norm_name)
    g["year"] = to_year(g["event_date"])
    g["amount_raw"] = to_num(g.get("ActualGuaranteeAmount", pd.Series(index=g.index, dtype=object))).fillna(0.0).clip(lower=0)
    g = g[g["src_stkcd"].isin(node_set)].copy()
    g = g[g["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    g = g[g["dst_name"].notna()].copy()
    # Keep only non-listed guaranteed targets.
    g = g[~g["dst_stkcd"].isin(node_set)].copy()
    g["year"] = g["year"].astype(int)

    by_src_target = (
        g.groupby(["year", "dst_name", "src_stkcd"], as_index=False)
        .agg(amount_sum=("amount_raw", "sum"), event_cnt=("src_stkcd", "size"))
    )

    rows = []
    for (year, dst_name), sub in by_src_target.groupby(["year", "dst_name"]):
        comps = sub[["src_stkcd", "amount_sum"]].drop_duplicates("src_stkcd")
        if len(comps) < 2:
            continue
        items = list(zip(comps["src_stkcd"].tolist(), comps["amount_sum"].tolist()))
        for i in range(len(items)):
            a, a_amt = items[i]
            for j in range(i + 1, len(items)):
                b, b_amt = items[j]
                min_amt = float(min(a_amt, b_amt))
                w = float(np.log1p(max(min_amt, 0.0)))
                if w <= 0:
                    w = 1.0
                rows.append((int(year), a, b, w, min_amt))
                rows.append((int(year), b, a, w, min_amt))

    if not rows:
        return pd.DataFrame(
            columns=[
                "year",
                "src_stkcd",
                "dst_stkcd",
                "weight",
                "shared_nonlisted_target_cnt",
                "shared_nonlisted_target_amt_sum",
            ]
        )

    ed = pd.DataFrame(rows, columns=["year", "src_stkcd", "dst_stkcd", "w", "min_amt"])
    ed = ed[ed["src_stkcd"] != ed["dst_stkcd"]].copy()
    ed = (
        ed.groupby(["year", "src_stkcd", "dst_stkcd"], as_index=False)
        .agg(
            weight=("w", "sum"),
            shared_nonlisted_target_cnt=("w", "size"),
            shared_nonlisted_target_amt_sum=("min_amt", "sum"),
        )
    )
    return ed


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


def build_equity_change_edges(node_set: Set[str], name_to_code: Dict[str, str]) -> pd.DataFrame:
    p = DATA_ROOT / "股权变更情况文件200923078(仅供北京大学使用)" / "HLD_Chgequity.xlsx"
    if not p.exists():
        return pd.DataFrame(columns=["year", "src_stkcd", "dst_stkcd", "weight", "equity_change_link_cnt"])

    chg = read_xlsx(p, ["Stkcd", "S0801a", "S0802a", "S0805a", "S0808a"])
    chg["Stkcd"] = chg["Stkcd"].map(norm_code)
    chg = chg[chg["Stkcd"].isin(node_set)].copy()
    chg["year"] = to_year(chg["S0805a"])
    chg = chg[chg["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    chg["year"] = chg["year"].astype(int)

    ratio = to_num(chg["S0808a"]).abs()
    ratio = ratio.where(ratio <= 1.0, ratio / 100.0)
    chg["w"] = np.log1p(ratio.fillna(0.0).clip(lower=0.0) * 100.0)
    chg.loc[chg["w"] <= 0, "w"] = 1.0

    rows = []
    for _, r in chg.iterrows():
        src = r["Stkcd"]
        year = int(r["year"])
        w = float(r["w"])
        for col in ("S0801a", "S0802a"):
            names = split_entity_names(norm_name(r.get(col)))
            for n in names:
                dst = name_to_code.get(n)
                if not dst or dst not in node_set or dst == src:
                    continue
                rows.append((year, src, dst, w))
                rows.append((year, dst, src, w))

    if not rows:
        return pd.DataFrame(columns=["year", "src_stkcd", "dst_stkcd", "weight", "equity_change_link_cnt"])

    ed = pd.DataFrame(rows, columns=["year", "src_stkcd", "dst_stkcd", "w"])
    ed = ed[ed["src_stkcd"] != ed["dst_stkcd"]].copy()
    ed = (
        ed.groupby(["year", "src_stkcd", "dst_stkcd"], as_index=False)
        .agg(weight=("w", "sum"), equity_change_link_cnt=("w", "size"))
    )
    return ed


def valid_controller_name(s: Optional[str]) -> bool:
    if s is None or pd.isna(s):
        return False
    s = str(s).strip()
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


def load_market_daily(node_set: Set[str]) -> pd.DataFrame:
    cached = STAGE1_DIR / "market_daily.csv"
    if cached.exists():
        mk = pd.read_csv(cached, dtype={"Stkcd": str}, low_memory=False)
    else:
        files = []
        for d in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("日个股回报率文件")]):
            files.extend(sorted(d.glob("TRD_Dalyr*.xlsx")))

        parts = []
        for i, p in enumerate(files, start=1):
            print(f"[market_corr] loading daily file {i}/{len(files)}: {p.name}")
            df = read_xlsx(p, ["Stkcd", "Trddt", "Dretwd", "Dretnd", "Trdsta"])
            parts.append(df)
        mk = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Stkcd", "Trddt", "Dretwd", "Dretnd", "Trdsta"])

    if mk.empty:
        return pd.DataFrame(columns=["Stkcd", "Trddt", "ret", "year"])

    mk["Stkcd"] = mk["Stkcd"].map(norm_code)
    mk["Trddt"] = pd.to_datetime(mk["Trddt"], errors="coerce")
    mk["Trdsta"] = pd.to_numeric(mk.get("Trdsta", pd.Series(index=mk.index, dtype=object)), errors="coerce")
    mk["ret"] = to_num(mk.get("Dretwd", pd.Series(index=mk.index, dtype=object)))
    alt = to_num(mk.get("Dretnd", pd.Series(index=mk.index, dtype=object)))
    mk["ret"] = mk["ret"].fillna(alt)
    mk = mk[mk["Stkcd"].isin(node_set)].copy()
    mk = mk.dropna(subset=["Stkcd", "Trddt", "ret"]).copy()
    # Remove obvious anomalies so pairwise correlation is stable.
    mk = mk[np.isfinite(mk["ret"]) & (mk["ret"].abs() <= MARKET_CORR_MAX_ABS_RET)].copy()
    mk["year"] = mk["Trddt"].dt.year
    mk = mk[mk["year"].between(YEAR_MIN, YEAR_MAX)].copy()

    if not mk.empty:
        q = float(MARKET_CORR_WINSOR_Q)
        if 0.0 < q < 0.5:
            lo = mk.groupby("year")["ret"].transform(lambda s: s.quantile(q))
            hi = mk.groupby("year")["ret"].transform(lambda s: s.quantile(1.0 - q))
            mk["ret"] = mk["ret"].clip(lower=lo, upper=hi)

    mk = mk[["Stkcd", "Trddt", "ret", "year"]].drop_duplicates(["Stkcd", "Trddt"], keep="last")
    return mk


def build_market_corr_edges(node_set: Set[str]) -> pd.DataFrame:
    mk = load_market_daily(node_set)
    if mk.empty:
        return pd.DataFrame(columns=["year", "src_stkcd", "dst_stkcd", "weight", "corr_raw", "market_link_cnt"])

    by_year = {int(y): g[["Trddt", "Stkcd", "ret"]].copy() for y, g in mk.groupby("year", sort=True)}
    rows = []
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        sub = by_year.get(year)
        if sub is None:
            continue
        if sub.empty:
            continue
        wide = sub.pivot_table(index="Trddt", columns="Stkcd", values="ret", aggfunc="mean")
        valid_cols = wide.columns[wide.count(axis=0) >= MARKET_CORR_MIN_DAYS]
        wide = wide[valid_cols]
        if wide.shape[1] < 2:
            continue

        corr = wide.corr(min_periods=MARKET_CORR_MIN_DAYS)
        cols = corr.columns.to_numpy()
        mat = corr.to_numpy(dtype=np.float32, copy=True)
        valid_obs = (~wide.isna()).to_numpy(dtype=bool, copy=False)
        np.fill_diagonal(mat, np.nan)

        for i, src in enumerate(cols):
            row = mat[i]
            abs_row = np.abs(row)
            cand = np.where(np.isfinite(abs_row) & (abs_row >= MARKET_CORR_MIN_ABS))[0]
            if cand.size == 0:
                continue
            if cand.size > MARKET_CORR_TOPK:
                top_idx = np.argpartition(abs_row[cand], -MARKET_CORR_TOPK)[-MARKET_CORR_TOPK:]
                cand = cand[top_idx]
            # Keep deterministic ordering for reproducibility.
            cand = cand[np.argsort(-abs_row[cand])]
            for j in cand:
                raw = float(row[j])
                w = float(abs_row[j])
                if not np.isfinite(w) or w <= 0:
                    continue
                common_days = int(np.count_nonzero(valid_obs[:, i] & valid_obs[:, j]))
                rows.append((int(year), str(src), str(cols[j]), w, raw, common_days))

        print(f"[market_corr] year={year}, n_stock={wide.shape[1]}, edge_rows_so_far={len(rows)}")

    if not rows:
        return pd.DataFrame(columns=["year", "src_stkcd", "dst_stkcd", "weight", "corr_raw", "common_days", "market_link_cnt"])

    ed = pd.DataFrame(rows, columns=["year", "src_stkcd", "dst_stkcd", "weight", "corr_raw", "common_days"])
    ed = ed[ed["src_stkcd"] != ed["dst_stkcd"]].copy()
    ed = (
        ed.groupby(["year", "src_stkcd", "dst_stkcd"], as_index=False)
        .agg(
            weight=("weight", "mean"),
            corr_raw=("corr_raw", "mean"),
            common_days=("common_days", "max"),
            market_link_cnt=("weight", "size"),
        )
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
    shared_nonlisted_guar: pd.DataFrame,
    equity: pd.DataFrame,
    equity_change: pd.DataFrame,
    ctrl: pd.DataFrame,
    market: pd.DataFrame,
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
        keep = [
            c
            for c in [
                "src_id",
                "dst_id",
                "weight",
                "guarantee_amount_sum",
                "guarantee_event_cnt",
                "shared_nonlisted_target_cnt",
                "shared_nonlisted_target_amt_sum",
                "equity_link_cnt",
                "equity_change_link_cnt",
                "shared_controller_cnt",
                "corr_raw",
                "common_days",
                "market_link_cnt",
            ]
            if c in e.columns
        ]
        e = e[keep].dropna(subset=["src_id", "dst_id"]).copy()
        e["src_id"] = e["src_id"].astype(int)
        e["dst_id"] = e["dst_id"].astype(int)
        e.to_csv(ydir / f"edges_{name}.csv", index=False, encoding="utf-8-sig")
        return int(len(e))

    n_guar = write_edge(guar, "guarantee")
    n_shared_nonlisted_guar = write_edge(shared_nonlisted_guar, "shared_nonlisted_guarantee")
    n_equity = write_edge(equity, "equity_assoc")
    n_equity_change = write_edge(equity_change, "equity_change")
    n_ctrl = write_edge(ctrl, "co_controller")
    n_market = write_edge(market, "market_corr")

    return {
        "year": year,
        "num_nodes": int(len(full)),
        "num_active_nodes": int(full["is_active"].sum()),
        "num_edges_guarantee": n_guar,
        "num_edges_shared_nonlisted_guarantee": n_shared_nonlisted_guar,
        "num_edges_equity_assoc": n_equity,
        "num_edges_equity_change": n_equity_change,
        "num_edges_co_controller": n_ctrl,
        "num_edges_market_corr": n_market,
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
    shared_nonlisted_guar = build_shared_nonlisted_guarantee_edges(node_set)
    equity = build_equity_assoc_edges(node_set, name_to_code)
    equity_change = build_equity_change_edges(node_set, name_to_code)
    ctrl = build_co_controller_edges(node_set)
    market = build_market_corr_edges(node_set)
    node_features = build_node_features(nodes)

    nodes.assign(node_id=nodes["Stkcd"].map(code_to_id))[["node_id", "Stkcd"]].to_csv(
        OUT_DIR / "node_mapping.csv", index=False, encoding="utf-8-sig"
    )

    feature_schema = {
        "node_feature_columns": [c for c in node_features.columns if c not in {"Stkcd", "year"}],
        "relation_types": [
            "guarantee",
            "shared_nonlisted_guarantee",
            "equity_assoc",
            "equity_change",
            "co_controller",
            "market_corr",
        ],
        "time_granularity": "year",
        "year_range": [YEAR_MIN, YEAR_MAX],
    }
    (OUT_DIR / "feature_schema.json").write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    yearly_stats = []
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        yearly_stats.append(
            write_snapshot(
                y,
                nodes,
                node_features,
                guar,
                shared_nonlisted_guar,
                equity,
                equity_change,
                ctrl,
                market,
                code_to_id,
            )
        )

    metadata = {
        "num_nodes_total": int(len(nodes)),
        "num_years": YEAR_MAX - YEAR_MIN + 1,
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
        "edge_totals": {
            "guarantee": int(len(guar)),
            "shared_nonlisted_guarantee": int(len(shared_nonlisted_guar)),
            "equity_assoc": int(len(equity)),
            "equity_change": int(len(equity_change)),
            "co_controller": int(len(ctrl)),
            "market_corr": int(len(market)),
        },
        "yearly": yearly_stats,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
