#!/usr/bin/env python3
"""
Build neighbor-contagion features for each company-year (v2).

Key improvements over v1:
  1. Time-lagged: use LAST year's neighbor distress (no data leakage, leading indicator)
  2. Directional: separate in-neighbors vs out-neighbors for directed edges
  3. Multi-year: include both current-year and lagged features
  4. Financial exposure: neighbor's financial health weighted features

For each edge type, compute:
  Current year (t):
    - nbr_cnt: number of neighbors
    - loss_cnt/ratio: neighbors with net_profit < 0 in year t
    - mean_roa/min_roa: neighbors' ROA in year t
  Lagged (t-1):
    - lag_loss_cnt/ratio: neighbors with net_profit < 0 in year t-1
    - lag_mean_roa/min_roa: neighbors' ROA in year t-1
  Directional (for guarantee edges):
    - in_loss_ratio: % of companies guaranteeing ME that are in distress
    - out_loss_ratio: % of companies I guarantee that are in distress
  Cross-type aggregates
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_ROOT / "processed" / "stage1"
GRAPH_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"
OUT_PATH = STAGE1_DIR / "distress_task" / "contagion_features.csv"

EDGE_TYPES = [
    "guarantee",
    "shared_nonlisted_guarantee",
    "equity_assoc",
    "equity_change",
    "co_controller",
    "market_corr",
    "industry",
]

# Directed edge types where in vs out matters
DIRECTED_TYPES = {"guarantee", "equity_assoc", "equity_change"}


def compute_nbr_stats(
    nbrs: List[str],
    annual_lookup: Dict[Tuple[str, int], dict],
    year: int,
    prefix: str,
) -> Dict[str, float]:
    """Compute distress statistics for a list of neighbors at a given year."""
    feats: Dict[str, float] = {}
    nbr_count = len(nbrs)
    if nbr_count == 0:
        return feats

    loss_count = 0
    roas = []
    matched = 0
    for nb in nbrs:
        info = annual_lookup.get((nb, year))
        if info:
            matched += 1
            loss_count += info["is_loss"]
            if not np.isnan(info["roa"]):
                roas.append(info["roa"])

    feats[f"{prefix}_nbr_cnt"] = nbr_count
    feats[f"{prefix}_loss_cnt"] = loss_count
    feats[f"{prefix}_loss_ratio"] = loss_count / nbr_count if nbr_count else 0
    feats[f"{prefix}_mean_roa"] = float(np.mean(roas)) if roas else np.nan
    feats[f"{prefix}_min_roa"] = float(np.min(roas)) if roas else np.nan
    feats[f"{prefix}_std_roa"] = float(np.std(roas)) if len(roas) > 1 else 0.0
    return feats


def main() -> None:
    # Load node mapping
    nm = pd.read_csv(GRAPH_DIR / "node_mapping.csv", dtype={"Stkcd": str})
    nm["Stkcd"] = nm["Stkcd"].astype(str).str.zfill(6)
    id_to_code = dict(zip(nm["node_id"], nm["Stkcd"]))

    # Load net profit for loss status
    profit = pd.read_csv(STAGE1_DIR / "net_profit_raw.csv", dtype={"Stkcd": str})
    profit["Stkcd"] = profit["Stkcd"].astype(str).str.zfill(6)
    profit["Accper"] = pd.to_datetime(profit["Accper"], errors="coerce")
    profit["net_profit"] = pd.to_numeric(profit["B002000000"], errors="coerce")
    profit = profit.dropna(subset=["Accper", "net_profit"])
    profit["year"] = profit["Accper"].dt.year
    profit["month"] = profit["Accper"].dt.month
    annual = profit[profit["month"] == 12].sort_values("Accper").drop_duplicates(["Stkcd", "year"], keep="last")
    annual["is_loss"] = (annual["net_profit"] < 0).astype(int)

    # Load panel for total_assets to compute ROA
    panel = pd.read_csv(STAGE1_DIR / "panel_company_year.csv", dtype={"Stkcd": str})
    panel["Stkcd"] = panel["Stkcd"].astype(str).str.zfill(6)
    annual = annual.merge(
        panel[["Stkcd", "year", "total_assets"]].drop_duplicates(),
        on=["Stkcd", "year"], how="left"
    )
    eps = 1e-8
    annual["roa"] = np.where(annual["total_assets"].abs() > eps, annual["net_profit"] / annual["total_assets"], np.nan)

    # Create lookup: (stkcd, year) -> {is_loss, roa, net_profit}
    annual_lookup = annual.set_index(["Stkcd", "year"])[["is_loss", "roa", "net_profit"]].to_dict("index")

    all_rows = []

    for year in range(2010, 2024):
        ydir = GRAPH_DIR / "snapshots" / str(year)
        year_features: Dict[str, Dict[str, float]] = {}

        for et in EDGE_TYPES:
            ep = ydir / f"edges_{et}.csv"
            if not ep.exists():
                continue
            edges = pd.read_csv(ep)
            if len(edges) == 0:
                continue

            # Build directed adjacency
            in_nbrs: Dict[str, Set[str]] = {}   # node -> set of sources (who points to me)
            out_nbrs: Dict[str, Set[str]] = {}  # node -> set of targets (who I point to)
            for _, row in edges.iterrows():
                src_code = id_to_code.get(int(row["src_id"]))
                dst_code = id_to_code.get(int(row["dst_id"]))
                if src_code and dst_code:
                    out_nbrs.setdefault(src_code, set()).add(dst_code)
                    in_nbrs.setdefault(dst_code, set()).add(src_code)

            all_nodes = set(in_nbrs.keys()) | set(out_nbrs.keys())
            for node in all_nodes:
                if node not in year_features:
                    year_features[node] = {}

                # Combined undirected neighbors
                all_nbrs = list(in_nbrs.get(node, set()) | out_nbrs.get(node, set()))

                # === Current year contagion (year t) ===
                prefix_t = f"ctg_{et}"
                feats_t = compute_nbr_stats(all_nbrs, annual_lookup, year, prefix_t)
                year_features[node].update(feats_t)

                # === Lagged contagion (year t-1) — the key improvement ===
                prefix_lag = f"ctg_{et}_lag"
                feats_lag = compute_nbr_stats(all_nbrs, annual_lookup, year - 1, prefix_lag)
                year_features[node].update(feats_lag)

                # === Directional features for directed edges ===
                if et in DIRECTED_TYPES:
                    in_list = list(in_nbrs.get(node, set()))
                    out_list = list(out_nbrs.get(node, set()))
                    if in_list:
                        feats_in = compute_nbr_stats(in_list, annual_lookup, year, f"ctg_{et}_in")
                        year_features[node].update(feats_in)
                        feats_in_lag = compute_nbr_stats(in_list, annual_lookup, year - 1, f"ctg_{et}_in_lag")
                        year_features[node].update(feats_in_lag)
                    if out_list:
                        feats_out = compute_nbr_stats(out_list, annual_lookup, year, f"ctg_{et}_out")
                        year_features[node].update(feats_out)
                        feats_out_lag = compute_nbr_stats(out_list, annual_lookup, year - 1, f"ctg_{et}_out_lag")
                        year_features[node].update(feats_out_lag)

        # Aggregated contagion across all edge types
        for node, feats in year_features.items():
            # Current year aggregates
            total_nbr = sum(v for k, v in feats.items() if k.endswith("_nbr_cnt") and "_lag" not in k and "_in_" not in k and "_out_" not in k)
            total_loss = sum(v for k, v in feats.items() if k.endswith("_loss_cnt") and "_lag" not in k and "_in_" not in k and "_out_" not in k)
            feats["ctg_all_nbr_cnt"] = total_nbr
            feats["ctg_all_loss_cnt"] = total_loss
            feats["ctg_all_loss_ratio"] = total_loss / total_nbr if total_nbr else 0

            # Lagged aggregates
            total_nbr_lag = sum(v for k, v in feats.items() if k.endswith("_nbr_cnt") and "_lag" in k and "_in_" not in k and "_out_" not in k)
            total_loss_lag = sum(v for k, v in feats.items() if k.endswith("_loss_cnt") and "_lag" in k and "_in_" not in k and "_out_" not in k)
            feats["ctg_all_lag_nbr_cnt"] = total_nbr_lag
            feats["ctg_all_lag_loss_cnt"] = total_loss_lag
            feats["ctg_all_lag_loss_ratio"] = total_loss_lag / total_nbr_lag if total_nbr_lag else 0

            # Max distress exposure across edge types
            loss_ratios = [v for k, v in feats.items() if k.endswith("_loss_ratio") and "_lag" not in k and "_all_" not in k and "_in_" not in k and "_out_" not in k]
            feats["ctg_max_loss_ratio"] = max(loss_ratios) if loss_ratios else 0
            lag_loss_ratios = [v for k, v in feats.items() if "_lag_loss_ratio" in k and "_all_" not in k and "_in_" not in k and "_out_" not in k]
            feats["ctg_max_lag_loss_ratio"] = max(lag_loss_ratios) if lag_loss_ratios else 0

            # Any high-distress neighbor flag (loss_ratio > 50%)
            feats["ctg_any_high_distress"] = int(any(v > 0.5 for k, v in feats.items() if k.endswith("_loss_ratio") and "_all_" not in k))

            row = {"Stkcd": node, "year": year}
            row.update(feats)
            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Contagion features v2: {len(result)} rows, {result['Stkcd'].nunique()} companies")
    ctg_cols = [c for c in result.columns if c.startswith("ctg_")]
    print(f"Feature columns ({len(ctg_cols)}): {ctg_cols}")
    print(f"\nSample stats for 2022:")
    r22 = result[result["year"] == 2022]
    for c in sorted(c for c in r22.columns if "loss_ratio" in c):
        print(f"  {c}: mean={r22[c].mean():.3f}, std={r22[c].std():.3f}, max={r22[c].max():.3f}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
