#!/usr/bin/env python3
"""
Build WEIGHTED contagion features for each company-year (v3).

Key insight: L1 features work because they encode financial EXPOSURE (guarantee amounts).
Other edge types also have weights (correlation strength, link counts, equity weights).
Use these weights to build exposure-weighted contagion metrics.

For each edge type:
  - weighted_loss_exposure: sum(weight * is_loss) for all neighbors
  - weighted_roa_exposure: sum(weight * roa) for all neighbors
  - max_weight_to_distressed: max weight to any distressed neighbor
  - total_weight: sum of all edge weights (total exposure)
  - lag versions of above (t-1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_ROOT / "processed" / "stage1"
GRAPH_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"
OUT_PATH = STAGE1_DIR / "distress_task" / "weighted_contagion_features.csv"

EDGE_TYPES = [
    "guarantee",
    "shared_nonlisted_guarantee",
    "equity_assoc",
    "equity_change",
    "co_controller",
    "market_corr",
    "industry",
]

SHORT = {
    "guarantee": "guar",
    "shared_nonlisted_guarantee": "sng",
    "equity_assoc": "eqa",
    "equity_change": "eqc",
    "co_controller": "coctl",
    "market_corr": "mktcorr",
    "industry": "ind",
}


def main() -> None:
    nm = pd.read_csv(GRAPH_DIR / "node_mapping.csv", dtype={"Stkcd": str})
    nm["Stkcd"] = nm["Stkcd"].astype(str).str.zfill(6)
    id_to_code = dict(zip(nm["node_id"], nm["Stkcd"]))

    # Load annual financial data
    profit = pd.read_csv(STAGE1_DIR / "net_profit_raw.csv", dtype={"Stkcd": str})
    profit["Stkcd"] = profit["Stkcd"].astype(str).str.zfill(6)
    profit["Accper"] = pd.to_datetime(profit["Accper"], errors="coerce")
    profit["net_profit"] = pd.to_numeric(profit["B002000000"], errors="coerce")
    profit = profit.dropna(subset=["Accper", "net_profit"])
    profit["year"] = profit["Accper"].dt.year
    profit["month"] = profit["Accper"].dt.month
    annual = profit[profit["month"] == 12].sort_values("Accper").drop_duplicates(["Stkcd", "year"], keep="last")
    annual["is_loss"] = (annual["net_profit"] < 0).astype(int)

    panel = pd.read_csv(STAGE1_DIR / "panel_company_year.csv", dtype={"Stkcd": str})
    panel["Stkcd"] = panel["Stkcd"].astype(str).str.zfill(6)
    annual = annual.merge(panel[["Stkcd", "year", "total_assets"]].drop_duplicates(), on=["Stkcd", "year"], how="left")
    eps = 1e-8
    annual["roa"] = np.where(annual["total_assets"].abs() > eps, annual["net_profit"] / annual["total_assets"], np.nan)
    annual_lookup = annual.set_index(["Stkcd", "year"])[["is_loss", "roa"]].to_dict("index")

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

            sn = SHORT[et]

            # Build weighted adjacency: node -> [(neighbor, weight)]
            nbr_weights: Dict[str, List[Tuple[str, float]]] = {}
            for _, row in edges.iterrows():
                src = id_to_code.get(int(row["src_id"]))
                dst = id_to_code.get(int(row["dst_id"]))
                w = float(row.get("weight", 1.0))
                if src and dst:
                    nbr_weights.setdefault(src, []).append((dst, w))
                    nbr_weights.setdefault(dst, []).append((src, w))

            for node, nw_list in nbr_weights.items():
                if node not in year_features:
                    year_features[node] = {}

                # Deduplicate neighbors, keep max weight
                nbr_max_w: Dict[str, float] = {}
                for nb, w in nw_list:
                    nbr_max_w[nb] = max(nbr_max_w.get(nb, 0), w)

                for yr_offset, suffix in [(0, ""), (-1, "_lag")]:
                    lookup_year = year + yr_offset
                    total_w = 0.0
                    weighted_loss = 0.0
                    weighted_roa = 0.0
                    max_w_to_distressed = 0.0
                    nbr_cnt = 0
                    loss_cnt = 0
                    roas = []

                    for nb, w in nbr_max_w.items():
                        nbr_cnt += 1
                        total_w += w
                        info = annual_lookup.get((nb, lookup_year))
                        if info:
                            is_loss = info["is_loss"]
                            roa = info["roa"]
                            loss_cnt += is_loss
                            weighted_loss += w * is_loss
                            if not np.isnan(roa):
                                weighted_roa += w * roa
                                roas.append(roa)
                            if is_loss and w > max_w_to_distressed:
                                max_w_to_distressed = w

                    p = f"wc_{sn}{suffix}"
                    year_features[node][f"{p}_nbr_cnt"] = nbr_cnt
                    year_features[node][f"{p}_loss_cnt"] = loss_cnt
                    year_features[node][f"{p}_loss_ratio"] = loss_cnt / nbr_cnt if nbr_cnt else 0
                    year_features[node][f"{p}_total_w"] = total_w
                    year_features[node][f"{p}_wt_loss"] = weighted_loss
                    year_features[node][f"{p}_wt_loss_norm"] = weighted_loss / total_w if total_w > 0 else 0
                    year_features[node][f"{p}_wt_roa"] = weighted_roa / total_w if total_w > 0 else 0
                    year_features[node][f"{p}_max_w_distress"] = max_w_to_distressed
                    year_features[node][f"{p}_mean_roa"] = float(np.mean(roas)) if roas else 0

        # Aggregates
        for node, feats in year_features.items():
            for suffix in ["", "_lag"]:
                total_wt_loss = sum(v for k, v in feats.items() if k.endswith(f"{suffix}_wt_loss") and not k.endswith("_norm"))
                total_w = sum(v for k, v in feats.items() if k.endswith(f"{suffix}_total_w"))
                feats[f"wc_all{suffix}_wt_loss"] = total_wt_loss
                feats[f"wc_all{suffix}_total_w"] = total_w
                feats[f"wc_all{suffix}_wt_loss_norm"] = total_wt_loss / total_w if total_w > 0 else 0
                # Max weighted distress exposure across types
                max_vals = [v for k, v in feats.items() if k.endswith(f"{suffix}_max_w_distress")]
                feats[f"wc_all{suffix}_max_w_distress"] = max(max_vals) if max_vals else 0

            row = {"Stkcd": node, "year": year}
            row.update(feats)
            all_rows.append(row)

    result = pd.DataFrame(all_rows).fillna(0)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    wc_cols = [c for c in result.columns if c.startswith("wc_")]
    print(f"Weighted contagion features: {len(result)} rows, {result['Stkcd'].nunique()} companies, {len(wc_cols)} features")

    # Show key feature stats for 2022
    r22 = result[result["year"] == 2022]
    print(f"\nKey features for 2022 (n={len(r22)}):")
    for c in sorted(c for c in wc_cols if "wt_loss_norm" in c or "max_w_distress" in c):
        nz = (r22[c] > 0).sum()
        print(f"  {c}: mean={r22[c].mean():.4f}, std={r22[c].std():.4f}, max={r22[c].max():.4f}, nonzero={nz}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
