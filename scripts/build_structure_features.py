#!/usr/bin/env python3
"""
Build graph STRUCTURE features for each company-year.

Key insight from experiments: L1 (guarantee structure) beats all contagion
features. The signal is in HOW CONNECTED a company is, not in whether
its neighbors are in distress.

For each edge type, compute:
  - in_degree: number of incoming edges
  - out_degree: number of outgoing edges
  - total_degree: in + out (undirected)
  - is_connected: binary flag if node has any edges of this type

Aggregate features:
  - total_edge_types: how many different relationship types the company has
  - total_connections: sum of all degrees across edge types
  - max_degree: maximum degree across edge types
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_ROOT / "processed" / "stage1"
GRAPH_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"
OUT_PATH = STAGE1_DIR / "distress_task" / "structure_features.csv"

EDGE_TYPES = [
    "guarantee",
    "shared_nonlisted_guarantee",
    "equity_assoc",
    "equity_change",
    "co_controller",
    "market_corr",
    "industry",
]

# Short names for feature columns
SHORT_NAMES = {
    "guarantee": "guar",
    "shared_nonlisted_guarantee": "sng",
    "equity_assoc": "eqa",
    "equity_change": "eqc",
    "co_controller": "coctl",
    "market_corr": "mktcorr",
    "industry": "ind",
}


def main() -> None:
    # Load node mapping
    nm = pd.read_csv(GRAPH_DIR / "node_mapping.csv", dtype={"Stkcd": str})
    nm["Stkcd"] = nm["Stkcd"].astype(str).str.zfill(6)
    id_to_code = dict(zip(nm["node_id"], nm["Stkcd"]))

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

            sn = SHORT_NAMES[et]

            # Count in/out degrees
            in_deg: Dict[str, int] = {}
            out_deg: Dict[str, int] = {}
            for _, row in edges.iterrows():
                src_code = id_to_code.get(int(row["src_id"]))
                dst_code = id_to_code.get(int(row["dst_id"]))
                if src_code and dst_code:
                    out_deg[src_code] = out_deg.get(src_code, 0) + 1
                    in_deg[dst_code] = in_deg.get(dst_code, 0) + 1

            all_nodes = set(in_deg.keys()) | set(out_deg.keys())
            for node in all_nodes:
                if node not in year_features:
                    year_features[node] = {}
                ind = in_deg.get(node, 0)
                outd = out_deg.get(node, 0)
                year_features[node][f"str_{sn}_in"] = ind
                year_features[node][f"str_{sn}_out"] = outd
                year_features[node][f"str_{sn}_deg"] = ind + outd
                year_features[node][f"str_{sn}_conn"] = 1  # is connected

        # Aggregate features
        for node, feats in year_features.items():
            # Count edge types
            n_types = sum(1 for k, v in feats.items() if k.endswith("_conn") and v > 0)
            feats["str_n_edge_types"] = n_types

            # Total connections
            total_deg = sum(v for k, v in feats.items() if k.endswith("_deg"))
            feats["str_total_deg"] = total_deg

            # Max degree across types
            degs = [v for k, v in feats.items() if k.endswith("_deg")]
            feats["str_max_deg"] = max(degs) if degs else 0

            # Log-transformed degree (diminishing returns)
            feats["str_log_total_deg"] = np.log1p(total_deg)

            row = {"Stkcd": node, "year": year}
            row.update(feats)
            all_rows.append(row)

    result = pd.DataFrame(all_rows).fillna(0)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    str_cols = [c for c in result.columns if c.startswith("str_")]
    print(f"Structure features: {len(result)} rows, {result['Stkcd'].nunique()} companies, {len(str_cols)} features")
    print(f"Features: {str_cols}")
    print(f"\nSample stats for 2022:")
    r22 = result[result["year"] == 2022]
    for c in sorted(str_cols):
        nz = (r22[c] > 0).sum()
        print(f"  {c}: mean={r22[c].mean():.2f}, max={r22[c].max():.0f}, nonzero={nz}/{len(r22)}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
