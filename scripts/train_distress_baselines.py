#!/usr/bin/env python3
"""
Train L0-L1 baselines for financial distress (loss next year) prediction.

L0: Pure financial features (no graph info)
L1: Financial features + handcrafted graph statistics

Models: LogisticRegression, RandomForest, XGBoost (if available)

Inputs:
  processed/stage1/distress_task/train.csv
  processed/stage1/distress_task/val.csv
  processed/stage1/distress_task/test.csv

Outputs:
  processed/stage1/distress_task/experiments/baseline_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISTRESS_DIR = PROJECT_ROOT / "processed" / "stage1" / "distress_task"
EDGE_FILE = PROJECT_ROOT / "processed" / "stage1" / "edges_guarantee_listed_to_listed.csv"
CONTAGION_FILE = DISTRESS_DIR / "contagion_features.csv"
STRUCTURE_FILE = DISTRESS_DIR / "structure_features.csv"
OUT_DIR = DISTRESS_DIR / "experiments"

LABEL_COL = "label_loss_next_year"

# Edge types for contagion features
CONTAGION_EDGE_TYPES = [
    "guarantee",
    "shared_nonlisted_guarantee",
    "equity_assoc",
    "equity_change",
    "co_controller",
    "market_corr",
    "industry",
]
CONTAGION_SUFFIXES = ["_nbr_cnt", "_loss_cnt", "_loss_ratio", "_mean_roa", "_min_roa"]

SHORT_NAMES = {
    "guarantee": "guar",
    "shared_nonlisted_guarantee": "sng",
    "equity_assoc": "eqa",
    "equity_change": "eqc",
    "co_controller": "coctl",
    "market_corr": "mktcorr",
    "industry": "ind",
}

# L0 features: pure financial, no graph info
L0_NUMERIC = [
    "total_assets", "total_liabilities", "total_equity",
    "revenue_total", "revenue_main", "cashflow_operating",
    "asset_liability_ratio", "Regcap",
    "net_profit", "roa", "equity_ratio", "cashflow_to_assets",
    "is_loss_current_year",
    "rating_event_cnt", "rating_agency_nunique",
]
L0_CATEGORICAL = ["Markettype", "Nnindcd", "Statco", "rating_latest_longterm", "rating_latest_prospect"]

# L1 features: L0 + handcrafted graph statistics
L1_GRAPH_FEATURES = [
    "guar_event_cnt", "guar_amt_sum", "guar_amt_mean",
    "guar_listed_target_ratio",
    # Extra: computed from guarantee edge file
    "g_out_event_cnt", "g_out_nbr_cnt", "g_out_amt_sum",
    "g_in_event_cnt", "g_in_nbr_cnt", "g_in_amt_sum",
    "g_cum_out_event", "g_cum_in_event",
]


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Stkcd": str})
    df["Stkcd"] = df["Stkcd"].astype(str).str.zfill(6)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    return df


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> Tuple[float, float]:
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    idx = np.argsort(-y_prob)[:k]
    hit = y_true[idx].sum()
    return safe_div(hit, k), safe_div(hit, y_true.sum())


def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }
    for frac in (0.01, 0.05, 0.10):
        p, r = precision_recall_at_k(y_true, y_prob, frac)
        out[f"precision@{frac:.2f}"] = p
        out[f"recall@{frac:.2f}"] = r
    return out


def build_graph_year_features(edge_path: Path) -> pd.DataFrame:
    """Build yearly graph features from guarantee edges."""
    if not edge_path.exists():
        return pd.DataFrame()
    edge = pd.read_csv(edge_path, dtype={"src_stkcd": str, "dst_stkcd": str})
    edge["src_stkcd"] = edge["src_stkcd"].astype(str).str.zfill(6)
    edge["dst_stkcd"] = edge["dst_stkcd"].astype(str).str.zfill(6)
    edge["year"] = pd.to_numeric(edge["year"], errors="coerce").astype("Int64")
    edge["ActualGuaranteeAmount"] = pd.to_numeric(edge["ActualGuaranteeAmount"], errors="coerce").fillna(0.0)
    edge = edge.dropna(subset=["year"]).copy()
    edge["year"] = edge["year"].astype(int)

    out_y = (
        edge.groupby(["src_stkcd", "year"], as_index=False)
        .agg(g_out_event_cnt=("dst_stkcd", "size"), g_out_nbr_cnt=("dst_stkcd", "nunique"), g_out_amt_sum=("ActualGuaranteeAmount", "sum"))
        .rename(columns={"src_stkcd": "Stkcd"})
    )
    in_y = (
        edge.groupby(["dst_stkcd", "year"], as_index=False)
        .agg(g_in_event_cnt=("src_stkcd", "size"), g_in_nbr_cnt=("src_stkcd", "nunique"), g_in_amt_sum=("ActualGuaranteeAmount", "sum"))
        .rename(columns={"dst_stkcd": "Stkcd"})
    )
    gy = out_y.merge(in_y, on=["Stkcd", "year"], how="outer")

    # Cumulative
    gy = gy.sort_values(["Stkcd", "year"])
    for c in ["g_out_event_cnt", "g_in_event_cnt"]:
        if c in gy.columns:
            gy[f"g_cum_{c.replace('g_', '').replace('_cnt', '')}"] = gy.groupby("Stkcd")[c].cumsum()
    return gy


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[ColumnTransformer, List[str]]:
    """Build preprocessing pipeline."""
    actual_cat = [c for c in categorical_cols]
    transformers = [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), numeric_cols),
    ]
    if actual_cat:
        transformers.append(
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), actual_cat)
        )
    return ColumnTransformer(transformers, remainder="drop")


def run_experiment(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   numeric_cols: List[str], categorical_cols: List[str], level: str) -> Dict:
    """Run all models for a given feature set level."""
    # Filter to columns that exist
    numeric_cols = [c for c in numeric_cols if c in train.columns]
    categorical_cols = [c for c in categorical_cols if c in train.columns]

    ct = build_pipeline(numeric_cols, categorical_cols)

    X_train = ct.fit_transform(train)
    X_val = ct.transform(val)
    X_test = ct.transform(test)
    y_train = train[LABEL_COL].values
    y_val = val[LABEL_COL].values
    y_test = test[LABEL_COL].values

    results = {}
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0),
        "rf": RandomForestClassifier(n_estimators=500, class_weight="balanced", max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "gbdt": GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        results[f"{level}_{name}"] = {
            "val": eval_binary(y_val, val_prob),
            "test": eval_binary(y_test, test_prob),
        }
        print(f"  {level}_{name}: val AUC={results[f'{level}_{name}']['val']['auc']:.4f} AP={results[f'{level}_{name}']['val']['ap']:.4f} | "
              f"test AUC={results[f'{level}_{name}']['test']['auc']:.4f} AP={results[f'{level}_{name}']['test']['ap']:.4f}")

    return results


def main() -> None:
    print("Loading data...")
    train = load_split(DISTRESS_DIR / "train.csv")
    val = load_split(DISTRESS_DIR / "val.csv")
    test = load_split(DISTRESS_DIR / "test.csv")
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Build graph features for L1
    print("Building graph features...")
    gf = build_graph_year_features(EDGE_FILE)
    if not gf.empty:
        for df_name, df in [("train", train), ("val", val), ("test", test)]:
            merged = df.merge(gf, on=["Stkcd", "year"], how="left")
            for c in gf.columns:
                if c not in ["Stkcd", "year"]:
                    merged[c] = merged[c].fillna(0)
            if df_name == "train":
                train = merged
            elif df_name == "val":
                val = merged
            else:
                test = merged

    # Load contagion features (v2 with ctg_ prefix)
    print("Loading contagion features v2...")
    if CONTAGION_FILE.exists():
        ctg = pd.read_csv(CONTAGION_FILE, dtype={"Stkcd": str})
        ctg["Stkcd"] = ctg["Stkcd"].astype(str).str.zfill(6)
        ctg["year"] = pd.to_numeric(ctg["year"], errors="coerce").astype(int)
        ctg_cols = [c for c in ctg.columns if c.startswith("ctg_")]
        for df_name, df in [("train", train), ("val", val), ("test", test)]:
            merged = df.merge(ctg[["Stkcd", "year"] + ctg_cols], on=["Stkcd", "year"], how="left")
            for c in ctg_cols:
                merged[c] = merged[c].fillna(0)
            if df_name == "train":
                train = merged
            elif df_name == "val":
                val = merged
            else:
                test = merged
        print(f"  Contagion features merged: {len(ctg_cols)} columns")
    else:
        ctg_cols = []
        print("  WARNING: contagion_features.csv not found, skipping L2")

    all_results = {}

    # L0: Pure financial features
    print("\n=== L0: Pure financial features ===")
    l0_results = run_experiment(train, val, test, L0_NUMERIC, L0_CATEGORICAL, "L0")
    all_results.update(l0_results)

    # L1: Financial + graph statistics (handcrafted guarantee features)
    print("\n=== L1: Financial + guarantee graph statistics ===")
    l1_numeric = L0_NUMERIC + [c for c in L1_GRAPH_FEATURES if c in train.columns]
    l1_results = run_experiment(train, val, test, l1_numeric, L0_CATEGORICAL, "L1")
    all_results.update(l1_results)

    if ctg_cols:
        # === Curated feature sets to avoid 165-feature overfitting ===

        # L2a: Compact — only lag loss_ratio + lag mean_roa per edge type (most informative, least leaky)
        lag_compact = []
        for et in CONTAGION_EDGE_TYPES:
            for s in ["_lag_loss_ratio", "_lag_mean_roa"]:
                col = f"ctg_{et}{s}"
                if col in ctg_cols:
                    lag_compact.append(col)
        # Add aggregate lag features
        for c in ["ctg_all_lag_loss_ratio", "ctg_max_lag_loss_ratio", "ctg_any_high_distress"]:
            if c in ctg_cols:
                lag_compact.append(c)
        print(f"\n=== L2a: Financial + lag contagion (compact, {len(lag_compact)} features) ===")
        l2a_results = run_experiment(train, val, test, L0_NUMERIC + lag_compact, L0_CATEGORICAL, "L2a_lag_compact")
        all_results.update(l2a_results)

        # L2b: Current-year loss_ratio + mean_roa per edge type (undirected)
        curr_compact = []
        for et in CONTAGION_EDGE_TYPES:
            for s in ["_loss_ratio", "_mean_roa"]:
                col = f"ctg_{et}{s}"
                if col in ctg_cols:
                    curr_compact.append(col)
        for c in ["ctg_all_loss_ratio", "ctg_max_loss_ratio", "ctg_any_high_distress"]:
            if c in ctg_cols:
                curr_compact.append(c)
        print(f"\n=== L2b: Financial + current contagion (compact, {len(curr_compact)} features) ===")
        l2b_results = run_experiment(train, val, test, L0_NUMERIC + curr_compact, L0_CATEGORICAL, "L2b_curr_compact")
        all_results.update(l2b_results)

        # L2c: Both lag + current compact
        both_compact = list(set(lag_compact + curr_compact))
        print(f"\n=== L2c: Financial + lag+current contagion (compact, {len(both_compact)} features) ===")
        l2c_results = run_experiment(train, val, test, L0_NUMERIC + both_compact, L0_CATEGORICAL, "L2c_both_compact")
        all_results.update(l2c_results)

        # L2d: L1 guarantee stats + lag contagion (combine both graph info sources)
        l1_cols = [c for c in L1_GRAPH_FEATURES if c in train.columns]
        print(f"\n=== L2d: L1 + lag contagion ===")
        l2d_results = run_experiment(train, val, test, L0_NUMERIC + l1_cols + lag_compact, L0_CATEGORICAL, "L2d_L1_plus_lag")
        all_results.update(l2d_results)

        # L2e: L1 guarantee stats + current+lag contagion
        print(f"\n=== L2e: L1 + both contagion ===")
        l2e_results = run_experiment(train, val, test, L0_NUMERIC + l1_cols + both_compact, L0_CATEGORICAL, "L2e_L1_plus_both")
        all_results.update(l2e_results)

        # L2f: All 165 contagion features (reference, likely overfit)
        print(f"\n=== L2f: Financial + ALL {len(ctg_cols)} contagion features ===")
        l2f_results = run_experiment(train, val, test, L0_NUMERIC + ctg_cols, L0_CATEGORICAL, "L2f_all165")
        all_results.update(l2f_results)

        # Per edge type with lag features
        for et in CONTAGION_EDGE_TYPES:
            et_lag = [c for c in ctg_cols if c.startswith(f"ctg_{et}_lag") and "_in_" not in c and "_out_" not in c]
            et_curr = [c for c in ctg_cols if c.startswith(f"ctg_{et}_") and "_lag" not in c and "_in_" not in c and "_out_" not in c]
            et_all = list(set(et_lag + et_curr))
            if et_all:
                print(f"\n=== L2_{et}: Financial + {et} contagion ({len(et_all)} features) ===")
                l2_et_results = run_experiment(train, val, test, L0_NUMERIC + et_all, L0_CATEGORICAL, f"L2_{et}")
                all_results.update(l2_et_results)

    # === L3: Structure features (graph topology) ===
    print("\nLoading structure features...")
    if STRUCTURE_FILE.exists():
        stf = pd.read_csv(STRUCTURE_FILE, dtype={"Stkcd": str})
        stf["Stkcd"] = stf["Stkcd"].astype(str).str.zfill(6)
        stf["year"] = pd.to_numeric(stf["year"], errors="coerce").astype(int)
        str_cols = [c for c in stf.columns if c.startswith("str_")]
        for df_name, df in [("train", train), ("val", val), ("test", test)]:
            merged = df.merge(stf[["Stkcd", "year"] + str_cols], on=["Stkcd", "year"], how="left")
            for c in str_cols:
                merged[c] = merged[c].fillna(0)
            if df_name == "train":
                train = merged
            elif df_name == "val":
                val = merged
            else:
                test = merged
        print(f"  Structure features merged: {len(str_cols)} columns")

        # L3a: L0 + all structure features
        print(f"\n=== L3a: L0 + structure ({len(str_cols)} features) ===")
        l3a_results = run_experiment(train, val, test, L0_NUMERIC + str_cols, L0_CATEGORICAL, "L3a_struct")
        all_results.update(l3a_results)

        # L3b: L1 + structure
        l1_cols = [c for c in L1_GRAPH_FEATURES if c in train.columns]
        print(f"\n=== L3b: L1 + structure ===")
        l3b_results = run_experiment(train, val, test, L0_NUMERIC + l1_cols + str_cols, L0_CATEGORICAL, "L3b_L1_struct")
        all_results.update(l3b_results)

        # L3c: L1 + structure + best contagion (lag compact)
        if ctg_cols:
            print(f"\n=== L3c: L1 + structure + lag contagion (full model) ===")
            l3c_results = run_experiment(train, val, test, L0_NUMERIC + l1_cols + str_cols + lag_compact, L0_CATEGORICAL, "L3c_full")
            all_results.update(l3c_results)

        # L3 per edge type structure only
        for et in CONTAGION_EDGE_TYPES:
            sn = SHORT_NAMES.get(et, et[:4])
            et_str = [c for c in str_cols if c.startswith(f"str_{sn}_")]
            if et_str:
                print(f"\n=== L3_{et}: L0 + {et} structure ===")
                l3_et_results = run_experiment(train, val, test, L0_NUMERIC + et_str, L0_CATEGORICAL, f"L3_{et}")
                all_results.update(l3_et_results)
    else:
        print("  WARNING: structure_features.csv not found, skipping L3")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "baseline_metrics.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Model':<35} {'Val AUC':>10} {'Val AP':>10} {'Test AUC':>10} {'Test AP':>10}")
    print("-" * 77)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["test"]["auc"]):
        print(f"{name:<35} {r['val']['auc']:>10.4f} {r['val']['ap']:>10.4f} {r['test']['auc']:>10.4f} {r['test']['ap']:>10.4f}")

    print(f"\nSaved to {OUT_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
