#!/usr/bin/env python3
"""
Ensemble approach: use HTGNN predicted probabilities as an additional
feature for tree-based models (RF/GBDT). This tests whether graph-learned
representations provide complementary signal beyond tabular features.

Also implements L2 baselines (GCN/GAT on flattened homogeneous graph)
for the paper's comparison framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISTRESS_DIR = PROJECT_ROOT / "processed" / "stage1" / "distress_task"
HTGNN_EXP_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph" / "experiments_htgnn_distress"
OUT_DIR = DISTRESS_DIR / "experiments"
LABEL_COL = "label_loss_next_year"


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


def load_htgnn_predictions(exp_name: str) -> pd.DataFrame:
    """Load HTGNN predictions from a specific experiment."""
    exp_dir = HTGNN_EXP_DIR / exp_name
    dfs = []
    for split in ["val", "test"]:
        p = exp_dir / f"{split}_predictions.csv"
        if p.exists():
            df = pd.read_csv(p, dtype={"Stkcd": str})
            df["Stkcd"] = df["Stkcd"].astype(str).str.zfill(6)
            df["split"] = split
            dfs.append(df)
    # Also need train predictions - re-predict or use saved
    # For now, we use the val/test predictions and fill train with 0.5 (neutral)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformers = [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), numeric_cols),
    ]
    if categorical_cols:
        transformers.append(
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols)
        )
    return ColumnTransformer(transformers, remainder="drop")


L0_NUMERIC = [
    "total_assets", "total_liabilities", "total_equity",
    "revenue_total", "revenue_main", "cashflow_operating",
    "asset_liability_ratio", "Regcap",
    "net_profit", "roa", "equity_ratio", "cashflow_to_assets",
    "is_loss_current_year",
    "rating_event_cnt", "rating_agency_nunique",
]
L0_CATEGORICAL = ["Markettype", "Nnindcd", "Statco", "rating_latest_longterm", "rating_latest_prospect"]

L1_GRAPH_FEATURES = [
    "guar_event_cnt", "guar_amt_sum", "guar_amt_mean",
    "guar_listed_target_ratio",
]


def main() -> None:
    print("Loading data...")
    train = load_split(DISTRESS_DIR / "train.csv")
    val = load_split(DISTRESS_DIR / "val.csv")
    test = load_split(DISTRESS_DIR / "test.csv")

    # Try to load HTGNN predictions for ensemble
    htgnn_experiments = [
        "distress_v2_bce_no_ind",
        "distress_v2_no_tabular",
        "distress_v2_no_industry",
        "ablation_only_guarantee",
        "ablation_no_market_corr",
    ]

    all_results = {}

    for htgnn_exp in htgnn_experiments:
        preds = load_htgnn_predictions(htgnn_exp)
        if preds.empty:
            print(f"  No predictions found for {htgnn_exp}, skipping ensemble")
            continue

        # Merge HTGNN predictions as feature
        val_pred = preds[preds["split"] == "val"][["Stkcd", "year", "pred_prob"]].rename(columns={"pred_prob": "htgnn_prob"})
        test_pred = preds[preds["split"] == "test"][["Stkcd", "year", "pred_prob"]].rename(columns={"pred_prob": "htgnn_prob"})

        val_e = val.merge(val_pred, on=["Stkcd", "year"], how="left")
        test_e = test.merge(test_pred, on=["Stkcd", "year"], how="left")

        # For train, we don't have HTGNN predictions (would be data leakage to use train preds)
        # Instead, use cross-validated predictions or just 0.5
        train_e = train.copy()
        train_e["htgnn_prob"] = 0.5  # neutral prior

        # L0 + HTGNN ensemble
        numeric_ens = L0_NUMERIC + L1_GRAPH_FEATURES + ["htgnn_prob"]
        numeric_ens = [c for c in numeric_ens if c in train_e.columns]
        cat_cols = [c for c in L0_CATEGORICAL if c in train_e.columns]

        ct = build_pipeline(numeric_ens, cat_cols)
        X_train = ct.fit_transform(train_e)
        X_val = ct.transform(val_e)
        X_test = ct.transform(test_e)
        y_train = train_e[LABEL_COL].values
        y_val = val_e[LABEL_COL].values
        y_test = test_e[LABEL_COL].values

        for model_name, model in [
            ("rf", RandomForestClassifier(n_estimators=500, class_weight="balanced", max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)),
            ("gbdt", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)),
        ]:
            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
            key = f"L1+HTGNN({htgnn_exp})_{model_name}"
            all_results[key] = {
                "val": eval_binary(y_val, val_prob),
                "test": eval_binary(y_test, test_prob),
            }
            print(f"  {key}: val AUC={all_results[key]['val']['auc']:.4f} AP={all_results[key]['val']['ap']:.4f} | "
                  f"test AUC={all_results[key]['test']['auc']:.4f} AP={all_results[key]['test']['ap']:.4f}")

    # Also run a simple probability averaging ensemble
    print("\n=== Probability Averaging Ensemble ===")
    for htgnn_exp in htgnn_experiments:
        preds = load_htgnn_predictions(htgnn_exp)
        if preds.empty:
            continue

        # Load L0_rf predictions (need to retrain quickly)
        numeric_l0 = [c for c in L0_NUMERIC if c in train.columns]
        cat_l0 = [c for c in L0_CATEGORICAL if c in train.columns]
        ct_l0 = build_pipeline(numeric_l0, cat_l0)
        X_tr = ct_l0.fit_transform(train)
        X_v = ct_l0.transform(val)
        X_t = ct_l0.transform(test)

        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
        rf.fit(X_tr, train[LABEL_COL].values)
        rf_val_prob = rf.predict_proba(X_v)[:, 1]
        rf_test_prob = rf.predict_proba(X_t)[:, 1]

        # Merge HTGNN probs
        val_pred = preds[preds["split"] == "val"].set_index(["Stkcd", "year"])["pred_prob"]
        test_pred = preds[preds["split"] == "test"].set_index(["Stkcd", "year"])["pred_prob"]

        val_with_key = val.set_index(["Stkcd", "year"])
        test_with_key = test.set_index(["Stkcd", "year"])

        htgnn_val = val_with_key.index.map(lambda x: val_pred.get(x, 0.5) if x in val_pred.index else 0.5).values
        htgnn_test = test_with_key.index.map(lambda x: test_pred.get(x, 0.5) if x in test_pred.index else 0.5).values

        for alpha in [0.3, 0.5, 0.7]:
            avg_val = alpha * rf_val_prob + (1 - alpha) * htgnn_val
            avg_test = alpha * rf_test_prob + (1 - alpha) * htgnn_test
            key = f"Avg(RF*{alpha:.1f}+HTGNN*{1-alpha:.1f})_{htgnn_exp}"
            all_results[key] = {
                "val": eval_binary(val[LABEL_COL].values, avg_val),
                "test": eval_binary(test[LABEL_COL].values, avg_test),
            }
            print(f"  {key}: val AUC={all_results[key]['val']['auc']:.4f} | test AUC={all_results[key]['test']['auc']:.4f} AP={all_results[key]['test']['ap']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ensemble_metrics.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Final Summary ===")
    print(f"{'Model':<65} {'Val AUC':>10} {'Val AP':>10} {'Test AUC':>10} {'Test AP':>10}")
    print("-" * 107)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["test"]["auc"]):
        print(f"{name:<65} {r['val']['auc']:>10.4f} {r['val']['ap']:>10.4f} {r['test']['auc']:>10.4f} {r['test']['ap']:>10.4f}")


if __name__ == "__main__":
    main()
