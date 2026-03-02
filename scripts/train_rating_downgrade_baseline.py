#!/usr/bin/env python3
"""
Train baseline models for rating-downgrade binary prediction.

Inputs:
  processed/stage1/rating_task/train.csv
  processed/stage1/rating_task/val.csv
  processed/stage1/rating_task/test.csv
  processed/stage1/edges_guarantee_listed_to_listed.csv

Outputs:
  processed/stage1/rating_task/experiments/baseline_metrics.json
  processed/stage1/rating_task/experiments/val_predictions.csv
  processed/stage1/rating_task/experiments/test_predictions.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATING_DIR = PROJECT_ROOT / "processed" / "stage1" / "rating_task"
EDGE_FILE = PROJECT_ROOT / "processed" / "stage1" / "edges_guarantee_listed_to_listed.csv"
OUT_DIR = RATING_DIR / "experiments"


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Stkcd": str})
    df["Stkcd"] = df["Stkcd"].astype(str).str.zfill(6)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    return df


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> Tuple[float, float, int]:
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    idx = np.argsort(-y_prob)[:k]
    hit = y_true[idx].sum()
    precision = safe_div(hit, k)
    recall = safe_div(hit, y_true.sum())
    return float(precision), float(recall), int(k)


def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "precision@0.01": None,
        "recall@0.01": None,
        "precision@0.05": None,
        "recall@0.05": None,
        "precision@0.10": None,
        "recall@0.10": None,
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }
    for frac in (0.01, 0.05, 0.10):
        p, r, _ = precision_recall_at_k(y_true, y_prob, frac)
        out[f"precision@{frac:.2f}"] = p
        out[f"recall@{frac:.2f}"] = r
    return out


def build_graph_year_features(edge_path: Path) -> pd.DataFrame:
    edge = pd.read_csv(edge_path, dtype={"src_stkcd": str, "dst_stkcd": str})
    edge["src_stkcd"] = edge["src_stkcd"].astype(str).str.zfill(6)
    edge["dst_stkcd"] = edge["dst_stkcd"].astype(str).str.zfill(6)
    edge["year"] = pd.to_numeric(edge["year"], errors="coerce").astype("Int64")
    edge["ActualGuaranteeAmount"] = pd.to_numeric(edge["ActualGuaranteeAmount"], errors="coerce").fillna(0.0)
    edge = edge.dropna(subset=["year"]).copy()
    edge["year"] = edge["year"].astype(int)

    out_y = (
        edge.groupby(["src_stkcd", "year"], as_index=False)
        .agg(
            g_out_event_cnt=("dst_stkcd", "size"),
            g_out_nbr_cnt=("dst_stkcd", "nunique"),
            g_out_amt_sum=("ActualGuaranteeAmount", "sum"),
        )
        .rename(columns={"src_stkcd": "Stkcd"})
    )
    in_y = (
        edge.groupby(["dst_stkcd", "year"], as_index=False)
        .agg(
            g_in_event_cnt=("src_stkcd", "size"),
            g_in_nbr_cnt=("src_stkcd", "nunique"),
            g_in_amt_sum=("ActualGuaranteeAmount", "sum"),
        )
        .rename(columns={"dst_stkcd": "Stkcd"})
    )

    g = out_y.merge(in_y, on=["Stkcd", "year"], how="outer")
    for c in [
        "g_out_event_cnt",
        "g_out_nbr_cnt",
        "g_out_amt_sum",
        "g_in_event_cnt",
        "g_in_nbr_cnt",
        "g_in_amt_sum",
    ]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    g = g.sort_values(["Stkcd", "year"]).reset_index(drop=True)

    # Cumulative relation features up to current year.
    for c in [
        "g_out_event_cnt",
        "g_out_nbr_cnt",
        "g_out_amt_sum",
        "g_in_event_cnt",
        "g_in_nbr_cnt",
        "g_in_amt_sum",
    ]:
        g[f"{c}_cum"] = g.groupby("Stkcd")[c].cumsum()
    return g


def attach_graph_features(df: pd.DataFrame, graph_df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(graph_df, on=["Stkcd", "year"], how="left")
    graph_cols = [c for c in graph_df.columns if c not in {"Stkcd", "year"}]
    for c in graph_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def make_preprocessor(df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = list(feature_cols)
    preset_cat = {"Markettype", "Nnindcd", "Statco", "rating_latest_prospect", "rating_norm", "rating_latest_longterm"}
    # Force object/string-like columns into categorical pipeline.
    inferred_cat = {c for c in feature_cols if df[c].dtype == "object"}
    cat_cols = sorted([c for c in feature_cols if c in preset_cat or c in inferred_cat])
    num_cols = sorted([c for c in feature_cols if c not in cat_cols])

    # Backward-compatible OneHotEncoder param across sklearn versions.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split(RATING_DIR / "train.csv")
    val = load_split(RATING_DIR / "val.csv")
    test = load_split(RATING_DIR / "test.csv")

    graph_feat = build_graph_year_features(EDGE_FILE)
    train = attach_graph_features(train, graph_feat)
    val = attach_graph_features(val, graph_feat)
    test = attach_graph_features(test, graph_feat)

    y_col = "label_downgrade_next_year"
    drop_cols = {
        "Stkcd",
        "year",
        "label_rating_next_year",
        "label_rating_score_next_year",
        "label_downgrade_next_year",
    }
    feature_cols = [c for c in train.columns if c not in drop_cols]
    pre, num_cols, cat_cols = make_preprocessor(train, feature_cols)

    X_train, y_train = train[feature_cols], train[y_col].astype(int).values
    X_val, y_val = val[feature_cols], val[y_col].astype(int).values
    X_test, y_test = test[feature_cols], test[y_col].astype(int).values

    models = {
        "logreg_balanced": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "rf_balanced": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    metrics = {
        "data": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_pos_rate": float(y_train.mean()),
            "val_pos_rate": float(y_val.mean()),
            "test_pos_rate": float(y_test.mean()),
            "n_features_raw": len(feature_cols),
            "n_numeric_features": len(num_cols),
            "n_categorical_features": len(cat_cols),
        },
        "models": {},
    }

    best_name = None
    best_val_ap = -1.0
    best_val_prob = None
    best_test_prob = None

    for name, clf in models.items():
        pipe = Pipeline(
            [
                ("pre", pre),
                ("clf", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        test_prob = pipe.predict_proba(X_test)[:, 1]

        val_metric = eval_binary(y_val, val_prob)
        test_metric = eval_binary(y_test, test_prob)
        metrics["models"][name] = {"val": val_metric, "test": test_metric}

        if val_metric["ap"] > best_val_ap:
            best_val_ap = val_metric["ap"]
            best_name = name
            best_val_prob = val_prob
            best_test_prob = test_prob

    # Persist best model predictions for inspection.
    val_out = val[["Stkcd", "year", y_col]].copy()
    val_out["pred_prob"] = best_val_prob
    val_out["model"] = best_name
    test_out = test[["Stkcd", "year", y_col]].copy()
    test_out["pred_prob"] = best_test_prob
    test_out["model"] = best_name
    val_out.to_csv(OUT_DIR / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_out.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    metrics["best_model_by_val_ap"] = best_name
    (OUT_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
