#!/usr/bin/env python3
"""
Train final HTGNN-style model on the heterogeneous temporal graph package.

This script is framework-light (PyTorch only) and does not require DGL.

Inputs:
  processed/final_hetero_temporal_graph/
  processed/stage1/rating_task/rating_panel_labeled.csv

Outputs:
  processed/final_hetero_temporal_graph/experiments_htgnn/
    config.json
    metrics.json
    train_curve.csv
    val_predictions.csv
    test_predictions.csv
    best_model.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"
LABEL_PATH = PROJECT_ROOT / "processed" / "stage1" / "rating_task" / "rating_panel_labeled.csv"
OUT_DIR = GRAPH_DIR / "experiments_htgnn"

RELATIONS = ["guarantee", "equity_assoc", "co_controller"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> Tuple[float, float, int]:
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    idx = np.argsort(-y_prob)[:k]
    hit = y_true[idx].sum()
    return safe_div(hit, k), safe_div(hit, y_true.sum()), k


def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }
    for frac in (0.01, 0.05, 0.10):
        p, r, _ = precision_recall_at_k(y_true, y_prob, frac)
        out[f"precision@{frac:.2f}"] = float(p)
        out[f"recall@{frac:.2f}"] = float(r)
    return out


def build_sparse_adj(src: np.ndarray, dst: np.ndarray, weight: np.ndarray, n_nodes: int) -> torch.Tensor:
    # Add self-loops to keep isolated nodes stable.
    sl = np.arange(n_nodes, dtype=np.int64)
    src_all = np.concatenate([src, sl])
    dst_all = np.concatenate([dst, sl])
    w_all = np.concatenate([weight, np.ones(n_nodes, dtype=np.float32)])

    # Aggregate duplicates (src,dst).
    tmp = pd.DataFrame({"src": src_all, "dst": dst_all, "w": w_all})
    tmp = tmp.groupby(["src", "dst"], as_index=False)["w"].sum()
    src_u = tmp["src"].to_numpy(dtype=np.int64)
    dst_u = tmp["dst"].to_numpy(dtype=np.int64)
    w_u = tmp["w"].to_numpy(dtype=np.float32)

    # Row normalization.
    deg = np.bincount(src_u, weights=w_u, minlength=n_nodes).astype(np.float32)
    deg[deg <= 0] = 1.0
    w_norm = w_u / deg[src_u]

    idx = torch.tensor(np.vstack([src_u, dst_u]), dtype=torch.long)
    val = torch.tensor(w_norm, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()
    return adj


def load_graph_data(year_min: int, year_max: int) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[int, Dict[str, torch.Tensor]]]:
    node_map = pd.read_csv(GRAPH_DIR / "node_mapping.csv", dtype={"Stkcd": str})
    node_map["Stkcd"] = node_map["Stkcd"].astype(str).str.zfill(6)
    node_map["node_id"] = pd.to_numeric(node_map["node_id"], errors="coerce").astype(int)
    n_nodes = len(node_map)

    node_feats: Dict[int, pd.DataFrame] = {}
    adjs: Dict[int, Dict[str, torch.Tensor]] = {}

    for year in range(year_min, year_max + 1):
        ydir = GRAPH_DIR / "snapshots" / str(year)
        nf = pd.read_csv(ydir / "node_features.csv", dtype={"Stkcd": str})
        nf["Stkcd"] = nf["Stkcd"].astype(str).str.zfill(6)
        nf["node_id"] = pd.to_numeric(nf["node_id"], errors="coerce").astype(int)
        nf = nf.sort_values("node_id").reset_index(drop=True)
        if len(nf) != n_nodes:
            raise ValueError(f"Year {year}: node count mismatch ({len(nf)} vs {n_nodes}).")
        node_feats[year] = nf

        adjs[year] = {}
        for rel in RELATIONS:
            ep = ydir / f"edges_{rel}.csv"
            e = pd.read_csv(ep)
            if len(e) == 0:
                src = np.array([], dtype=np.int64)
                dst = np.array([], dtype=np.int64)
                w = np.array([], dtype=np.float32)
            else:
                src = pd.to_numeric(e["src_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
                dst = pd.to_numeric(e["dst_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
                w = pd.to_numeric(e["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
                mask = (src >= 0) & (dst >= 0)
                src, dst, w = src[mask], dst[mask], w[mask]
            adjs[year][rel] = build_sparse_adj(src, dst, w, n_nodes)
    return node_map, node_feats, adjs


def build_feature_tensors(
    node_feats: Dict[int, pd.DataFrame],
    train_year_end: int,
) -> Tuple[Dict[int, torch.Tensor], List[str]]:
    years = sorted(node_feats.keys())
    all_df = pd.concat([node_feats[y].assign(_year=y) for y in years], ignore_index=True)

    id_cols = {"Stkcd", "year", "node_id", "is_active", "_year"}
    candidate_cols = [c for c in all_df.columns if c not in id_cols]

    cat_cols = [c for c in ["Markettype", "Nnindcd", "Statco", "rating_latest_longterm", "rating_latest_prospect"] if c in candidate_cols]
    num_cols = [c for c in candidate_cols if c not in cat_cols]

    train_mask = all_df["_year"] <= train_year_end

    # Numeric pipeline.
    for c in num_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    num_train = all_df.loc[train_mask, num_cols].copy()
    num_train = num_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(num_train)

    # Categorical pipeline.
    cat_train = all_df.loc[train_mask, cat_cols].fillna("NA").astype(str)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(cat_train)

    features: Dict[int, torch.Tensor] = {}
    feat_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
    for y in years:
        df = node_feats[y].sort_values("node_id").reset_index(drop=True).copy()
        num = df[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        num_x = scaler.transform(num)
        cat = df[cat_cols].fillna("NA").astype(str)
        cat_x = ohe.transform(cat)
        x = np.hstack([num_x, cat_x]).astype(np.float32)
        features[y] = torch.from_numpy(x)
    return features, feat_names


def build_labels(
    node_map: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    labels = pd.read_csv(LABEL_PATH, dtype={"Stkcd": str})
    labels["Stkcd"] = labels["Stkcd"].astype(str).str.zfill(6)
    labels["year"] = pd.to_numeric(labels["year"], errors="coerce").astype(int)
    labels["label_downgrade_next_year"] = pd.to_numeric(labels["label_downgrade_next_year"], errors="coerce")
    labels = labels.dropna(subset=["label_downgrade_next_year"])
    labels = labels[(labels["year"] >= year_min) & (labels["year"] <= year_max)].copy()

    code_to_id = dict(zip(node_map["Stkcd"], node_map["node_id"]))
    labels["node_id"] = labels["Stkcd"].map(code_to_id)
    labels = labels[labels["node_id"].notna()].copy()
    labels["node_id"] = labels["node_id"].astype(int)
    labels["y"] = labels["label_downgrade_next_year"].astype(int)

    splits = {
        "train": (2010, 2018),
        "val": (2019, 2021),
        "test": (2022, 2024),
    }
    out: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {"train": {}, "val": {}, "test": {}}
    for split, (y0, y1) in splits.items():
        sub = labels[(labels["year"] >= y0) & (labels["year"] <= y1)].copy()
        for year, g in sub.groupby("year"):
            out[split][int(year)] = {
                "node_id": g["node_id"].to_numpy(dtype=np.int64),
                "label": g["y"].to_numpy(dtype=np.float32),
                "stkcd": g["Stkcd"].to_numpy(dtype=object),
            }
    return out


class HTGNNFinal(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, relations: List[str], dropout: float = 0.2):
        super().__init__()
        self.relations = relations
        self.rel_linears = nn.ModuleDict({r: nn.Linear(in_dim, hidden_dim) for r in relations})
        self.rel_query = nn.Parameter(torch.randn(hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat_seq: List[torch.Tensor], adj_seq: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h_time = []
        for x, adjs in zip(feat_seq, adj_seq):
            rel_msgs = []
            for r in self.relations:
                z = self.rel_linears[r](x)
                m = torch.sparse.mm(adjs[r], z)
                rel_msgs.append(m)
            rel_stack = torch.stack(rel_msgs, dim=0)  # [R, N, H]
            score = torch.einsum("rnh,h->rn", rel_stack, self.rel_query)
            alpha = torch.softmax(score, dim=0).unsqueeze(-1)
            h = (alpha * rel_stack).sum(dim=0)
            h_time.append(torch.relu(h))
        h_seq = torch.stack(h_time, dim=1)  # [N, T, H]
        out, _ = self.gru(h_seq)
        h_last = out[:, -1, :]
        logits = self.classifier(h_last).squeeze(-1)  # [N]
        return logits


def predict_year(
    model: nn.Module,
    year: int,
    window: int,
    features: Dict[int, torch.Tensor],
    adjs: Dict[int, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    ys = list(range(year - window + 1, year + 1))
    feat_seq = [features[y] for y in ys]
    adj_seq = [adjs[y] for y in ys]
    return model(feat_seq, adj_seq)


def evaluate_split(
    model: nn.Module,
    split_data: Dict[int, Dict[str, np.ndarray]],
    years: List[int],
    window: int,
    features: Dict[int, torch.Tensor],
    adjs: Dict[int, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    recs = []
    with torch.no_grad():
        for year in years:
            if year not in split_data:
                continue
            if year - window + 1 < min(features.keys()):
                continue
            logits = predict_year(model, year, window, features, adjs)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            ids = split_data[year]["node_id"]
            y = split_data[year]["label"]
            stk = split_data[year]["stkcd"]
            for nid, code, yy, pp in zip(ids, stk, y, probs[ids]):
                recs.append((int(year), int(nid), str(code), float(yy), float(pp)))
    pred_df = pd.DataFrame(recs, columns=["year", "node_id", "Stkcd", "label", "pred_prob"])
    m = eval_binary(pred_df["label"].to_numpy(dtype=np.float32), pred_df["pred_prob"].to_numpy(dtype=np.float32))
    return m, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final HTGNN-style model on yearly heterogeneous graph snapshots.")
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    node_map, node_feats_raw, adjs_raw = load_graph_data(args.year_min, args.year_max)
    features, feat_names = build_feature_tensors(node_feats_raw, train_year_end=2018)
    labels = build_labels(node_map, args.year_min, args.year_max)

    # Move tensors to device.
    features = {y: x.to(device) for y, x in features.items()}
    adjs = {y: {r: a.to(device) for r, a in ad.items()} for y, ad in adjs_raw.items()}

    train_years = sorted([y for y in labels["train"].keys() if y - args.window + 1 >= args.year_min])
    val_years = sorted([y for y in labels["val"].keys() if y - args.window + 1 >= args.year_min])
    test_years = sorted([y for y in labels["test"].keys() if y - args.window + 1 >= args.year_min])

    if not train_years:
        raise RuntimeError("No train years available after window constraint.")

    # Class imbalance weight from training labels.
    train_y = np.concatenate([labels["train"][y]["label"] for y in train_years])
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    model = HTGNNFinal(in_dim=features[train_years[0]].shape[1], hidden_dim=args.hidden_dim, relations=RELATIONS, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_ap = -1.0
    patience_count = 0
    curve = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        random.shuffle(train_years)
        for year in train_years:
            d = labels["train"][year]
            idx = torch.tensor(d["node_id"], dtype=torch.long, device=device)
            y = torch.tensor(d["label"], dtype=torch.float32, device=device)
            logits = predict_year(model, year, args.window, features, adjs)
            loss = criterion(logits[idx], y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else np.nan
        val_metric, _ = evaluate_split(model, labels["val"], val_years, args.window, features, adjs, device)
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_ap": val_metric["ap"], "val_auc": val_metric["auc"]})

        if val_metric["ap"] > best_val_ap:
            best_val_ap = val_metric["ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state captured.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

    val_metric, val_pred = evaluate_split(model, labels["val"], val_years, args.window, features, adjs, device)
    test_metric, test_pred = evaluate_split(model, labels["test"], test_years, args.window, features, adjs, device)
    val_pred.to_csv(OUT_DIR / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve).to_csv(OUT_DIR / "train_curve.csv", index=False, encoding="utf-8-sig")

    cfg = vars(args).copy()
    cfg["device_actual"] = str(device)
    cfg["in_dim"] = int(features[train_years[0]].shape[1])
    cfg["feature_count"] = len(feat_names)
    cfg["train_years"] = train_years
    cfg["val_years"] = val_years
    cfg["test_years"] = test_years
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "val": val_metric,
        "test": test_metric,
        "train_rows": int(len(train_y)),
        "val_rows": int(len(val_pred)),
        "test_rows": int(len(test_pred)),
        "best_val_ap": float(best_val_ap),
        "epochs_ran": int(len(curve)),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

