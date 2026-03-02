#!/usr/bin/env python3
"""
DGL-based HTGNN training on final heterogeneous temporal graph snapshots.

Run example:
  PYTHONPATH=./third_party/python DGLDEFAULTDIR=./.dgl DGLBACKEND=pytorch \
  python3 scripts/train_htgnn_dgl.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEP = PROJECT_ROOT / "third_party" / "python"
if LOCAL_DEP.exists():
    sys.path.insert(0, str(LOCAL_DEP))

os.environ.setdefault("DGLDEFAULTDIR", str(PROJECT_ROOT / ".dgl"))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.makedirs(os.environ["DGLDEFAULTDIR"], exist_ok=True)

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgl.nn import GraphConv, HeteroGraphConv
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


GRAPH_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"
LABEL_PATH = PROJECT_ROOT / "processed" / "stage1" / "rating_task" / "rating_panel_labeled.csv"
OUT_DIR = GRAPH_DIR / "experiments_htgnn_dgl"
RELATIONS = ["guarantee", "equity_assoc", "co_controller"]
NTYPE = "company"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> Tuple[float, float]:
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    idx = np.argsort(-y_prob)[:k]
    hit = y_true[idx].sum()
    return safe_div(hit, k), safe_div(hit, y_true.sum())


def eval_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }
    for frac in (0.01, 0.05, 0.10):
        p, r = precision_recall_at_k(y_true, y_prob, frac)
        out[f"precision@{frac:.2f}"] = float(p)
        out[f"recall@{frac:.2f}"] = float(r)
    return out


def load_node_mapping() -> pd.DataFrame:
    node_map = pd.read_csv(GRAPH_DIR / "node_mapping.csv", dtype={"Stkcd": str})
    node_map["Stkcd"] = node_map["Stkcd"].astype(str).str.zfill(6)
    node_map["node_id"] = pd.to_numeric(node_map["node_id"], errors="coerce").astype(int)
    return node_map.sort_values("node_id").reset_index(drop=True)


def load_snapshots(year_min: int, year_max: int, n_nodes: int, device: torch.device) -> Tuple[Dict[int, dgl.DGLHeteroGraph], Dict[int, pd.DataFrame]]:
    graphs: Dict[int, dgl.DGLHeteroGraph] = {}
    node_frames: Dict[int, pd.DataFrame] = {}

    for year in range(year_min, year_max + 1):
        ydir = GRAPH_DIR / "snapshots" / str(year)
        nf = pd.read_csv(ydir / "node_features.csv", dtype={"Stkcd": str})
        nf["Stkcd"] = nf["Stkcd"].astype(str).str.zfill(6)
        nf["node_id"] = pd.to_numeric(nf["node_id"], errors="coerce").astype(int)
        nf = nf.sort_values("node_id").reset_index(drop=True)
        if len(nf) != n_nodes:
            raise ValueError(f"Year {year}: node count mismatch.")
        node_frames[year] = nf

        data_dict = {}
        edge_weights = {}
        for rel in RELATIONS:
            ep = ydir / f"edges_{rel}.csv"
            e = pd.read_csv(ep)
            if len(e) == 0:
                src = torch.tensor([], dtype=torch.int64)
                dst = torch.tensor([], dtype=torch.int64)
                w = torch.tensor([], dtype=torch.float32)
            else:
                src = torch.tensor(pd.to_numeric(e["src_id"], errors="coerce").fillna(-1).to_numpy(np.int64))
                dst = torch.tensor(pd.to_numeric(e["dst_id"], errors="coerce").fillna(-1).to_numpy(np.int64))
                w = torch.tensor(pd.to_numeric(e["weight"], errors="coerce").fillna(1.0).to_numpy(np.float32))
                mask = (src >= 0) & (dst >= 0)
                src, dst, w = src[mask], dst[mask], w[mask]
            data_dict[(NTYPE, rel, NTYPE)] = (src, dst)
            edge_weights[rel] = w

        g = dgl.heterograph(data_dict, num_nodes_dict={NTYPE: n_nodes}).to(device)
        for rel in RELATIONS:
            g.edges[rel].data["w"] = edge_weights[rel].to(device)
        graphs[year] = g
    return graphs, node_frames


def build_features(node_frames: Dict[int, pd.DataFrame], train_year_end: int) -> Tuple[Dict[int, torch.Tensor], int]:
    years = sorted(node_frames.keys())
    full = pd.concat([node_frames[y].assign(_year=y) for y in years], ignore_index=True)
    id_cols = {"Stkcd", "year", "node_id", "is_active", "_year"}
    cols = [c for c in full.columns if c not in id_cols]
    cat_cols = [c for c in ["Markettype", "Nnindcd", "Statco", "rating_latest_longterm", "rating_latest_prospect"] if c in cols]
    num_cols = [c for c in cols if c not in cat_cols]

    train_mask = full["_year"] <= train_year_end
    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    num_train = full.loc[train_mask, num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler().fit(num_train)

    cat_train = full.loc[train_mask, cat_cols].fillna("NA").astype(str)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(cat_train)

    feats: Dict[int, torch.Tensor] = {}
    for y in years:
        df = node_frames[y].sort_values("node_id").reset_index(drop=True).copy()
        num = df[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        num_x = scaler.transform(num)
        cat = df[cat_cols].fillna("NA").astype(str)
        cat_x = ohe.transform(cat)
        x = np.hstack([num_x, cat_x]).astype(np.float32)
        feats[y] = torch.from_numpy(x)
    return feats, feats[years[0]].shape[1]


def build_labels(node_map: pd.DataFrame, year_min: int, year_max: int) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
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

    splits = {"train": (2010, 2018), "val": (2019, 2021), "test": (2022, 2024)}
    out: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {"train": {}, "val": {}, "test": {}}
    for split, (y0, y1) in splits.items():
        sub = labels[(labels["year"] >= y0) & (labels["year"] <= y1)].copy()
        for year, g in sub.groupby("year"):
            out[split][int(year)] = {
                "node_id": g["node_id"].to_numpy(np.int64),
                "label": g["y"].to_numpy(np.float32),
                "stkcd": g["Stkcd"].to_numpy(object),
            }
    return out


class DGLHTGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, relations: List[str], dropout: float):
        super().__init__()
        self.relations = relations
        self.conv1 = HeteroGraphConv(
            {r: GraphConv(in_dim, hidden_dim, norm="right", weight=True, bias=True, allow_zero_in_degree=True) for r in relations},
            aggregate="sum",
        )
        self.conv2 = HeteroGraphConv(
            {r: GraphConv(hidden_dim, hidden_dim, norm="right", weight=True, bias=True, allow_zero_in_degree=True) for r in relations},
            aggregate="sum",
        )
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cls = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def spatial_forward(self, g: dgl.DGLHeteroGraph, x: torch.Tensor) -> torch.Tensor:
        h = {NTYPE: x}
        kwargs = {r: {"edge_weight": g.edges[r].data["w"]} for r in self.relations}
        h1 = self.conv1(g, h, mod_kwargs=kwargs)[NTYPE]
        h1 = torch.relu(h1)
        h1 = self.drop(h1)
        h2 = self.conv2(g, {NTYPE: h1}, mod_kwargs=kwargs)[NTYPE]
        h2 = torch.relu(h2)
        return h2

    def forward(self, graph_seq: List[dgl.DGLHeteroGraph], feat_seq: List[torch.Tensor]) -> torch.Tensor:
        hs = [self.spatial_forward(g, x) for g, x in zip(graph_seq, feat_seq)]
        h = torch.stack(hs, dim=1)  # [N,T,H]
        out, _ = self.gru(h)
        z = out[:, -1, :]
        logit = self.cls(z).squeeze(-1)
        return logit


def predict_year(
    model: DGLHTGNN,
    year: int,
    window: int,
    graphs: Dict[int, dgl.DGLHeteroGraph],
    features: Dict[int, torch.Tensor],
) -> torch.Tensor:
    ys = list(range(year - window + 1, year + 1))
    gs = [graphs[y] for y in ys]
    xs = [features[y] for y in ys]
    return model(gs, xs)


def evaluate_split(
    model: DGLHTGNN,
    split_data: Dict[int, Dict[str, np.ndarray]],
    years: List[int],
    window: int,
    graphs: Dict[int, dgl.DGLHeteroGraph],
    features: Dict[int, torch.Tensor],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    recs = []
    model.eval()
    with torch.no_grad():
        for year in years:
            if year not in split_data:
                continue
            logits = predict_year(model, year, window, graphs, features)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            ids = split_data[year]["node_id"]
            ys = split_data[year]["label"]
            codes = split_data[year]["stkcd"]
            for nid, code, y, p in zip(ids, codes, ys, prob[ids]):
                recs.append((int(year), int(nid), str(code), float(y), float(p)))
    pred = pd.DataFrame(recs, columns=["year", "node_id", "Stkcd", "label", "pred_prob"])
    metric = eval_binary(pred["label"].to_numpy(np.float32), pred["pred_prob"].to_numpy(np.float32))
    return metric, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DGL HTGNN on final hetero temporal graph.")
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    node_map = load_node_mapping()
    n_nodes = len(node_map)
    graphs, node_frames = load_snapshots(args.year_min, args.year_max, n_nodes, device)
    feat_raw, in_dim = build_features(node_frames, train_year_end=2018)
    features = {y: x.to(device) for y, x in feat_raw.items()}
    labels = build_labels(node_map, args.year_min, args.year_max)

    train_years = sorted([y for y in labels["train"].keys() if y - args.window + 1 >= args.year_min])
    val_years = sorted([y for y in labels["val"].keys() if y - args.window + 1 >= args.year_min])
    test_years = sorted([y for y in labels["test"].keys() if y - args.window + 1 >= args.year_min])

    train_y = np.concatenate([labels["train"][y]["label"] for y in train_years])
    pos, neg = float(train_y.sum()), float(len(train_y) - train_y.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    model = DGLHTGNN(in_dim=in_dim, hidden_dim=args.hidden_dim, relations=RELATIONS, dropout=args.dropout).to(device)
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
            logits = predict_year(model, year, args.window, graphs, features)
            loss = criterion(logits[idx], y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else np.nan
        val_metric, _ = evaluate_split(model, labels["val"], val_years, args.window, graphs, features)
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_ap": val_metric["ap"], "val_auc": val_metric["auc"]})

        if val_metric["ap"] > best_val_ap:
            best_val_ap = val_metric["ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

    val_metric, val_pred = evaluate_split(model, labels["val"], val_years, args.window, graphs, features)
    test_metric, test_pred = evaluate_split(model, labels["test"], test_years, args.window, graphs, features)

    val_pred.to_csv(OUT_DIR / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve).to_csv(OUT_DIR / "train_curve.csv", index=False, encoding="utf-8-sig")

    cfg = vars(args).copy()
    cfg["device_actual"] = str(device)
    cfg["in_dim"] = int(in_dim)
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

