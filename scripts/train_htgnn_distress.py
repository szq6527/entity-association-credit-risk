#!/usr/bin/env python3
"""
DGL HTGNN training for financial distress (loss-next-year) prediction.

Based on train_htgnn_dgl_improved.py, adapted for:
- distress label (net_profit_next_year < 0) instead of rating downgrade
- much larger sample size (~24k vs ~3k)
- all A-share listed companies
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
    # Keep system numpy/torch ahead of vendored deps.
    sys.path.append(str(LOCAL_DEP))

os.environ.setdefault("DGLDEFAULTDIR", str(PROJECT_ROOT / ".dgl"))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.makedirs(os.environ["DGLDEFAULTDIR"], exist_ok=True)

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


GRAPH_DIR = PROJECT_ROOT / "processed" / "final_hetero_temporal_graph"
LABEL_PATH = PROJECT_ROOT / "processed" / "stage1" / "distress_task" / "distress_panel_labeled.csv"
OUT_ROOT = GRAPH_DIR / "experiments_htgnn_distress"
DEFAULT_RELATIONS = [
    "guarantee",
    "shared_nonlisted_guarantee",
    "equity_assoc",
    "equity_change",
    "co_controller",
    "market_corr",
    "industry",
    # v3 improved edges
    "market_corr_strong",
    "co_controller_weighted",
    "industry_knn",
]
DISTRESS_PANEL_PATH = PROJECT_ROOT / "processed" / "stage1" / "distress_task" / "distress_panel_labeled.csv"
NTYPE = "company"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_relations(text: str) -> List[str]:
    rels = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [r for r in rels if r not in DEFAULT_RELATIONS]
    if bad:
        raise ValueError(f"Unknown relations: {bad}. Supported: {DEFAULT_RELATIONS}")
    if not rels:
        raise ValueError("At least one relation is required.")
    return rels


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
        "threshold": float(threshold),
    }
    for frac in (0.01, 0.05, 0.10):
        p, r = precision_recall_at_k(y_true, y_prob, frac)
        out[f"precision@{frac:.2f}"] = float(p)
        out[f"recall@{frac:.2f}"] = float(r)
    return out


def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t


def load_node_mapping() -> pd.DataFrame:
    node_map = pd.read_csv(GRAPH_DIR / "node_mapping.csv", dtype={"Stkcd": str})
    node_map["Stkcd"] = node_map["Stkcd"].astype(str).str.zfill(6)
    node_map["node_id"] = pd.to_numeric(node_map["node_id"], errors="coerce").astype(int)
    return node_map.sort_values("node_id").reset_index(drop=True)


def load_snapshots_with_graph_stats(
    year_min: int,
    year_max: int,
    n_nodes: int,
    relations: List[str],
    device: torch.device,
) -> Tuple[Dict[int, dgl.DGLHeteroGraph], Dict[int, pd.DataFrame]]:
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

        data_dict = {}
        edge_weights = {}
        for rel in relations:
            ep = ydir / f"edges_{rel}.csv"
            if ep.exists():
                e = pd.read_csv(ep)
            else:
                e = pd.DataFrame(columns=["src_id", "dst_id", "weight"])
            if len(e) == 0:
                src = np.array([], dtype=np.int64)
                dst = np.array([], dtype=np.int64)
                w = np.array([], dtype=np.float32)
            else:
                src = pd.to_numeric(e["src_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
                dst = pd.to_numeric(e["dst_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
                w = pd.to_numeric(e["weight"], errors="coerce").fillna(1.0).to_numpy(np.float32)
                m = (src >= 0) & (dst >= 0)
                src, dst, w = src[m], dst[m], w[m]

            data_dict[(NTYPE, rel, NTYPE)] = (torch.from_numpy(src), torch.from_numpy(dst))
            edge_weights[rel] = torch.from_numpy(w)

            # Graph-structure features for this year.
            out_cnt = np.bincount(src, minlength=n_nodes).astype(np.float32)
            in_cnt = np.bincount(dst, minlength=n_nodes).astype(np.float32)
            out_w = np.bincount(src, weights=w, minlength=n_nodes).astype(np.float32)
            in_w = np.bincount(dst, weights=w, minlength=n_nodes).astype(np.float32)
            nf[f"{rel}_out_cnt"] = out_cnt
            nf[f"{rel}_in_cnt"] = in_cnt
            nf[f"{rel}_out_wsum"] = out_w
            nf[f"{rel}_in_wsum"] = in_w

        g = dgl.heterograph(data_dict, num_nodes_dict={NTYPE: n_nodes}).to(device)
        for rel in relations:
            g.edges[rel].data["w"] = edge_weights[rel].to(device)
        graphs[year] = g
        node_frames[year] = nf

    return graphs, node_frames


def build_features(node_frames: Dict[int, pd.DataFrame], train_year_end: int) -> Tuple[Dict[int, torch.Tensor], int]:
    years = sorted(node_frames.keys())
    full = pd.concat([node_frames[y].assign(_year=y) for y in years], ignore_index=True)

    # Inject strong features from distress panel (net_profit, ROA, etc.)
    if DISTRESS_PANEL_PATH.exists():
        dp = pd.read_csv(DISTRESS_PANEL_PATH, dtype={"Stkcd": str})
        dp["Stkcd"] = dp["Stkcd"].astype(str).str.zfill(6)
        dp["year"] = pd.to_numeric(dp["year"], errors="coerce").astype(int)
        inject_cols = ["net_profit", "roa", "equity_ratio", "cashflow_to_assets", "is_loss_current_year", "net_profit_next_year"]
        inject_cols = [c for c in inject_cols if c in dp.columns and c != "net_profit_next_year"]  # don't leak label
        dp_sub = dp[["Stkcd", "year"] + inject_cols].copy()
        full = full.merge(dp_sub, left_on=["Stkcd", "_year"], right_on=["Stkcd", "year"], how="left", suffixes=("", "_dp"))
        if "year_dp" in full.columns:
            full.drop(columns=["year_dp"], inplace=True)

    id_cols = {"Stkcd", "year", "node_id", "is_active", "_year"}
    cols = [c for c in full.columns if c not in id_cols]
    cat_cols = [c for c in ["Markettype", "Nnindcd", "Statco", "rating_latest_longterm", "rating_latest_prospect"] if c in cols]
    num_cols = [c for c in cols if c not in cat_cols]

    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")

    # Log-transform heavy-tailed fields.
    skew_cols = [c for c in ["Regcap", "total_assets", "total_liabilities", "total_equity", "revenue_total", "revenue_main", "cashflow_operating", "guar_amt_sum", "guar_amt_mean", "net_profit"] if c in num_cols]
    for c in skew_cols:
        full[f"log1p_{c}"] = np.log1p(np.clip(full[c].fillna(0).to_numpy(dtype=np.float64), a_min=0, a_max=None))

    # Lag and change features.
    lag_base = [c for c in ["total_assets", "total_liabilities", "total_equity", "revenue_total", "cashflow_operating", "asset_liability_ratio", "guar_event_cnt", "guar_amt_sum", "rating_event_cnt", "net_profit", "roa"] if c in full.columns]
    full = full.sort_values(["node_id", "_year"]).reset_index(drop=True)
    for c in lag_base:
        full[f"{c}_lag1"] = full.groupby("node_id")[c].shift(1)
        full[f"{c}_chg1"] = full[c] - full[f"{c}_lag1"]

    full["year_idx"] = full["_year"] - int(min(years))

    # Recompute numeric columns after feature expansion.
    id_cols2 = {"Stkcd", "year", "node_id", "is_active", "_year"}
    cols2 = [c for c in full.columns if c not in id_cols2]
    cat_cols = [c for c in ["Markettype", "Nnindcd", "Statco", "rating_latest_longterm", "rating_latest_prospect"] if c in cols2]
    num_cols = [c for c in cols2 if c not in cat_cols]

    train_mask = full["_year"] <= train_year_end
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
        df = full[full["_year"] == y].sort_values("node_id").reset_index(drop=True)
        num = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
    labels["label_loss_next_year"] = pd.to_numeric(labels["label_loss_next_year"], errors="coerce")
    labels = labels.dropna(subset=["label_loss_next_year"])
    labels = labels[(labels["year"] >= year_min) & (labels["year"] <= year_max)].copy()
    code_to_id = dict(zip(node_map["Stkcd"], node_map["node_id"]))
    labels["node_id"] = labels["Stkcd"].map(code_to_id)
    labels = labels[labels["node_id"].notna()].copy()
    labels["node_id"] = labels["node_id"].astype(int)
    labels["y"] = labels["label_loss_next_year"].astype(int)

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


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        pt = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()


class RelationAttnTemporalHTGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, relations: List[str], dropout: float, use_tabular_residual: bool = True):
        super().__init__()
        self.relations = relations
        self.use_tabular_residual = use_tabular_residual
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.rel_conv1 = nn.ModuleDict(
            {r: GraphConv(hidden_dim, hidden_dim, norm="right", weight=True, bias=True, allow_zero_in_degree=True) for r in relations}
        )
        self.rel_conv2 = nn.ModuleDict(
            {r: GraphConv(hidden_dim, hidden_dim, norm="right", weight=True, bias=True, allow_zero_in_degree=True) for r in relations}
        )
        self.rel_q1 = nn.Parameter(torch.randn(hidden_dim))
        self.rel_q2 = nn.Parameter(torch.randn(hidden_dim))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temp_q = nn.Parameter(torch.randn(hidden_dim))

        if use_tabular_residual:
            self.tab_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            cls_in = hidden_dim * 3
        else:
            cls_in = hidden_dim * 2

        self.cls = nn.Sequential(
            nn.Linear(cls_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _relation_attn(self, hs: List[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
        stack = torch.stack(hs, dim=0)  # [R,N,H]
        score = torch.einsum("rnh,h->rn", torch.tanh(stack), q)
        alpha = torch.softmax(score, dim=0).unsqueeze(-1)
        out = (alpha * stack).sum(dim=0)
        return out

    def spatial_forward(self, g: dgl.DGLHeteroGraph, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.in_proj(x))
        x = self.drop(x)

        rel_h1 = []
        for r in self.relations:
            gr = g[(NTYPE, r, NTYPE)]
            h = self.rel_conv1[r](gr, x, edge_weight=gr.edata["w"])
            rel_h1.append(h)
        h1 = self._relation_attn(rel_h1, self.rel_q1)
        h1 = self.norm1(x + self.drop(torch.relu(h1)))

        rel_h2 = []
        for r in self.relations:
            gr = g[(NTYPE, r, NTYPE)]
            h = self.rel_conv2[r](gr, h1, edge_weight=gr.edata["w"])
            rel_h2.append(h)
        h2 = self._relation_attn(rel_h2, self.rel_q2)
        h2 = self.norm2(h1 + self.drop(torch.relu(h2)))
        return h2

    def forward(self, graph_seq: List[dgl.DGLHeteroGraph], feat_seq: List[torch.Tensor]) -> torch.Tensor:
        hs = [self.spatial_forward(g, x) for g, x in zip(graph_seq, feat_seq)]
        seq = torch.stack(hs, dim=1)  # [N,T,H]
        out, _ = self.gru(seq)
        last = out[:, -1, :]
        t_score = torch.einsum("nth,h->nt", torch.tanh(out), self.temp_q)
        t_alpha = torch.softmax(t_score, dim=1).unsqueeze(-1)
        attn = (t_alpha * out).sum(dim=1)
        if self.use_tabular_residual:
            tab = self.tab_mlp(feat_seq[-1])
            z = torch.cat([last, attn, tab], dim=-1)
        else:
            z = torch.cat([last, attn], dim=-1)
        return self.cls(z).squeeze(-1)


def predict_year(model: RelationAttnTemporalHTGNN, year: int, window: int, graphs: Dict[int, dgl.DGLHeteroGraph], features: Dict[int, torch.Tensor]) -> torch.Tensor:
    ys = list(range(year - window + 1, year + 1))
    gs = [graphs[y] for y in ys]
    xs = [features[y] for y in ys]
    return model(gs, xs)


def evaluate_split(
    model: RelationAttnTemporalHTGNN,
    split_data: Dict[int, Dict[str, np.ndarray]],
    years: List[int],
    window: int,
    graphs: Dict[int, dgl.DGLHeteroGraph],
    features: Dict[int, torch.Tensor],
    threshold: float = 0.5,
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
    metric = eval_binary(pred["label"].to_numpy(np.float32), pred["pred_prob"].to_numpy(np.float32), threshold=threshold)
    return metric, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved DGL HTGNN training.")
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--loss", type=str, choices=["focal", "bce"], default="focal")
    parser.add_argument(
        "--relations",
        type=str,
        default="guarantee,shared_nonlisted_guarantee,equity_assoc,equity_change,co_controller_weighted,market_corr_strong,industry_knn",
    )
    parser.add_argument("--no-tabular-residual", action="store_true")
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda")
    relations = parse_relations(args.relations)
    use_tabular_residual = not args.no_tabular_residual

    node_map = load_node_mapping()
    n_nodes = len(node_map)
    graphs, node_frames = load_snapshots_with_graph_stats(args.year_min, args.year_max, n_nodes, relations, device)
    feat_raw, in_dim = build_features(node_frames, train_year_end=2018)
    features = {y: x.to(device) for y, x in feat_raw.items()}
    labels = build_labels(node_map, args.year_min, args.year_max)

    train_years = sorted([y for y in labels["train"].keys() if y - args.window + 1 >= args.year_min])
    val_years = sorted([y for y in labels["val"].keys() if y - args.window + 1 >= args.year_min])
    test_years = sorted([y for y in labels["test"].keys() if y - args.window + 1 >= args.year_min])

    train_y = np.concatenate([labels["train"][y]["label"] for y in train_years]).astype(np.float32)
    pos, neg = float(train_y.sum()), float(len(train_y) - train_y.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    model = RelationAttnTemporalHTGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        relations=relations,
        dropout=args.dropout,
        use_tabular_residual=use_tabular_residual,
    ).to(device)
    if args.loss == "focal":
        criterion: nn.Module = FocalBCEWithLogitsLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4, min_lr=1e-5)

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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else np.nan
        val_metric_05, val_pred = evaluate_split(model, labels["val"], val_years, args.window, graphs, features, threshold=0.5)
        scheduler.step(val_metric_05["ap"])
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_ap@0.5": val_metric_05["ap"], "val_auc": val_metric_05["auc"], "lr": optimizer.param_groups[0]["lr"]})

        if val_metric_05["ap"] > best_val_ap:
            best_val_ap = val_metric_05["ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")

    # Threshold tuning on validation set.
    _, val_pred_for_tune = evaluate_split(model, labels["val"], val_years, args.window, graphs, features, threshold=0.5)
    best_t = best_threshold_by_f1(val_pred_for_tune["label"].to_numpy(np.float32), val_pred_for_tune["pred_prob"].to_numpy(np.float32))

    val_metric_05, val_pred = evaluate_split(model, labels["val"], val_years, args.window, graphs, features, threshold=0.5)
    test_metric_05, test_pred = evaluate_split(model, labels["test"], test_years, args.window, graphs, features, threshold=0.5)
    val_metric_tuned = eval_binary(val_pred["label"].to_numpy(np.float32), val_pred["pred_prob"].to_numpy(np.float32), threshold=best_t)
    test_metric_tuned = eval_binary(test_pred["label"].to_numpy(np.float32), test_pred["pred_prob"].to_numpy(np.float32), threshold=best_t)

    val_pred.to_csv(out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve).to_csv(out_dir / "train_curve.csv", index=False, encoding="utf-8-sig")

    config = vars(args).copy()
    config["relations_parsed"] = relations
    config["use_tabular_residual"] = use_tabular_residual
    config["device_actual"] = str(device)
    config["in_dim"] = int(in_dim)
    config["train_years"] = train_years
    config["val_years"] = val_years
    config["test_years"] = test_years
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "val@0.5": val_metric_05,
        "test@0.5": test_metric_05,
        "best_threshold_by_val_f1": float(best_t),
        "val@tuned": val_metric_tuned,
        "test@tuned": test_metric_tuned,
        "train_rows": int(len(train_y)),
        "val_rows": int(len(val_pred)),
        "test_rows": int(len(test_pred)),
        "best_val_ap": float(best_val_ap),
        "epochs_ran": int(len(curve)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
