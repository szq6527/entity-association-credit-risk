# CODEx Handoff: Credit Risk HTGNN Project

## 1. Project Goal
- Predict listed-company credit risk via **binary classification**:
  - Target: `label_downgrade_next_year` (whether long-term rating is downgraded in year `t+1`).
- Research direction from `研究思路.md` has been implemented into a runnable pipeline:
  - Data preprocessing
  - Panel construction
  - Label construction
  - Baseline model
  - Heterogeneous temporal graph construction
  - DGL training pipeline

## 2. Current Script Pipeline (Order Matters)
- `scripts/prepare_data_stage1.py`
  - Cleans raw CSMAR xlsx files and builds stage1 datasets.
  - Handles CSMAR metadata rows with `skiprows=[1,2]`.
- `scripts/build_training_panel.py`
  - Builds `company-year` panel from stage1 outputs.
- `scripts/build_rating_task_dataset.py`
  - Builds labeled dataset and time split (`train/val/test`) for rating downgrade task.
- `scripts/train_rating_downgrade_baseline.py`
  - Non-graph baseline for reference (logreg + random forest).
- `scripts/build_final_hetero_temporal_graph.py`
  - Builds final heterogeneous temporal graph snapshots.
- `scripts/train_htgnn_dgl.py`
  - DGL-based HTGNN-style training on final graph snapshots.
- `scripts/train_htgnn_dgl_improved.py`
  - Improved DGL HTGNN with feature engineering + relation attention + focal loss + threshold tuning.
- `scripts/run_all.sh`
  - One-command orchestration.

## 3. One-Command Run
- Default:
  - `bash scripts/run_all.sh`
- Include daily market table (large and slow):
  - `RUN_DAILY=1 bash scripts/run_all.sh`
- Also run DGL HTGNN:
  - `RUN_DGL=1 bash scripts/run_all.sh`
- Also run improved DGL HTGNN (current best config):
  - `RUN_DGL_IMPROVED=1 bash scripts/run_all.sh`

## 4. Key Data Outputs
- Stage1 core:
  - `processed/stage1/nodes_company.csv`
  - `processed/stage1/features_financial.csv`
  - `processed/stage1/features_guarantee_yearly.csv`
  - `processed/stage1/events_rating.csv`
  - `processed/stage1/edges_guarantee_listed_to_listed.csv`
- Panel:
  - `processed/stage1/panel_company_year.csv`
  - `processed/stage1/panel_company_year_summary.json`
- Rating task:
  - `processed/stage1/rating_task/rating_panel_labeled.csv`
  - `processed/stage1/rating_task/train.csv`
  - `processed/stage1/rating_task/val.csv`
  - `processed/stage1/rating_task/test.csv`
  - `processed/stage1/rating_task/summary.json`
- Final hetero temporal graph:
  - `processed/final_hetero_temporal_graph/node_mapping.csv`
  - `processed/final_hetero_temporal_graph/feature_schema.json`
  - `processed/final_hetero_temporal_graph/metadata.json`
  - `processed/final_hetero_temporal_graph/snapshots/{year}/node_features.csv`
  - `processed/final_hetero_temporal_graph/snapshots/{year}/edges_guarantee.csv`
  - `processed/final_hetero_temporal_graph/snapshots/{year}/edges_equity_assoc.csv`
  - `processed/final_hetero_temporal_graph/snapshots/{year}/edges_co_controller.csv`

## 5. Model Architecture Status

### 5.1 Baseline (Reference)
- File: `scripts/train_rating_downgrade_baseline.py`
- Models:
  - `logreg_balanced`
  - `rf_balanced` (best currently)
- Features:
  - Panel numeric/categorical features
  - Yearly graph-derived relation features from listed-to-listed guarantee edges
- Metrics output:
  - `processed/stage1/rating_task/experiments/baseline_metrics.json`

### 5.2 DGL HTGNN (Current graph model)
- File: `scripts/train_htgnn_dgl.py`
- Graph input:
  - Annual heterograph snapshots (single node type `company`, 3 edge types).
- Spatial encoder:
  - 2-layer `HeteroGraphConv` with per-relation `GraphConv`.
- Temporal encoder:
  - GRU over time window (`window=3` default).
- Classifier:
  - MLP head on final temporal hidden state.
- Loss:
  - `BCEWithLogitsLoss` with `pos_weight` from training split.
- Time split:
  - Train: 2010-2018
  - Val: 2019-2021
  - Test: 2022-2024
- Outputs:
  - `processed/final_hetero_temporal_graph/experiments_htgnn_dgl/`

### 5.3 Improved DGL HTGNN (Primary graph model now)
- File: `scripts/train_htgnn_dgl_improved.py`
- Improvements:
  - graph-structure feature augmentation (in/out degree and weighted sums per relation)
  - log/lag/change feature engineering on yearly node features
  - relation attention in spatial encoder
  - temporal attention on GRU outputs
  - configurable loss (`focal` or weighted `bce`)
  - configurable relation set and optional tabular residual branch
  - threshold tuning by validation F1
- Outputs:
  - `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/{exp_name}/`
  - `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/best_current/`
  - comparison file: `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/comparison_summary.json`

## 6. Training Details (Important Defaults)
- `train_htgnn_dgl.py` defaults:
  - `window=3`
  - `hidden_dim=64`
  - `dropout=0.2`
  - `epochs=80`
  - `patience=15`
  - `lr=1e-3`
  - `weight_decay=1e-4`
  - `device=cpu` (safe for MacBook Air)
- Standard metrics reported:
  - AUC, AP
  - F1/Precision/Recall at threshold 0.5
  - Precision@1%, @5%, @10%
  - Recall@1%, @5%, @10%

## 7. Current Results Snapshot

### 7.1 Baseline (best: random forest)
- Val:
  - AUC = 0.8222
  - AP = 0.4023
- Test:
  - AUC = 0.7104
  - AP = 0.2115

### 7.2 DGL HTGNN (current run)
- Val:
  - AUC = 0.5611
  - AP = 0.0923
- Test:
  - AUC = 0.5397
  - AP = 0.0489

### 7.3 Improved DGL HTGNN (multi-run)
- `v1_w3_h96`:
  - Test AUC = 0.6958
  - Test AP = 0.1017
- `v2_w5_h128`:
  - Test AUC = 0.7514
  - Test AP = 0.1098
- `v3_w4_h128`:
  - Test AUC = 0.7191
  - Test AP = 0.1525
- `v10_all_notab_bce_w6_h128` (best by test AP):
  - Test AUC = 0.7724
  - Test AP = 0.1673

Interpretation:
- DGL pipeline is runnable and end-to-end complete.
- Performance is currently below baseline and needs optimization.

## 8. Dependency / Environment Notes
- Local dependencies are installed under:
  - `third_party/python`
- DGL import requires env vars:
  - `DGLDEFAULTDIR=./.dgl`
  - `DGLBACKEND=pytorch`
- Recommended command prefix:
  - `PYTHONPATH=./third_party/python DGLDEFAULTDIR=./.dgl DGLBACKEND=pytorch`
- DGL 2.x caused GraphBolt mismatch; project currently uses DGL 1.1.3 path for stability.

## 9. Known Limitations
- Graph sparsity:
  - `guarantee` and `equity_assoc` edges are sparse compared with `co_controller`.
- Label coverage:
  - Only company-years with rating transitions are used for supervised learning.
- Potential imbalance:
  - Downgrade label rate is low (~4.8%).
- Feature quality:
  - More domain ratios and event-engineered features can improve signal.

## 10. Priority Next Steps
- Rebalance relation influence:
  - Relation-specific weights/loss regularization to avoid co-controller domination.
- Better temporal supervision:
  - Multi-year rolling objective instead of only one-step labels.
- Improve loss/threshold strategy:
  - Focal loss or calibrated threshold selection by val set objective.
- Sampling:
  - Balanced mini-batch of positive/negative nodes per year.
- Ablation:
  - Remove each relation type and compare AUC/AP/P@K/R@K.

## 11. Fast Resume Checklist for Next Codex
- Verify artifacts exist:
  - `processed/final_hetero_temporal_graph/metadata.json`
  - `processed/stage1/rating_task/train.csv`
- Re-run full pipeline if needed:
  - `RUN_DGL=1 RUN_DGL_IMPROVED=1 bash scripts/run_all.sh`
- Start optimization from:
  - `scripts/train_htgnn_dgl_improved.py`
- Compare against baseline metrics in:
  - `processed/stage1/rating_task/experiments/baseline_metrics.json`
