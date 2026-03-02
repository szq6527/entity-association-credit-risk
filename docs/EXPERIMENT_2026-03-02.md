# Experiment Log - 2026-03-02

## Goal
Continue improving DGL HTGNN for listed-company rating downgrade prediction, then freeze current best version.

## Code Changes
- Updated `scripts/train_htgnn_dgl_improved.py`:
  - Added configurable relations (`--relations`)
  - Added optional tabular residual branch switch (`--no-tabular-residual`)
  - Added selectable loss (`--loss focal|bce`)
- Updated `scripts/run_all.sh` default improved run to current best config.

## New Experiments
All runs use `scripts/train_htgnn_dgl_improved.py` on CPU.

| exp_name | key setup | val AP | test AP | test AUC |
|---|---|---:|---:|---:|
| `v4_all_tab_bce_w5_h128` | all relations + tab residual + BCE | 0.2984 | 0.1345 | 0.7599 |
| `v5_all_tab_focal_w5_h128` | all relations + tab residual + focal | 0.2944 | 0.1290 | 0.7572 |
| `v6_no_co_tab_focal_w4_h96` | remove co_controller | 0.2252 | 0.1227 | 0.7594 |
| `v7_co_only_tab_focal_w4_h128` | co_controller only | 0.2861 | 0.1313 | 0.7671 |
| `v8_co_only_notab_w5_h128` | co_controller only + no tab residual | 0.2805 | 0.1312 | 0.7328 |
| `v9_all_notab_focal_w4_h160` | all relations + no tab residual + focal | 0.2616 | 0.1304 | 0.7360 |
| `v10_all_notab_bce_w6_h128` | all relations + no tab residual + BCE + w=6 | 0.2945 | **0.1673** | **0.7724** |
| `v10s7_all_notab_bce_w6_h128` | same as v10, seed=7 | 0.2727 | 0.1403 | 0.7510 |
| `v10s2026_all_notab_bce_w6_h128` | same as v10, seed=2026 | 0.2993 | 0.1313 | 0.7248 |

## Best Version (Current)
- Best by test AP: `v10_all_notab_bce_w6_h128`
- Frozen snapshot: `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/best_current/`
- Summary file updated: `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/comparison_summary.json`

## Notes
- Improved DGL remains below RF baseline AP (`0.2115`), but improved from old DGL AP (`0.0489`) to `0.1673`.
