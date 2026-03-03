#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT_DIR/third_party/python"
export DGLDEFAULTDIR="$ROOT_DIR/.dgl"
export DGLBACKEND="pytorch"
mkdir -p "$DGLDEFAULTDIR"

run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  "$@"
}

echo "Project root: $ROOT_DIR"
echo "PYTHONPATH: $PYTHONPATH"

run_step "Stage1 data (nodes + rating + guarantee)" \
  python3 scripts/prepare_data_stage1.py --skip-financial

run_step "Stage1 data (nodes + financial)" \
  python3 scripts/prepare_data_stage1.py --skip-rating --skip-guarantee

if [[ "${RUN_DAILY:-0}" == "1" ]]; then
  run_step "Stage1 daily market data (optional, large)" \
    python3 scripts/prepare_data_stage1.py --skip-financial --skip-rating --skip-guarantee --include-daily
fi

run_step "Build company-year panel" \
  python3 scripts/build_training_panel.py

run_step "Build rating task datasets" \
  python3 scripts/build_rating_task_dataset.py

run_step "Train downgrade baseline" \
  python3 scripts/train_rating_downgrade_baseline.py

run_step "Build final hetero temporal graph" \
  python3 scripts/build_final_hetero_temporal_graph.py

if [[ "${RUN_DGL:-0}" == "1" ]]; then
  run_step "Train DGL HTGNN (optional)" \
    python3 scripts/train_htgnn_dgl.py --device cpu
fi

if [[ "${RUN_DGL_IMPROVED:-0}" == "1" ]]; then
  run_step "Train improved DGL HTGNN (optional)" \
    python3 scripts/train_htgnn_dgl_improved.py --device cpu --exp-name v10_all_notab_bce_w6_h128 --relations guarantee,shared_nonlisted_guarantee,equity_assoc,equity_change,co_controller,market_corr --no-tabular-residual --window 6 --hidden-dim 128 --dropout 0.25 --lr 5e-4 --weight-decay 8e-4 --loss bce --epochs 220 --patience 45 --seed 42
fi

echo
echo "All done."
echo "Key outputs:"
echo "  processed/stage1/panel_company_year.csv"
echo "  processed/stage1/rating_task/summary.json"
echo "  processed/stage1/rating_task/experiments/baseline_metrics.json"
echo "  processed/final_hetero_temporal_graph/metadata.json"
if [[ "${RUN_DGL:-0}" == "1" ]]; then
  echo "  processed/final_hetero_temporal_graph/experiments_htgnn_dgl/metrics.json"
fi
if [[ "${RUN_DGL_IMPROVED:-0}" == "1" ]]; then
  echo "  processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/v10_all_notab_bce_w6_h128/metrics.json"
fi
