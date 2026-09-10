#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jinhyun/prj_ws/jiho/AI/reenact"
HERE="$ROOT/experiments/rm_decomp_causal"
PYTHON="/home/jinhyun/.conda/envs/torch/bin/python"
CONFIG="$HERE/config_full20.json"
AUTH="$HERE/execution_authorization_full20_frozen.json"
LOG="$ROOT/liref_outputs/rm_decomp/v2/c_causal_v2_c02_full20/full20_run.log"

cd "$HERE"
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1

{
  "$PYTHON" run_causal.py --phase sanity --config "$CONFIG" --authorization "$AUTH"
  "$PYTHON" run_causal.py --phase gap --config "$CONFIG" --authorization "$AUTH"
  "$PYTHON" run_causal.py --phase gap-report --config "$CONFIG"
} 2>&1 | tee "$LOG"

