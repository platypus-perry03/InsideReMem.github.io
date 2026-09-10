#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:?phase required}"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SESSION="liref_d07_${PHASE}"
ROOT="/home/jinhyun/prj_ws/jiho"
PYTHON="/home/jinhyun/.conda/envs/torch/bin/python"
SCRIPT="$ROOT/AI/reenact/experiments/rm_decomp_feature_causal/run_feature_patching.py"
CONFIG="$ROOT/AI/reenact/experiments/rm_decomp_feature_causal/exploratory_config.json"
RUN_ID="$("$PYTHON" -c "import json; print(json.load(open('$CONFIG'))['run_id'])")"
LOG="$ROOT/AI/reenact/liref_outputs/rm_decomp/v2/d_feature_causal_${RUN_ID}/logs/${PHASE}.log"

mkdir -p "$(dirname "$LOG")"
tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$GPU_ID' '$PYTHON' -u '$SCRIPT' --phase '$PHASE' --gpu-id 0 --batch-size '$BATCH_SIZE' 2>&1 | tee '$LOG'"
echo "$SESSION"
