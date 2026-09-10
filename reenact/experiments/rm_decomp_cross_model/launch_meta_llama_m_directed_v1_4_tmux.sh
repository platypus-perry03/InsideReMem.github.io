#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jinhyun/prj_ws/jiho/AI/reenact"
HERE="$ROOT/experiments/rm_decomp_cross_model"
PYTHON="/home/jinhyun/.conda/envs/torch/bin/python"
DESIGN="$HERE/design_v1_4_meta_llama_m_directed_frozen.json"
AUTH="$HERE/execution_authorization_v1_4_meta_llama_m_directed_frozen.json"
LOG="$ROOT/liref_outputs/rm_decomp/meta_llama_m_directed_v1_4/run.log"

mkdir -p "$(dirname "$LOG")"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1

{
  "$PYTHON" run_meta_llama_m_directed_v1_4.py --phase model --model Meta-Llama-3-8B --device cuda:0 --design "$DESIGN" --authorization "$AUTH"
  "$PYTHON" run_meta_llama_m_directed_v1_4.py --phase report --design "$DESIGN"
} 2>&1 | tee "$LOG"
