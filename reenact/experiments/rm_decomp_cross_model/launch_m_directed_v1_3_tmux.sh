#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/jinhyun/prj_ws/jiho"
PYTHON_BIN="/home/jinhyun/.conda/envs/torch/bin/python"
RUNNER="AI/reenact/experiments/rm_decomp_cross_model/run_cross_model_m_directed_v1_3.py"
DESIGN="AI/reenact/experiments/rm_decomp_cross_model/design_v1_3_m_directed_frozen.json"
AUTH="AI/reenact/experiments/rm_decomp_cross_model/execution_authorization_v1_3_m_directed_frozen.json"
OUTPUT="AI/reenact/liref_outputs/rm_decomp/cross_model_m_directed_v1_3"
SOCKET="${WORKSPACE}/.tmux-rm-m-v1-3.sock"
SESSION="rm_m_v1_3"

cd "${WORKSPACE}"
mkdir -p "${OUTPUT}/logs"

if tmux -S "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already exists: ${SESSION}"
  tmux -S "${SOCKET}" list-windows -t "${SESSION}"
  exit 1
fi

tmux -S "${SOCKET}" new-session -d -s "${SESSION}" -n mistral \
  "cd ${WORKSPACE} && CUDA_VISIBLE_DEVICES=1 ${PYTHON_BIN} ${RUNNER} --phase model --model Mistral-7B-v0.3 --device cuda:0 --design ${DESIGN} --authorization ${AUTH} 2>&1 | tee ${OUTPUT}/logs/mistral.log"

tmux -S "${SOCKET}" new-window -t "${SESSION}" -n olmo \
  "cd ${WORKSPACE} && CUDA_VISIBLE_DEVICES=2 ${PYTHON_BIN} ${RUNNER} --phase model --model OLMo-2-1124-7B --device cuda:0 --design ${DESIGN} --authorization ${AUTH} 2>&1 | tee ${OUTPUT}/logs/olmo.log"

tmux -S "${SOCKET}" new-window -t "${SESSION}" -n gemma \
  "cd ${WORKSPACE} && CUDA_VISIBLE_DEVICES=3 ${PYTHON_BIN} ${RUNNER} --phase model --model gemma-2-9b --device cuda:0 --design ${DESIGN} --authorization ${AUTH} 2>&1 | tee ${OUTPUT}/logs/gemma.log"

tmux -S "${SOCKET}" list-windows -t "${SESSION}"
