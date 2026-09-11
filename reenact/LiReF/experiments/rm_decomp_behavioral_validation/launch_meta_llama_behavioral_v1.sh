#!/usr/bin/env bash
set -euo pipefail

cd /home/jinhyun/prj_ws/jiho/AI/reenact
export CUDA_VISIBLE_DEVICES=1

PYTHON_BIN=/home/jinhyun/.conda/envs/torch/bin/python
RUNNER=experiments/rm_decomp_behavioral_validation/run_meta_llama_behavioral_validation_v1.py
AUTH=experiments/rm_decomp_behavioral_validation/execution_authorization_v1_frozen.json
LOG=experiments/rm_decomp_behavioral_validation/meta_llama_behavioral_prevalidation_20260901_01.log

"${PYTHON_BIN}" "${RUNNER}" --phase execute --authorization "${AUTH}" --device cuda:0 2>&1 | tee "${LOG}"
"${PYTHON_BIN}" "${RUNNER}" --phase report
