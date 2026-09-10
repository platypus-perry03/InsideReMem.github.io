#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/jinhyun/prj_ws/jiho
PYTHON_BIN=/home/jinhyun/.conda/envs/torch/bin/python
RUNNER=AI/reenact/experiments/rm_decomp_behavioral_validation/run_behavioral_validation_v2.py
AUTH=AI/reenact/experiments/rm_decomp_behavioral_validation/execution_authorization_v2_frozen.json
OUTPUT=AI/reenact/liref_outputs/rm_decomp/behavioral_validation_v2/cross_dataset_behavioral_validation_v2_20260901_01

cd "${WORKSPACE}"
mkdir -p "${OUTPUT}/logs"

run_model() {
  local model="$1"
  local gpu="$2"
  local split="$3"
  local slug
  slug=$(printf '%s' "${model}" | tr '/' '_' | tr ' ' '_')
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${RUNNER}" --phase model --model "${model}" --split "${split}" --device cuda:0 --authorization "${AUTH}" 2>&1 | tee "${OUTPUT}/logs/${slug}_${split}.log"
}

summary_value() {
  local path="$1"
  local key="$2"
  "${PYTHON_BIN}" -c "import json; x=json.load(open('${path}')); print(len(x.get('${key}', [])))"
}

run_model Meta-Llama-3-8B 1 primary
META_PRIMARY="${OUTPUT}/Meta-Llama-3-8B/primary/summary.json"
STRICT=$(summary_value "${META_PRIMARY}" strict_components)
PROBABILITY=$(summary_value "${META_PRIMARY}" probability_only_components)

if [[ "${STRICT}" -gt 0 ]]; then
  echo "[gate] Meta primary strict PASS; launching three models"
elif [[ "${PROBABILITY}" -gt 0 ]]; then
  echo "[gate] Meta primary probability-only; running sealed confirmation"
  run_model Meta-Llama-3-8B 1 confirmation
  META_CONFIRM="${OUTPUT}/Meta-Llama-3-8B/confirmation/summary.json"
  STRICT=$(summary_value "${META_CONFIRM}" strict_components)
  if [[ "${STRICT}" -eq 0 ]]; then
    echo "[gate] confirmation has no strict accuracy signal; cross-model launch blocked"
    "${PYTHON_BIN}" "${RUNNER}" --phase report
    exit 0
  fi
  echo "[gate] Meta confirmation strict PASS; launching three models"
else
  echo "[gate] Meta has no qualifying behavioral signal; stopping"
  "${PYTHON_BIN}" "${RUNNER}" --phase report
  exit 0
fi

run_model Mistral-7B-v0.3 2 primary &
PID_MISTRAL=$!
run_model OLMo-2-1124-7B 3 primary &
PID_OLMO=$!
run_model gemma-2-9b 4 primary &
PID_GEMMA=$!
wait "${PID_MISTRAL}"
wait "${PID_OLMO}"
wait "${PID_GEMMA}"
"${PYTHON_BIN}" "${RUNNER}" --phase report
echo "[complete] conditional behavioral validation v2"
