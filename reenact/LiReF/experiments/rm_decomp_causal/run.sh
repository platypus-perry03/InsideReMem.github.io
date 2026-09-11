#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jinhyun/.conda/envs/torch/bin/python}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.json}"

for phase in prepare sanity gap mediation report; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_causal.py" --phase "${phase}" --config "${CONFIG}" "$@"
done
