#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jinhyun/.conda/envs/torch/bin/python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_pre_stage_c.py" --config "${SCRIPT_DIR}/config.json"
