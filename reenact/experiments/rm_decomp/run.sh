#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-sanity}"
CONFIG_PATH="${2:-${SCRIPT_DIR}/config.json}"
PYTHON_BIN="/home/jinhyun/.conda/envs/torch/bin/python"

if [[ "${PHASE}" != "sanity" && "${PHASE}" != "full" ]]; then
  echo "Usage: $0 {sanity|full} [config.json]" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run.py" --stage a --phase "${PHASE}" --config "${CONFIG_PATH}"
