#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-prepare}"
CONFIG_PATH="${2:-${SCRIPT_DIR}/config.json}"
PYTHON_BIN="/home/jinhyun/.conda/envs/torch/bin/python"

case "${PHASE}" in
  prepare|sanity|pilot|freeze_confirmatory|confirmatory|report) ;;
  *)
    echo "Usage: $0 {prepare|sanity|pilot|freeze_confirmatory|confirmatory|report} [config.json]" >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_extension.py" --phase "${PHASE}" --config "${CONFIG_PATH}"
