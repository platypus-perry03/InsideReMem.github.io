#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jinhyun/prj_ws/jiho/AI/reenact"
PYTHON_BIN="/home/jinhyun/.conda/envs/torch/bin/python"
GPU_ID=1
BATCH_SIZE=4

usage() {
    printf 'Usage: %s [--gpu-id N] [--batch-size N]\n' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

OUTPUT_DIR="$ROOT_DIR/liref_outputs/mgsm_language_robustness"
LOG_DIR="$OUTPUT_DIR/logs"
LOCK_FILE="$OUTPUT_DIR/.run.lock"
mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'Another MGSM language robustness run is active: %s\n' "$LOCK_FILE" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

printf 'MGSM 11-language robustness run\n'
printf '  GPU: %s\n' "$GPU_ID"
printf '  batch size: %s\n' "$BATCH_SIZE"
printf '  output: %s\n' "$OUTPUT_DIR"
printf '  log: %s\n' "$LOG_FILE"

export PYTHONUNBUFFERED=1

{
    printf '[START cache completion] %s\n' "$(date --iso-8601=seconds)"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_mgsm_11lang_cache.py" \
        --models all \
        --device-id "$GPU_ID" \
        --batch-size "$BATCH_SIZE" \
        --skip-existing
    printf '[DONE cache completion] %s\n' "$(date --iso-8601=seconds)"

    printf '[START analysis] %s\n' "$(date --iso-8601=seconds)"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/analyze_mgsm_language_robustness.py" \
        --models all \
        --skip-existing
    printf '[DONE analysis] %s\n' "$(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG_FILE"
