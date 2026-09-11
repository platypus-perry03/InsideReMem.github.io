#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIREF_DIR="${SCRIPT_DIR}/liref"
MODEL_DIR="${SCRIPT_DIR}/liref_models"
OUTPUT_DIR="${SCRIPT_DIR}/liref_outputs"
HIDDEN_STATE_DIR="${OUTPUT_DIR}/hidden_states"
LOG_DIR="${HIDDEN_STATE_DIR}/logs"
EXECUTED_NOTEBOOK_DIR="${HIDDEN_STATE_DIR}/executed_notebooks"
NOTEBOOK="${LIREF_DIR}/reasoning_representation/LiReFs_storing_hs.ipynb"
PYTHON_BIN="/home/jinhyun/.conda/envs/torch/bin/python"
GPU_ID="${LIREF_GPU_ID:-1}"
DRY_RUN=0

MODELS=(
    "Meta-Llama-3-8B"
    "Meta-Llama-3-8B-Instruct"
    "Mistral-7B-v0.3"
    "Mistral-7B-Instruct-v0.3"
    "gemma-2-9b"
    "gemma-2-9b-it"
    "OLMo-2-1124-7B"
    "OLMo-2-1124-7B-Instruct"
)

usage() {
    printf 'Usage: %s [--gpu-id N] [--dry-run]\n' "$0"
    printf '  --gpu-id N  Physical CUDA device index (default: LIREF_GPU_ID or 1)\n'
    printf '  --dry-run   Validate paths and show what would run without loading models\n'
}

while (($# > 0)); do
    case "$1" in
        --gpu-id)
            if (($# < 2)); then
                printf 'ERROR: --gpu-id requires an integer.\n' >&2
                exit 2
            fi
            GPU_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: GPU ID must be a non-negative integer: %s\n' "$GPU_ID" >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$NOTEBOOK" ]]; then
    printf 'ERROR: notebook not found: %s\n' "$NOTEBOOK" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import ipykernel, nbconvert' 2>/dev/null; then
    printf 'ERROR: ipykernel and nbconvert must be installed in the torch environment.\n' >&2
    exit 1
fi

if ((DRY_RUN == 0)); then
    if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | grep -qx "$GPU_ID"; then
        printf 'ERROR: GPU %s is not reported by nvidia-smi.\n' "$GPU_ID" >&2
        exit 1
    fi
fi

for model_name in "${MODELS[@]}"; do
    if [[ ! -d "${MODEL_DIR}/${model_name}" ]]; then
        printf 'ERROR: model directory not found: %s\n' "${MODEL_DIR}/${model_name}" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR" "$EXECUTED_NOTEBOOK_DIR" "${OUTPUT_DIR}/.ipython" "${OUTPUT_DIR}/.jupyter_runtime"

exec 9>"${HIDDEN_STATE_DIR}/.run_liref_hidden_states.lock"
if ! flock -n 9; then
    printf 'ERROR: another LiReF hidden-state runner is already active.\n' >&2
    exit 1
fi

if ((DRY_RUN == 0)) && [[ "${LIREF_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
    gpu_processes="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$gpu_processes" ]]; then
        printf 'ERROR: GPU %s already has compute processes: %s\n' "$GPU_ID" "$gpu_processes" >&2
        printf 'Choose a free GPU. To override intentionally, set LIREF_ALLOW_BUSY_GPU=1.\n' >&2
        exit 1
    fi
fi

printf 'LiReF hidden-state run\n'
printf '  GPU: %s\n' "$GPU_ID"
printf '  Notebook: %s\n' "$NOTEBOOK"
printf '  Hidden states: %s\n' "$HIDDEN_STATE_DIR"
printf '  Logs: %s\n' "$LOG_DIR"

for model_name in "${MODELS[@]}"; do
    final_cache="${HIDDEN_STATE_DIR}/${model_name}-base_hs_cache_no_cot_all.pt"

    if [[ -s "$final_cache" ]]; then
        printf '[SKIP] %s: final cache already exists.\n' "$model_name"
        continue
    fi

    if ((DRY_RUN == 1)); then
        printf '[RUN ] %s\n' "$model_name"
        continue
    fi

    run_stamp="$(date '+%Y%m%d_%H%M%S')"
    log_file="${LOG_DIR}/${model_name}_${run_stamp}.log"
    executed_notebook_name="${model_name}_LiReFs_storing_hs.executed.ipynb"

    printf '[START] %s at %s\n' "$model_name" "$(date --iso-8601=seconds)" | tee "$log_file"

    set +e
    env \
        "PATH=/home/jinhyun/.conda/envs/torch/bin:${PATH}" \
        "IPYTHONDIR=${OUTPUT_DIR}/.ipython" \
        "JUPYTER_RUNTIME_DIR=${OUTPUT_DIR}/.jupyter_runtime" \
        "LIREF_GPU_ID=${GPU_ID}" \
        "LIREF_MODEL_NAME=${model_name}" \
        "$PYTHON_BIN" -m jupyter nbconvert \
            --to notebook \
            --execute "$NOTEBOOK" \
            --ExecutePreprocessor.kernel_name=python3 \
            --ExecutePreprocessor.timeout=-1 \
            --ExecutePreprocessor.startup_timeout=180 \
            --output "$executed_notebook_name" \
            --output-dir "$EXECUTED_NOTEBOOK_DIR" \
            2>&1 | tee -a "$log_file"
    run_status=${PIPESTATUS[0]}
    set -e

    if ((run_status != 0)); then
        printf '[FAILED] %s (exit %s). See %s\n' "$model_name" "$run_status" "$log_file" >&2
        exit "$run_status"
    fi

    if [[ ! -s "$final_cache" ]]; then
        printf '[FAILED] %s finished without creating %s\n' "$model_name" "$final_cache" >&2
        exit 1
    fi

    printf '[DONE] %s at %s\n' "$model_name" "$(date --iso-8601=seconds)" | tee -a "$log_file"
done

if ((DRY_RUN == 1)); then
    printf 'Dry run passed; no notebook was executed.\n'
else
    printf 'All eight LiReF hidden-state caches are complete.\n'
fi
