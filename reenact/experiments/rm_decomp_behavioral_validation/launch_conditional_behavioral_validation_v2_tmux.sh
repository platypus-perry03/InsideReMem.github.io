#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/jinhyun/prj_ws/jiho
SOCKET=${WORKSPACE}/.tmux-behavior-v2.sock
SESSION=rm_behavior_v2
PIPELINE=${WORKSPACE}/AI/reenact/experiments/rm_decomp_behavioral_validation/run_conditional_behavioral_validation_v2.sh

if tmux -S "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already exists: ${SESSION}"
  exit 1
fi

tmux -S "${SOCKET}" new-session -d -s "${SESSION}" -n pipeline "bash ${PIPELINE}"
tmux -S "${SOCKET}" list-windows -t "${SESSION}"
