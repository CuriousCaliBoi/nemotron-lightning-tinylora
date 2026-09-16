#!/usr/bin/env bash
# Run the continual loop in the research image with a colocated vLLM actor,
# the same way the RLtests training scripts run: the Lightning serving
# container is stopped for the duration and restored on exit.
set -Eeuo pipefail

SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
STATE_ROOT="${STATE_ROOT:-/home/nimitz/cl-state}"
STATE_DIR="${STATE_DIR:-$STATE_ROOT/nemotron35-gsm8k}"
LOG="${LOG:-$STATE_DIR/loop.log}"
SERVER_WAS_RUNNING=0

mkdir -p "$STATE_DIR"
rm -f "$STATE_DIR/STOP"

if docker inspect -f '{{.State.Running}}' "$SERVER_CONTAINER" 2>/dev/null | grep -qx true; then
  SERVER_WAS_RUNNING=1
  echo "[runner] stopping $SERVER_CONTAINER"
  docker stop --timeout 60 "$SERVER_CONTAINER" >/dev/null
fi

restore_server() {
  if [[ "$SERVER_WAS_RUNNING" == "1" ]]; then
    echo "[runner] restoring $SERVER_CONTAINER"
    docker start "$SERVER_CONTAINER" >/dev/null || true
    for _ in $(seq 1 120); do
      if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
        echo "[runner] restored $SERVER_CONTAINER; health check passed"
        return
      fi
      sleep 5
    done
    echo "[runner] restored $SERVER_CONTAINER, but its health check has not passed yet" >&2
  fi
}
trap restore_server EXIT INT TERM

sleep 3
remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c . || true)
if [[ "$remaining" != "0" ]]; then
  echo "[runner] refusing: $remaining compute process(es) still hold the GPU" >&2
  exit 1
fi

docker run --rm --gpus all --ipc=host \
  --name "${LOOP_CONTAINER:-cl-loop}" \
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ROLLOUT_BACKEND=colocated -e STATE_DIR="$STATE_DIR" \
  -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.23}" \
  -e PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}" -e GENERATIONS="${GENERATIONS:-8}" \
  -e MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}" -e TEMPERATURE="${TEMPERATURE:-1.0}" \
  -e LEARNING_RATE="${LEARNING_RATE:-1e-4}" -e MICRO_BATCH="${MICRO_BATCH:-1}" \
  -e PROMPT_STYLE="${PROMPT_STYLE:-concise}" -e REWARD_MODE="${REWARD_MODE:-strict}" \
  -e EVAL_EVERY="${EVAL_EVERY:-5}" -e EVAL_ROWS="${EVAL_ROWS:-128}" -e EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-512}" \
  -e ROLLBACK_DROP="${ROLLBACK_DROP:-0.10}" -e MAX_STEPS="${MAX_STEPS:-40}" -e SEED="${SEED:-42}" \
  -v /home/nimitz/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nimitz/.cache/vllm:/root/.cache/vllm \
  -v /home/nimitz/projects/RLtests:/workspace \
  -v "$STATE_ROOT:$STATE_ROOT" \
  -w /workspace \
  "$IMAGE" python3 -u -m continual.loop 2>&1 | tee -a "$LOG"
echo "[runner] loop finished with exit ${PIPESTATUS[0]}"
