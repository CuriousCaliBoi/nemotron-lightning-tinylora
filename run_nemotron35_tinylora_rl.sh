#!/usr/bin/env bash
set -euo pipefail

# One-GPU DGX Spark run: BF16 Nemotron is the differentiable learner and the
# already-cached NVFP4 checkpoint is the vLLM rollout policy. TinyLoRA updates
# are transferred as native rank-2 LoRA overlays; the quantized base is never
# merged or requantized.

SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
SERVER_WAS_RUNNING=0
TINYLORA_RANK_VALUE="${TINYLORA_RANK:-2}"

if docker inspect -f '{{.State.Running}}' "$SERVER_CONTAINER" 2>/dev/null | grep -qx true; then
  SERVER_WAS_RUNNING=1
  docker stop --timeout 60 "$SERVER_CONTAINER"
fi

restore_server() {
  if [[ "$SERVER_WAS_RUNNING" == "1" ]]; then
    docker start "$SERVER_CONTAINER" >/dev/null
    for _ in $(seq 1 60); do
      if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
        printf '%s\n' "Restored $SERVER_CONTAINER; health check passed."
        return
      fi
      sleep 5
    done
    printf '%s\n' "Restored $SERVER_CONTAINER, but its health check has not passed yet." >&2
  fi
}
trap restore_server EXIT INT TERM

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}" \
ROLLOUT_MODEL="${ROLLOUT_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}" \
ROLLOUT_SYNC="${ROLLOUT_SYNC:-lora}" \
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}" \
TARGET_LAYER_INDICES="${TARGET_LAYER_INDICES:-}" \
TINYLORA_RANK="$TINYLORA_RANK_VALUE" \
PROJECTION_DIM="${PROJECTION_DIM:-1}" \
NUM_GROUPS="${NUM_GROUPS:-13}" \
PARAMETER_DTYPE="${PARAMETER_DTYPE:-float32}" \
STEPS="${STEPS:-1}" \
SAMPLES="${SAMPLES:-64}" \
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}" \
GENERATIONS="${GENERATIONS:-8}" \
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}" \
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-512}" \
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
TEMPERATURE="${TEMPERATURE:-1.2}" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.23}" \
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-marlin}" \
VLLM_MAMBA_BACKEND="${VLLM_MAMBA_BACKEND:-flashinfer}" \
VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-align}" \
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}" \
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/nemotron35-nvfp4-tinylora-smoke}" \
FACTOR_CACHE="${FACTOR_CACHE:-/workspace/cache/nemotron35-lightning-attention-r${TINYLORA_RANK_VALUE}-svd.safetensors}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
./run_tinylora_rl.sh
