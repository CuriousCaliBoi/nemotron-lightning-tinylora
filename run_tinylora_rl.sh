#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-$MODEL}"
ROLLOUT_REVISION="${ROLLOUT_REVISION:-}"
ROLLOUT_SYNC="${ROLLOUT_SYNC:-merged}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/tinylora-from-scratch}"
FACTOR_CACHE="${FACTOR_CACHE:-}"
STEPS="${STEPS:-1}"
SAMPLES="${SAMPLES:-128}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_REVISION="${DATASET_REVISION:-}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-1}"
GENERATIONS="${GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
TIS_MODE="${TIS_MODE:-token_clip}"
TIS_MINIMUM="${TIS_MINIMUM:-0.1}"
TIS_MAXIMUM="${TIS_MAXIMUM:-10.0}"
LOSS_REDUCTION="${LOSS_REDUCTION:-sample_mean}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TINYLORA_RANK="${TINYLORA_RANK:-2}"
PROJECTION_DIM="${PROJECTION_DIM:-1}"
MODULES_PER_GROUP="${MODULES_PER_GROUP:-16}"
NUM_GROUPS="${NUM_GROUPS:-}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
TARGET_LAYER_INDICES="${TARGET_LAYER_INDICES:-}"
PARAMETER_DTYPE="${PARAMETER_DTYPE:-float32}"
GROUPING="${GROUPING:-tiled}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-}"
VLLM_MAMBA_BACKEND="${VLLM_MAMBA_BACKEND:-}"
VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-1024}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SAVE_EVERY="${SAVE_EVERY:-0}"
PROMPT_STYLE="${PROMPT_STYLE:-concise}"
REWARD_MODE="${REWARD_MODE:-flexible}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
PROFILE_SAMPLE_INTERVAL="${PROFILE_SAMPLE_INTERVAL:-0.20}"
SEED="${SEED:-42}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ARGS=(
  --model "$MODEL"
  --rollout-model "$ROLLOUT_MODEL"
  --rollout-sync "$ROLLOUT_SYNC"
  --output-dir "$OUTPUT_DIR"
  --steps "$STEPS"
  --samples "$SAMPLES"
  --dataset-split "$DATASET_SPLIT"
  --prompts-per-step "$PROMPTS_PER_STEP"
  --generations "$GENERATIONS"
  --max-completion-length "$MAX_COMPLETION_LENGTH"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM"
  --ppo-epochs "$PPO_EPOCHS"
  --clip-epsilon "$CLIP_EPSILON"
  --tis-mode "$TIS_MODE"
  --tis-minimum "$TIS_MINIMUM"
  --tis-maximum "$TIS_MAXIMUM"
  --loss-reduction "$LOSS_REDUCTION"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --rank "$TINYLORA_RANK"
  --projection-dim "$PROJECTION_DIM"
  --modules-per-group "$MODULES_PER_GROUP"
  --target-modules "$TARGET_MODULES"
  --parameter-dtype "$PARAMETER_DTYPE"
  --grouping "$GROUPING"
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-lora-rank "$MAX_LORA_RANK"
  --vllm-kv-cache-dtype "$VLLM_KV_CACHE_DTYPE"
  --max-model-length "$MAX_MODEL_LENGTH"
  --save-every "$SAVE_EVERY"
  --prompt-style "$PROMPT_STYLE"
  --reward-mode "$REWARD_MODE"
  --seed "$SEED"
)
if [[ -n "$MODEL_REVISION" ]]; then
  TRAIN_ARGS+=(--model-revision "$MODEL_REVISION")
fi
if [[ -n "$ROLLOUT_REVISION" ]]; then
  TRAIN_ARGS+=(--rollout-revision "$ROLLOUT_REVISION")
fi
if [[ -n "$DATASET_REVISION" ]]; then
  TRAIN_ARGS+=(--dataset-revision "$DATASET_REVISION")
fi
if [[ -n "$FACTOR_CACHE" ]]; then
  TRAIN_ARGS+=(--factor-cache "$FACTOR_CACHE")
fi
if [[ -n "$NUM_GROUPS" ]]; then
  TRAIN_ARGS+=(--num-groups "$NUM_GROUPS")
fi
if [[ -n "$TARGET_LAYER_INDICES" ]]; then
  TRAIN_ARGS+=(--target-layer-indices "$TARGET_LAYER_INDICES")
fi
if [[ -n "$VLLM_MOE_BACKEND" ]]; then
  TRAIN_ARGS+=(--vllm-moe-backend "$VLLM_MOE_BACKEND")
fi
if [[ -n "$VLLM_KV_CACHE_MEMORY_BYTES" ]]; then
  TRAIN_ARGS+=(--vllm-kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES")
fi
if [[ -n "$VLLM_MAMBA_BACKEND" ]]; then
  TRAIN_ARGS+=(--vllm-mamba-backend "$VLLM_MAMBA_BACKEND")
fi
if [[ -n "$VLLM_MAMBA_CACHE_MODE" ]]; then
  TRAIN_ARGS+=(--vllm-mamba-cache-mode "$VLLM_MAMBA_CACHE_MODE")
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  TRAIN_ARGS+=(--gradient-checkpointing)
else
  TRAIN_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "$PROFILE_MEMORY" == "1" ]]; then
  TRAIN_ARGS+=(--profile-memory --profile-sample-interval "$PROFILE_SAMPLE_INTERVAL")
fi

docker run --rm --gpus all --ipc=host \
  --name spark-tinylora-rl \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -v /home/nimitz/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nimitz/.cache/vllm:/root/.cache/vllm \
  -v /home/nimitz/projects/RLtests:/workspace \
  "$IMAGE" \
  python3 /workspace/train_tinylora_rl.py "${TRAIN_ARGS[@]}"
