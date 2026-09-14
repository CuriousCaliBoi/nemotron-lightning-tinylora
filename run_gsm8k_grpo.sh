#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-spark-vllm-trl:0.1}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_STEPS="${MAX_STEPS:-4}"
SAMPLES="${SAMPLES:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/gsm8k-grpo-lora}"

docker run --rm --gpus all --ipc=host \
  --name spark-gsm8k-grpo \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /home/nimitz/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nimitz/projects/RLtests:/workspace \
  "$IMAGE" \
  python3 /workspace/train_gsm8k_grpo.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --samples "$SAMPLES"
