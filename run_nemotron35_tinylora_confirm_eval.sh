#!/usr/bin/env bash
# Confirm the Lightning TinyLoRA screen on the untouched final 384 training
# rows. The shared evaluator validates and hashes the completed 128-row screen
# before loading the model, then proves the two question sets are disjoint.
set -Eeuo pipefail

REPO="${REPO:-/home/nimitz/projects/RLtests}"
IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

exec env \
  REPO="$REPO" \
  IMAGE="$IMAGE" \
  SERVER_CONTAINER="$SERVER_CONTAINER" \
  SWEEP_CONTAINER=nemotron35-tinylora-sweep \
  EVAL_CONTAINER=nemotron35-tinylora-confirm-eval \
  OUTPUT_ROOT_REL=outputs/nemotron35-tinylora-lr-scale-replay-canary-20260916 \
  EVAL_PROTOCOL=confirmatory-untouched384-after-screen-v1 \
  CANDIDATES=lr1e-5_s1,lr5e-5_s1,lr1e-4_s1,lr1e-4_s0p5 \
  SELECTED_CANDIDATE=lr1e-4_s0p5 \
  'SELECTION_RATIONALE=selected after the 128-row screen: highest trained strict accuracy (99/128), +2/128 versus the mean of the two zero repeats, with conservative 0.5 deployment scale' \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_BATCH_INVARIANT=0 \
  LEARNER_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  LEARNER_REVISION=a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
  MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  MODEL_REVISION=bee7596271d1495f6992ae224aefde4410e816b8 \
  DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866 \
  'TRAIN_SPLIT=train[:-512]' \
  'EVAL_SPLIT=train[-384:]' \
  SAMPLES=384 \
  MAX_TOKENS=1024 \
  MAX_MODEL_LENGTH=1280 \
  PREFLIGHT_ONLY="$PREFLIGHT_ONLY" \
  bash "$REPO/run_nemotron35_tinylora_eval.sh"
