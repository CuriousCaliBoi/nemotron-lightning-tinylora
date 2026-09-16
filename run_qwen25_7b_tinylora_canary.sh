#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/nimitz/projects/RLtests}"
IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
MODEL_REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
DATASET_REVISION="${DATASET_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
FACTOR_CACHE="${FACTOR_CACHE:-/workspace/cache/Qwen--Qwen2.5-7B-Instruct-r2-svd.safetensors}"
EXPECTED_FACTOR_SHA256="${EXPECTED_FACTOR_SHA256:-ab49b94216b6d2d46760ea44620766afc30417ee5ed626af620a182cd86344d7}"
STEPS="${STEPS:-64}"
NUM_GROUPS="${NUM_GROUPS:-}"
TARGET_LAYER_INDICES="${TARGET_LAYER_INDICES:-}"
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-}"
VLLM_MAMBA_BACKEND="${VLLM_MAMBA_BACKEND:-}"
VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
PROFILE_SAMPLE_INTERVAL="${PROFILE_SAMPLE_INTERVAL:-0.20}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
TRAIN_CONTAINER=spark-tinylora-rl

container_state() {
  local state
  if state="$(docker inspect -f '{{if .State.Running}}running{{else}}stopped{{end}}' "$1" 2>/dev/null)"; then
    printf '%s\n' "$state"
  else
    printf '%s\n' unknown
  fi
}

container_is_running() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true
}

container_uses_gpu() {
  local requests
  requests="$(docker inspect -f '{{json .HostConfig.DeviceRequests}}' "$1")" || return 1
  jq -e '
    (. // []) |
    any(.[];
      .Driver == "nvidia" or
      (((.Capabilities // []) | flatten | index("gpu")) != null)
    )
  ' <<<"$requests" >/dev/null
}

assert_no_competing_gpu_owners() {
  local allowed_container="${1:-}"
  local allowed_id=""
  local name rows pid process_name cgroup
  local -a running_containers=()

  if [[ -n "$allowed_container" ]] && container_is_running "$allowed_container"; then
    allowed_id="$(docker inspect -f '{{.Id}}' "$allowed_container")"
  fi
  mapfile -t running_containers < <(docker ps --format '{{.Names}}')
  for name in "${running_containers[@]}"; do
    [[ -n "$allowed_container" && "$name" == "$allowed_container" ]] && continue
    if container_uses_gpu "$name"; then
      printf 'Refusing: GPU container %s is running concurrently.\n' "$name" >&2
      return 1
    fi
  done

  command -v nvidia-smi >/dev/null 2>&1 || {
    printf 'Refusing: nvidia-smi is required to verify exclusive GPU ownership.\n' >&2
    return 1
  }
  rows="$(nvidia-smi \
    --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits 2>/dev/null)" || {
    printf 'Refusing: could not query active GPU compute processes.\n' >&2
    return 1
  }
  while IFS=',' read -r pid process_name; do
    pid="${pid//[[:space:]]/}"
    [[ -n "$pid" ]] || continue
    [[ -r "/proc/$pid/cgroup" ]] || {
      printf 'Refusing: could not identify GPU process %s.\n' "$pid" >&2
      return 1
    }
    cgroup="$(<"/proc/$pid/cgroup")"
    if [[ -n "$allowed_id" && "$cgroup" == *"$allowed_id"* ]]; then
      continue
    fi
    process_name="${process_name#"${process_name%%[![:space:]]*}"}"
    printf 'Refusing: GPU process %s (%s) is running concurrently.\n' \
      "$pid" "$process_name" >&2
    return 1
  done <<<"$rows"
}

print_recovery() {
  local state="$1"
  printf 'Training supervisor ended; container %s is %s and was left untouched.\n' \
    "$TRAIN_CONTAINER" "$state" >&2
  printf 'Inspect it with: docker inspect -f '\''{{.State.Status}} exit={{.State.ExitCode}}'\'' %q\n' \
    "$TRAIN_CONTAINER" >&2
  printf 'Follow its logs with: docker logs -f %q\n' "$TRAIN_CONTAINER" >&2
  printf 'Do not start %s until Docker reports that training is no longer running.\n' \
    "$SERVER_CONTAINER" >&2
  printf 'Afterward, remove the retained container with: docker rm %q\n' \
    "$TRAIN_CONTAINER" >&2
}

server_was_running=0
server_stop_attempted=0
training_container_created=0
training_may_be_running=0
normal_completion=0
interrupted_signal=
log_follower_pid=

stop_log_follower() {
  if [[ -n "$log_follower_pid" ]] && kill -0 "$log_follower_pid" 2>/dev/null; then
    kill -TERM "$log_follower_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$log_follower_pid" 2>/dev/null || break
      sleep 0.05
    done
    if kill -0 "$log_follower_pid" 2>/dev/null; then
      kill -KILL "$log_follower_pid" 2>/dev/null || true
    fi
    wait "$log_follower_pid" 2>/dev/null || true
  fi
  log_follower_pid=
}

restore_server_if_safe() {
  local state="$1"
  [[ "$server_was_running" == 1 && "$server_stop_attempted" == 1 ]] || return 0

  if [[ "$training_may_be_running" == 1 && "$state" != stopped ]]; then
    printf 'Not restoring %s: training state is %s.\n' \
      "$SERVER_CONTAINER" "$state" >&2
    return 2
  fi

  if ! assert_no_competing_gpu_owners "$SERVER_CONTAINER"; then
    printf 'Not restoring %s while another process owns the GPU.\n' \
      "$SERVER_CONTAINER" >&2
    return 2
  fi

  if ! container_is_running "$SERVER_CONTAINER"; then
    docker start "$SERVER_CONTAINER" >/dev/null || return 1
  fi
  local healthy=0
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
      healthy=1
      break
    fi
    sleep 5
  done
  if [[ "$healthy" != 1 ]]; then
    printf 'Restored %s, but its health check has not passed yet.\n' \
      "$SERVER_CONTAINER" >&2
    return 1
  fi
  printf 'Restored %s; health check passed.\n' "$SERVER_CONTAINER"
}

finalize() {
  local rc=$?
  trap - EXIT INT TERM HUP
  stop_log_follower
  local state
  state="$(container_state "$TRAIN_CONTAINER")"

  if ! restore_server_if_safe "$state"; then
    [[ "$rc" != 0 ]] || rc=1
  fi

  if [[ "$normal_completion" == 1 ]]; then
    if ! docker rm "$TRAIN_CONTAINER" >/dev/null; then
      printf 'Warning: could not remove stopped container %s.\n' \
        "$TRAIN_CONTAINER" >&2
      [[ "$rc" != 0 ]] || rc=1
    fi
  elif [[ "$training_container_created" == 1 &&
    "$training_may_be_running" == 0 && "$state" == stopped ]]; then
    if ! docker rm "$TRAIN_CONTAINER" >/dev/null; then
      printf 'Warning: could not remove never-started container %s.\n' \
        "$TRAIN_CONTAINER" >&2
      [[ "$rc" != 0 ]] || rc=1
    fi
  elif [[ "$training_container_created" == 1 ||
    "$training_may_be_running" == 1 || -n "$interrupted_signal" ]]; then
    print_recovery "$state"
  fi
  exit "$rc"
}

handle_signal() {
  interrupted_signal="$1"
  exit "$2"
}

trap finalize EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

cd "$REPO"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || {
  printf 'Refusing: PREFLIGHT_ONLY must be 0 or 1.\n' >&2
  exit 1
}
if docker inspect "$TRAIN_CONTAINER" >/dev/null 2>&1; then
  printf 'Refusing: container %s already exists (state: %s).\n' \
    "$TRAIN_CONTAINER" "$(container_state "$TRAIN_CONTAINER")" >&2
  printf 'Inspect it with: docker logs -f %q\n' "$TRAIN_CONTAINER" >&2
  exit 1
fi
for gpu_job in \
  nemotron35-tinylora-sweep \
  nemotron35-tinylora-heldout-eval \
  nemotron35-tinylora-recovered-eval; do
  if container_is_running "$gpu_job"; then
    printf 'Refusing: %s still owns the GPU.\n' "$gpu_job" >&2
    exit 1
  fi
done
assert_no_competing_gpu_owners "$SERVER_CONTAINER" || exit 1

factor_sha="$({
  docker run --rm --entrypoint sha256sum \
    -v "$REPO:/workspace:ro" \
    "$IMAGE" "$FACTOR_CACHE"
} | awk '{print $1}')"
[[ "$factor_sha" == "$EXPECTED_FACTOR_SHA256" ]] || {
  printf 'Factor-cache hash mismatch: %s\n' "$factor_sha" >&2
  exit 1
}

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_SLUG="${RUN_SLUG:-qwen2.5-7b-tinylora13-verl-strict-canary-s42-$run_stamp}"
[[ "$RUN_SLUG" =~ ^[A-Za-z0-9._-]+$ && "$RUN_SLUG" != . && "$RUN_SLUG" != .. ]] || {
  printf 'Refusing: RUN_SLUG must be one safe path segment.\n' >&2
  exit 1
}
[[ ! -e "$REPO/outputs/$RUN_SLUG" ]] || {
  printf 'Output already exists: %s\n' "$REPO/outputs/$RUN_SLUG" >&2
  exit 1
}

if container_is_running "$SERVER_CONTAINER"; then
  server_was_running=1
fi

# These variables were inherited by run_tinylora_rl.sh in the original
# launcher.  Preserve that optional interface in the explicit detached command.
OPTIONAL_TRAIN_ARGS=()
if [[ -n "$NUM_GROUPS" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--num-groups "$NUM_GROUPS")
fi
if [[ -n "$TARGET_LAYER_INDICES" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--target-layer-indices "$TARGET_LAYER_INDICES")
fi
if [[ -n "$VLLM_MOE_BACKEND" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--vllm-moe-backend "$VLLM_MOE_BACKEND")
fi
if [[ -n "$VLLM_MAMBA_BACKEND" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--vllm-mamba-backend "$VLLM_MAMBA_BACKEND")
fi
if [[ -n "$VLLM_MAMBA_CACHE_MODE" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--vllm-mamba-cache-mode "$VLLM_MAMBA_CACHE_MODE")
fi
if [[ "$PROFILE_MEMORY" == 1 ]]; then
  OPTIONAL_TRAIN_ARGS+=(--profile-memory --profile-sample-interval "$PROFILE_SAMPLE_INTERVAL")
fi

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  printf 'Preflight passed; GPU training was not launched.\n'
  printf 'Output will be: %s\n' "$REPO/outputs/$RUN_SLUG"
  printf 'Factor cache SHA-256: %s\n' "$factor_sha"
  exit 0
fi

# Keep this command explicit instead of delegating to run_tinylora_rl.sh: the
# generic wrapper uses an attached `docker run --rm`, which would reintroduce
# terminal signal forwarding.  These are the same canary arguments and env.
docker create --gpus all --ipc=host \
  --name "$TRAIN_CONTAINER" \
  --label ai.tinylora.launcher=run_qwen25_7b_tinylora_canary.sh \
  --label "ai.tinylora.managed-server=$SERVER_CONTAINER" \
  --label "ai.tinylora.restore-server=$server_was_running" \
  --label "ai.tinylora.output-dir=/workspace/outputs/$RUN_SLUG" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /home/nimitz/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nimitz/.cache/vllm:/root/.cache/vllm \
  -v "$REPO:/workspace" \
  "$IMAGE" \
  python3 /workspace/train_tinylora_rl.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model-revision "$MODEL_REVISION" \
    --rollout-model Qwen/Qwen2.5-7B-Instruct \
    --rollout-revision "$MODEL_REVISION" \
    --rollout-sync lora \
    --output-dir "/workspace/outputs/$RUN_SLUG" \
    --factor-cache "$FACTOR_CACHE" \
    --factor-cache-sha256 "$factor_sha" \
    --steps "$STEPS" \
    --samples 512 \
    --dataset-split 'train[:-512]' \
    --dataset-revision "$DATASET_REVISION" \
    --prompts-per-step 16 \
    --generations 4 \
    --max-completion-length 512 \
    --micro-batch-size 2 \
    --learning-rate 1e-4 \
    --weight-decay 0 \
    --max-grad-norm 1 \
    --ppo-epochs 1 \
    --clip-epsilon 0.2 \
    --tis-mode token_clip \
    --tis-minimum 0.1 \
    --tis-maximum 10 \
    --loss-reduction token_mean \
    --temperature 1 \
    --top-p 1 \
    --rank 2 \
    --projection-dim 1 \
    --modules-per-group 16 \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --parameter-dtype bfloat16 \
    --grouping tiled \
    --vllm-gpu-memory-utilization 0.35 \
    --vllm-kv-cache-memory-bytes 2147483648 \
    --max-lora-rank 8 \
    --vllm-kv-cache-dtype auto \
    --max-model-length 1024 \
    --gradient-checkpointing \
    --save-every 16 \
    --prompt-style verl \
    --reward-mode strict \
    --seed 42 \
    "${OPTIONAL_TRAIN_ARGS[@]}" >/dev/null
training_container_created=1

printf 'Training output: %s\n' "$REPO/outputs/$RUN_SLUG"
if [[ "$server_was_running" == 1 ]]; then
  server_stop_attempted=1
  docker stop --timeout 60 "$SERVER_CONTAINER" >/dev/null
fi

# Re-check after the managed server has stopped.  This closes the window
# between the initial ownership check and handing the GPU to the trainer.
assert_no_competing_gpu_owners || exit 1

training_may_be_running=1
docker start "$TRAIN_CONTAINER" >/dev/null
printf 'Training is detached in container %s. Follow-up commands are safe if this shell disconnects.\n' \
  "$TRAIN_CONTAINER"

# `docker logs` is only a follower.  Run it asynchronously so a signal sent to
# just the supervisor (rather than its process group) can interrupt `wait`.
# Stopping this client never forwards a signal to the detached container.
docker logs -f "$TRAIN_CONTAINER" &
log_follower_pid=$!
set +e
wait "$log_follower_pid"
logs_rc=$?
set -e
log_follower_pid=

state="$(container_state "$TRAIN_CONTAINER")"
if [[ "$state" != stopped ]]; then
  printf 'Log follower exited with status %s while training state is %s.\n' \
    "$logs_rc" "$state" >&2
  [[ "$logs_rc" != 0 ]] || logs_rc=1
  exit "$logs_rc"
fi
training_exit_code="$(docker inspect -f '{{.State.ExitCode}}' "$TRAIN_CONTAINER")"
printf 'Training container %s exited with status %s.\n' \
  "$TRAIN_CONTAINER" "$training_exit_code"
normal_completion=1
exit "$training_exit_code"
