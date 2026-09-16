#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/nemotron35-tinylora-lr-screen}"
CANDIDATES="${CANDIDATES:-lr1e-5:1e-5:1.0,lr1e-4:1e-4:1.0,lr2e-4:2e-4:1.0}"
STEPS="${STEPS:-2}"
SAMPLES="${SAMPLES:-256}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}"
GENERATIONS="${GENERATIONS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-512}"
TRAIN_CONTAINER=nemotron35-tinylora-sweep

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

  local gpu_job
  for gpu_job in \
    nemotron35-tinylora-sweep \
    spark-tinylora-rl \
    nemotron35-tinylora-heldout-eval \
    nemotron35-tinylora-recovered-eval; do
    if container_is_running "$gpu_job"; then
      printf 'Not restoring %s: %s still owns the GPU.\n' \
        "$SERVER_CONTAINER" "$gpu_job" >&2
      return 2
    fi
  done

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

if docker inspect "$TRAIN_CONTAINER" >/dev/null 2>&1; then
  printf 'Refusing: container %s already exists (state: %s).\n' \
    "$TRAIN_CONTAINER" "$(container_state "$TRAIN_CONTAINER")" >&2
  printf 'Inspect it with: docker logs -f %q\n' "$TRAIN_CONTAINER" >&2
  exit 1
fi
for gpu_job in \
  spark-tinylora-rl \
  nemotron35-tinylora-heldout-eval \
  nemotron35-tinylora-recovered-eval; do
  if container_is_running "$gpu_job"; then
    printf 'Refusing: %s still owns the GPU.\n' "$gpu_job" >&2
    exit 1
  fi
done

CANDIDATE_ARGS=()
IFS=',' read -r -a candidate_values <<< "$CANDIDATES"
for candidate in "${candidate_values[@]}"; do
  CANDIDATE_ARGS+=(--candidate "$candidate")
done

if container_is_running "$SERVER_CONTAINER"; then
  server_was_running=1
fi

# Create first, then stop inference and start without attaching.  Signals sent
# to this shell or its log follower therefore cannot be proxied into training.
docker create --gpus all --ipc=host \
  --name "$TRAIN_CONTAINER" \
  --label ai.tinylora.launcher=run_nemotron35_tinylora_sweep.sh \
  --label "ai.tinylora.managed-server=$SERVER_CONTAINER" \
  --label "ai.tinylora.restore-server=$server_was_running" \
  --label "ai.tinylora.output-root=$OUTPUT_ROOT" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -v /home/nimitz/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nimitz/.cache/vllm:/root/.cache/vllm \
  -v /home/nimitz/projects/RLtests:/workspace \
  "$IMAGE" \
  python3 /workspace/train_nemotron35_tinylora_sweep.py \
    --output-root "$OUTPUT_ROOT" \
    --steps "$STEPS" \
    --samples "$SAMPLES" \
    --prompts-per-step "$PROMPTS_PER_STEP" \
    --generations "$GENERATIONS" \
    --max-completion-length "$MAX_COMPLETION_LENGTH" \
    --max-model-length "$MAX_MODEL_LENGTH" \
    "${CANDIDATE_ARGS[@]}" >/dev/null
training_container_created=1

if [[ "$server_was_running" == 1 ]]; then
  server_stop_attempted=1
  docker stop --timeout 60 "$SERVER_CONTAINER" >/dev/null
fi

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
