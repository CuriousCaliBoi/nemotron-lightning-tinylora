#!/usr/bin/env bash
# Evaluate a normally completed four-candidate Nemotron 3.5 TinyLoRA sweep.
# A zero LoRA is evaluated twice under distinct request IDs so
# zero-vs-zero_repeat measures inference repeatability on the same path.
set -Eeuo pipefail

REPO="${REPO:-/home/nimitz/projects/RLtests}"
IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
SWEEP_CONTAINER="${SWEEP_CONTAINER:-nemotron35-tinylora-sweep}"
SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
EVAL_CONTAINER="${EVAL_CONTAINER:-nemotron35-tinylora-heldout-eval}"
OUTPUT_ROOT_REL="${OUTPUT_ROOT_REL:-outputs/nemotron35-tinylora-lr-scale-replay-canary-20260916}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-heldout512-v1}"
CANDIDATES="${CANDIDATES:-lr1e-5_s1,lr5e-5_s1,lr1e-4_s1,lr1e-4_s0p5}"
LEARNER_MODEL="${LEARNER_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
LEARNER_REVISION="${LEARNER_REVISION:-a9904d24bcc1d289a1950fa9d2b978c47cf903b9}"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-bee7596271d1495f6992ae224aefde4410e816b8}"
DATASET_REVISION="${DATASET_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train[:-512]}"
EVAL_SPLIT="${EVAL_SPLIT:-train[-512:]}"
SAMPLES="${SAMPLES:-512}"
MAX_TOKENS="${MAX_TOKENS:-768}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-1024}"
SCREEN_EVALUATION_REL="${SCREEN_EVALUATION_REL:-$OUTPUT_ROOT_REL/screen-consumed128-max1024.json}"
SELECTED_CANDIDATE="${SELECTED_CANDIDATE:-lr1e-4_s0p5}"
EXPECTED_SELECTION_RATIONALE='selected after the 128-row screen: highest trained strict accuracy (99/128), +2/128 versus the mean of the two zero repeats, with conservative 0.5 deployment scale'
SELECTION_RATIONALE="${SELECTION_RATIONALE:-$EXPECTED_SELECTION_RATIONALE}"
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-}"
VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

host_root="$REPO/$OUTPUT_ROOT_REL"
container_root="/workspace/$OUTPUT_ROOT_REL"
output=""
zero_host="$host_root/zero_control"
zero_container="$container_root/zero_control"
screen_host=""
screen_container=""
disjoint_args=()
docker_environment_args=()
evaluator_environment_args=()

refuse() {
  printf 'Refusing: %s\n' "$*" >&2
  exit 1
}

container_is_running() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

require_file() {
  [[ -s "$1" ]] || refuse "missing or empty file $1"
}

container_uses_gpu() {
  local requests
  requests=$(docker inspect -f '{{json .HostConfig.DeviceRequests}}' "$1") || return 1
  jq -e '
    (. // []) |
    any(.[];
      .Driver == "nvidia" or
      (((.Capabilities // []) | flatten | index("gpu")) != null)
    )
  ' <<<"$requests" >/dev/null
}

# The serving container is the only GPU owner allowed before evaluation: this
# script stops and restores it. Checking Docker reservations and compute PIDs
# catches both idle competing containers and host-side GPU jobs.
assert_no_competing_gpu_owners() {
  local allowed_container=${1:-}
  local allowed_id=""
  local id name rows pid process_name cgroup
  local -a running_containers=()

  if [[ -n "$allowed_container" ]] && container_is_running "$allowed_container"; then
    allowed_id=$(docker inspect -f '{{.Id}}' "$allowed_container")
  fi

  mapfile -t running_containers < <(docker ps -q)
  for id in "${running_containers[@]}"; do
    container_uses_gpu "$id" || continue
    name=$(docker inspect -f '{{.Name}}' "$id")
    name=${name#/}
    [[ -n "$allowed_id" && "$id" == "$allowed_id" ]] && continue
    refuse "GPU container $name is running concurrently"
  done

  command -v nvidia-smi >/dev/null 2>&1 ||
    refuse 'nvidia-smi is required to verify exclusive GPU ownership'
  rows=$(nvidia-smi \
    --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits 2>/dev/null) ||
    refuse 'could not query active GPU compute processes'
  while IFS=',' read -r pid process_name; do
    pid=${pid//[[:space:]]/}
    [[ -n "$pid" ]] || continue
    [[ -r "/proc/$pid/cgroup" ]] || refuse "could not identify GPU process $pid"
    cgroup=$(<"/proc/$pid/cgroup")
    if [[ -n "$allowed_id" && "$cgroup" == *"$allowed_id"* ]]; then
      continue
    fi
    process_name=${process_name#"${process_name%%[![:space:]]*}"}
    refuse "GPU process $pid ($process_name) is running concurrently"
  done <<<"$rows"
}

require_complete_peft_or_absent() {
  local path=$1
  local description=$2
  if [[ -e "$path" ]]; then
    [[ -s "$path/adapter_config.json" && -s "$path/adapter_model.safetensors" ]] ||
      refuse "$description is partial"
    return 0
  fi
  return 1
}

cd "$REPO"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] ||
  refuse 'PREFLIGHT_ONLY must be 0 or 1'
is_positive_integer "$SAMPLES" || refuse 'SAMPLES must be a positive integer'
is_positive_integer "$MAX_TOKENS" || refuse 'MAX_TOKENS must be a positive integer'
is_positive_integer "$MAX_MODEL_LENGTH" ||
  refuse 'MAX_MODEL_LENGTH must be a positive integer'
(( MAX_MODEL_LENGTH > MAX_TOKENS )) ||
  refuse 'MAX_MODEL_LENGTH must be greater than MAX_TOKENS'
[[ "$OUTPUT_ROOT_REL" != /* && "/$OUTPUT_ROOT_REL/" != *'/../'* ]] ||
  refuse 'OUTPUT_ROOT_REL must be relative and may not contain parent traversal'
[[ -d "$host_root" ]] || refuse "missing sweep output root $host_root"

case "$EVAL_PROTOCOL" in
  heldout512-v1)
    [[ "$SAMPLES" == 512 && "$EVAL_SPLIT" == 'train[-512:]' ]] ||
      refuse 'heldout512-v1 requires SAMPLES=512 and EVAL_SPLIT=train[-512:]'
    output="$host_root/heldout-gsm8k-512-canonical-zero-control.json"
    ;;
  confirmatory-untouched384-after-screen-v1)
    [[ "$SAMPLES" == 384 && "$EVAL_SPLIT" == 'train[-384:]' ]] ||
      refuse 'the confirmatory protocol requires SAMPLES=384 and EVAL_SPLIT=train[-384:]'
    [[ "$MAX_TOKENS" == 1024 && "$MAX_MODEL_LENGTH" == 1280 ]] ||
      refuse 'the confirmatory protocol requires MAX_TOKENS=1024 and MAX_MODEL_LENGTH=1280'
    [[ "$SCREEN_EVALUATION_REL" == "$OUTPUT_ROOT_REL/screen-consumed128-max1024.json" ]] ||
      refuse 'the confirmatory protocol must bind the fixed 128-row screen artifact'
    [[ "$SELECTED_CANDIDATE" == lr1e-4_s0p5 ]] ||
      refuse 'the confirmatory protocol predeclares lr1e-4_s0p5 only'
    [[ "$SELECTION_RATIONALE" == "$EXPECTED_SELECTION_RATIONALE" ]] ||
      refuse 'the confirmatory selection rationale is fixed by the completed screen'
    [[ "$VLLM_ENABLE_V1_MULTIPROCESSING" == 0 ]] ||
      refuse 'the confirmatory protocol requires VLLM_ENABLE_V1_MULTIPROCESSING=0'
    [[ "$VLLM_BATCH_INVARIANT" == 0 ]] ||
      refuse 'the Lightning Mamba backend requires VLLM_BATCH_INVARIANT=0'
    screen_host="$REPO/$SCREEN_EVALUATION_REL"
    screen_container="/workspace/$SCREEN_EVALUATION_REL"
    disjoint_args=(--disjoint-evaluation "selection_screen=$screen_container")
    docker_environment_args=(
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0
      -e VLLM_BATCH_INVARIANT=0
    )
    evaluator_environment_args=(
      --require-environment VLLM_ENABLE_V1_MULTIPROCESSING=0
      --require-environment VLLM_BATCH_INVARIANT=0
    )
    output="$host_root/confirmatory-untouched384-lr1e-4-s0p5-abba-bf16.json"
    ;;
  *)
    refuse "unknown EVAL_PROTOCOL $EVAL_PROTOCOL"
    ;;
esac

IFS=',' read -r -a labels <<<"$CANDIDATES"
[[ "${#labels[@]}" == 4 ]] || refuse 'CANDIDATES must name exactly four candidates'
declare -A seen_labels=()
for label in "${labels[@]}"; do
  [[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || refuse "unsafe candidate label $label"
  [[ -z "${seen_labels[$label]:-}" ]] || refuse "duplicate candidate label $label"
  seen_labels[$label]=1
done
candidate_labels_json=$(jq -cn --arg labels "$CANDIDATES" '$labels | split(",")')

if [[ "$EVAL_PROTOCOL" == confirmatory-untouched384-after-screen-v1 ]]; then
  require_file "$screen_host"
  jq -e \
    --arg actor "$MODEL" \
    --arg actor_revision "$MODEL_REVISION" \
    --arg dataset_revision "$DATASET_REVISION" \
    --arg selected "$SELECTED_CANDIDATE" \
    --arg selected_path "$container_root/$SELECTED_CANDIDATE/peft_adapter" \
    --arg zero_path "$zero_container" \
    --argjson candidates "$candidate_labels_json" '
      . as $root |
      .schema_version >= 3 and
      .metric == "gsm8k_greedy_exact_match" and
      .model == $actor and
      .requested_revision == $actor_revision and
      .dataset == "openai/gsm8k" and
      .dataset_config == "main" and
      .dataset_revision == $dataset_revision and
      .split == "train[-512:-384]" and
      .source_dataset_rows == 128 and
      .selected_dataset_rows == 128 and
      .require_exact_samples == true and
      .shuffle_seed == null and
      .prompt_style == "concise" and
      .score_mode == "strict" and
      .score_implementations.strict == "verl_gsm8k_strict_last_300_chars_v1" and
      .temperature == 0 and .top_p == 1 and .seed == 42 and
      .bootstrap_samples == 10000 and .bootstrap_seed == 42 and
      .max_tokens == 1024 and .max_model_length == 1280 and
      .comparison_baseline == "zero" and
      .holm_family_labels == $candidates and
      (($root.candidates | keys | sort) ==
        (($candidates + ["zero", "zero_repeat"]) | sort)) and
      (($root.comparisons | keys | sort) ==
        (($candidates + ["base", "zero_repeat"]) | sort)) and
      ([ $candidates[] as $label | select(
        $root.comparisons[$label].holm_family_member != true or
        $root.comparisons[$label].multiple_comparison_family_size != 4 or
        ($root.comparisons[$label].mcnemar_holm_p | type) != "number"
      )] | length == 0) and
      (["base", "zero_repeat"] | all(.[]; . as $label |
        $root.comparisons[$label].holm_family_member == false and
        ($root.comparisons[$label] | has("mcnemar_holm_p") | not)
      )) and
      ($root.question_sha256 | length) == 128 and
      ($root.question_sha256 | unique | length) == 128 and
      ($root.answer_sha256 | length) == 128 and
      $root.base.samples == 128 and
      ([$root.candidates[] | select(.samples != 128)] | length == 0) and
      $root.candidates[$selected].adapter == $selected_path and
      $root.candidates.zero.adapter == $zero_path and
      $root.candidates.zero_repeat.adapter == $zero_path and
      $root.candidates.zero.adapter_artifacts ==
        $root.candidates.zero_repeat.adapter_artifacts and
      $root.candidates[$selected].strict_correct == 99 and
      $root.candidates[$selected].flexible_correct == 115 and
      (2 * $root.candidates[$selected].strict_correct -
        $root.candidates.zero.strict_correct -
        $root.candidates.zero_repeat.strict_correct) == 4 and
      ([$candidates[] as $label | $root.candidates[$label].strict_correct] | max) == 99 and
      ([$candidates[] as $label | select(
        $root.candidates[$label].strict_correct == 99
      )] | length) == 1 and
      ($root.software_versions | type) == "object"
    ' "$screen_host" >/dev/null ||
    refuse 'selection screen is incomplete or does not match the preregistered protocol'
  screen_sha256=$(sha256sum "$screen_host" | awk '{print $1}')
  printf 'Verified selection screen sha256:%s (%s)\n' \
    "$screen_sha256" "$SELECTION_RATIONALE"
fi

container_is_running "$SWEEP_CONTAINER" && refuse "$SWEEP_CONTAINER is still running"
container_is_running spark-tinylora-rl && refuse 'spark-tinylora-rl is still running'
container_is_running "$EVAL_CONTAINER" && refuse "$EVAL_CONTAINER is already running"
container_is_running nemotron35-tinylora-recovered-eval &&
  refuse 'the recovered held-out evaluation is already running'
if pgrep -f '[r]un_nemotron35_tinylora_sweep\.sh' >/dev/null 2>&1; then
  refuse 'the sweep wrapper is still active (serving-container restore may be pending)'
fi
assert_no_competing_gpu_owners "$SERVER_CONTAINER"

require_file "$host_root/sweep_status.json"
require_file "$host_root/sweep_manifest.json"
require_file "$host_root/sweep_results.json"
[[ ! -e "$output" ]] || refuse "evaluation output already exists: $output"

# Bind evaluation to the exact model/dataset identities, held-out design,
# canonical scorer, and exact-first-step replay protocol.
jq -e \
  --arg learner "$LEARNER_MODEL" \
  --arg learner_revision "$LEARNER_REVISION" \
  --arg actor "$MODEL" \
  --arg actor_revision "$MODEL_REVISION" \
  --arg dataset_revision "$DATASET_REVISION" \
  --arg train_split "$TRAIN_SPLIT" \
  --argjson candidates "$candidate_labels_json" '
    .model == $learner and
    .model_revision == $learner_revision and
    .rollout_model == $actor and
    .rollout_revision == $actor_revision and
    .pairing.protocol == "exact_first_step_token_text_logprob_replay_v1" and
    .pairing.later_steps_replayed == false and
    .common.steps == 2 and
    .common.train_split == $train_split and
    .common.dataset_revision == $dataset_revision and
    .common.prompt_style == "concise" and
    .common.reward_mode == "strict" and
    .common.seed == 42 and
    ([.candidates[].label] == $candidates) and
    ([.candidates[] | select(
      (.learning_rate | type) != "number" or .learning_rate <= 0 or
      (.scaling | type) != "number" or .scaling <= 0
    )] | length == 0)
  ' "$host_root/sweep_manifest.json" >/dev/null ||
  refuse 'sweep_manifest.json does not match the canonical replay-sweep protocol'

jq -e --argjson candidates "$candidate_labels_json" '
  .status == "complete" and
  (.results | type == "object") and
  ((.results | keys | sort) == ($candidates | sort))
' "$host_root/sweep_status.json" >/dev/null ||
  refuse 'sweep_status.json is not a complete four-candidate result'

jq -e --argjson candidates "$candidate_labels_json" '
  (type == "object") and
  ((keys | sort) == ($candidates | sort)) and
  ([.[] | select(
    .last_training_metrics.step != 2 or
    .first_step_trajectory_matches_reference != true or
    (.first_step_request_sha256 | type) != "string" or
    (.first_step_request_sha256 | length) != 64
  )] | length == 0) and
  ([to_entries[] | select(
    (.key == $candidates[0] and .value.first_step_rollout_replayed != false) or
    (.key != $candidates[0] and .value.first_step_rollout_replayed != true)
  )] | length == 0) and
  ([.[].first_step_request_sha256] | unique | length == 1)
' "$host_root/sweep_results.json" >/dev/null ||
  refuse 'sweep_results.json does not prove exact first-step replay for all candidates'

jq -e -s '.[0].results == .[1]' \
  "$host_root/sweep_status.json" "$host_root/sweep_results.json" >/dev/null ||
  refuse 'sweep status/results disagree'

for label in "${labels[@]}"; do
  candidate="$host_root/$label"
  require_file "$candidate/run_manifest.json"
  require_file "$candidate/metrics.jsonl"
  require_file "$candidate/trajectories.jsonl"
  require_file "$candidate/final_adapter/adapter_config.json"
  require_file "$candidate/final_adapter/adapter.safetensors"
  require_file "$candidate/peft_adapter/adapter_config.json"
  require_file "$candidate/peft_adapter/adapter_model.safetensors"

  expected_lr=$(jq -er --arg label "$label" \
    '.candidates[] | select(.label == $label) | .learning_rate' \
    "$host_root/sweep_manifest.json")
  expected_scaling=$(jq -er --arg label "$label" \
    '.candidates[] | select(.label == $label) | .scaling' \
    "$host_root/sweep_manifest.json")

  jq -e \
    --arg learner "$LEARNER_MODEL" \
    --arg learner_revision "$LEARNER_REVISION" \
    --arg actor "$MODEL" \
    --arg actor_revision "$MODEL_REVISION" \
    --arg dataset_revision "$DATASET_REVISION" \
    --arg train_split "$TRAIN_SPLIT" \
    --argjson expected_lr "$expected_lr" \
    --argjson expected_scaling "$expected_scaling" '
      .schema_version >= 3 and
      .learner.model_id == $learner and
      .learner.requested_revision == $learner_revision and
      .rollout.model_id == $actor and
      .rollout.requested_revision == $actor_revision and
      .rollout.sync_mode == "lora" and
      .dataset.id == "openai/gsm8k" and
      .dataset.config == "main" and
      .dataset.split == $train_split and
      .dataset.revision == $dataset_revision and
      .prompt.style == "concise" and
      .reward.mode == "strict" and
      .reward.implementation == "verl_gsm8k_strict_last_300_chars_v1" and
      .sweep.first_step_rollout_protocol == "exact_token_logprob_replay_v1" and
      .train_config.steps == 2 and
      .train_config.learning_rate == $expected_lr and
      .adapter_config.scaling == $expected_scaling and
      .adapter_config.rank == 2 and
      .adapter_config.projection_dim == 1 and
      .adapter_config.num_groups == 13 and
      .trainable_parameters == 13 and
      .target_layers == 24
    ' "$candidate/run_manifest.json" >/dev/null ||
    refuse "$label/run_manifest.json does not match the canonical replay protocol"

  jq -s -e '
    length == 2 and
    (map(.step) == [1, 2]) and
    all(.[];
      (.grad_norm | type) == "number" and .grad_norm >= 0 and
      (.adapter_norm | type) == "number" and .adapter_norm > 0 and
      (.step_seconds | type) == "number" and .step_seconds > 0 and
      .synced_weights == 24
    )
  ' "$candidate/metrics.jsonl" >/dev/null ||
    refuse "$label must contain exactly two valid optimizer-step metric rows"

  jq -s -e '
    length == 64 and
    (sort_by(.step) | group_by(.step) | map({step: .[0].step, rows: length})) ==
      [{"step": 1, "rows": 32}, {"step": 2, "rows": 32}]
  ' "$candidate/trajectories.jsonl" >/dev/null ||
    refuse "$label must contain 32 trajectories for each of two steps"

  jq -e --arg actor "$MODEL" '
    .base_model_name_or_path == $actor and
    .r == 2 and .lora_alpha == 2 and
    ((.target_modules | sort) == (["q_proj", "k_proj", "v_proj", "o_proj"] | sort))
  ' "$candidate/peft_adapter/adapter_config.json" >/dev/null ||
    refuse "$label PEFT adapter is not the expected rank-2 actor adapter"
done

if ! require_complete_peft_or_absent "$zero_host" 'zero-control PEFT adapter'; then
  docker run --rm \
    -v "$REPO:/workspace" \
    -w /workspace \
    "$IMAGE" \
    python3 /workspace/export_lora_adapter.py \
      --adapter "$container_root/${labels[0]}/final_adapter" \
      --output "$zero_container" \
      --base-model "$MODEL" \
      --multiplier 0
fi
require_file "$zero_host/adapter_config.json"
require_file "$zero_host/adapter_model.safetensors"

# Prove that the control reuses the source checkpoint's right factors while
# every materialized trainable/left factor is exactly zero.
docker run --rm -i \
  -v "$REPO:/workspace" \
  -w /workspace \
  "$IMAGE" \
  python3 - \
    "$container_root/${labels[0]}/final_adapter" \
    "$zero_container" \
    "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

import torch

from tinylora_rl.adapter_analysis import load_adapter_updates


native_path, zero_path, expected_base = sys.argv[1:]
config = json.loads((Path(zero_path) / "adapter_config.json").read_text())
if config.get("base_model_name_or_path") != expected_base:
    raise RuntimeError("zero control declares the wrong actor base")
if config.get("r") != 2 or config.get("lora_alpha") != 2:
    raise RuntimeError("zero control is not rank 2 with alpha 2")

native = {update.name: update for update in load_adapter_updates(native_path)}
zero = {update.name: update for update in load_adapter_updates(zero_path)}
if len(native) != 24 or native.keys() != zero.keys():
    raise RuntimeError("zero-control and source module sets differ")
for name in native:
    if not torch.equal(native[name].right, zero[name].right):
        raise RuntimeError(f"zero-control right factor differs for {name}")
    if torch.count_nonzero(zero[name].left).item() != 0:
        raise RuntimeError(f"zero-control left factor is nonzero for {name}")
print("verified exact zero-LoRA control")
PY

if [[ "$EVAL_PROTOCOL" == confirmatory-untouched384-after-screen-v1 ]]; then
  docker run --rm -i \
    -v "$REPO:/workspace" \
    -w /workspace \
    "$IMAGE" \
    python3 - \
      "$zero_container" \
      "$container_root/$SELECTED_CANDIDATE/peft_adapter" \
      "$screen_container" \
      "$SELECTED_CANDIDATE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


zero_path, trained_path, screen_path, selected = map(Path, sys.argv[1:])
selected = selected.name
names = ("adapter_config.json", "adapter_model.safetensors")


def fingerprint(path):
    return {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in names
    }


roles = {
    "zero_a": (zero_path, fingerprint(zero_path)),
    "trained_a": (trained_path, fingerprint(trained_path)),
    "trained_b": (trained_path, fingerprint(trained_path)),
    "zero_b": (zero_path, fingerprint(zero_path)),
}
if roles["zero_a"] != roles["zero_b"]:
    raise RuntimeError("ABBA zero repeats differ by path or artifact hash")
if roles["trained_a"] != roles["trained_b"]:
    raise RuntimeError("ABBA trained repeats differ by path or artifact hash")
screen = json.loads(screen_path.read_text())
if roles["zero_a"][1] != {
    name: screen["candidates"]["zero"]["adapter_artifacts"][name]["sha256"]
    for name in names
}:
    raise RuntimeError("current zero artifact hashes do not match the screen")
if roles["trained_a"][1] != {
    name: screen["candidates"][selected]["adapter_artifacts"][name]["sha256"]
    for name in names
}:
    raise RuntimeError("current selected adapter hashes do not match the screen")
print("Verified ABBA paths and hashes: " + json.dumps({
    role: {"path": str(path), "sha256": hashes}
    for role, (path, hashes) in roles.items()
}, sort_keys=True))
PY
fi

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  printf 'Preflight passed; evaluation was not launched. Output will be %s\n' "$output"
  exit 0
fi

adapter_args=()
holm_args=()
analysis_args=()
execution_args=()
comparison_baseline=zero
if [[ "$EVAL_PROTOCOL" == confirmatory-untouched384-after-screen-v1 ]]; then
  adapter_args=(
    --adapter "zero_a=$zero_container"
    --adapter "trained_a=$container_root/$SELECTED_CANDIDATE/peft_adapter"
    --adapter "trained_b=$container_root/$SELECTED_CANDIDATE/peft_adapter"
    --adapter "zero_b=$zero_container"
  )
  comparison_baseline=zero_a
  analysis_args=(
    --no-holm
    --repeated-contrast zero_a,trained_a,trained_b,zero_b
    --selection-rationale "$SELECTION_RATIONALE"
  )
  execution_args=(--reuse-lora-slot --enforce-eager)
else
  adapter_args=(--adapter "zero=$zero_container")
  for label in "${labels[@]}"; do
    adapter_args+=(--adapter "$label=$container_root/$label/peft_adapter")
    holm_args+=(--holm-label "$label")
  done
  adapter_args+=(--adapter "zero_repeat=$zero_container")
fi

server_was_running=0
if container_is_running "$SERVER_CONTAINER"; then
  server_was_running=1
fi

restore_server() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ "$server_was_running" == 1 ]]; then
    if ! container_is_running "$SERVER_CONTAINER"; then
      docker start "$SERVER_CONTAINER" >/dev/null || rc=1
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
      printf 'Restored %s, but its health check failed.\n' "$SERVER_CONTAINER" >&2
      [[ "$rc" != 0 ]] || rc=1
    fi
  fi
  exit "$rc"
}
trap restore_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$server_was_running" == 1 ]]; then
  docker stop --timeout 60 "$SERVER_CONTAINER" >/dev/null
fi
assert_no_competing_gpu_owners

docker run --rm --gpus all --ipc=host \
  --name "$EVAL_CONTAINER" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${docker_environment_args[@]}" \
  -v /home/nimitz/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nimitz/.cache/vllm:/root/.cache/vllm \
  -v "$REPO:/workspace" \
  "$IMAGE" \
  python3 /workspace/evaluate_gsm8k_adapters.py \
    --model "$MODEL" \
    --revision "$MODEL_REVISION" \
    "${adapter_args[@]}" \
    --comparison-baseline "$comparison_baseline" \
    "${holm_args[@]}" \
    "${analysis_args[@]}" \
    "${execution_args[@]}" \
    --split "$EVAL_SPLIT" \
    --dataset-revision "$DATASET_REVISION" \
    "${disjoint_args[@]}" \
    "${evaluator_environment_args[@]}" \
    --samples "$SAMPLES" \
    --require-exact-samples \
    --max-tokens "$MAX_TOKENS" \
    --max-model-length "$MAX_MODEL_LENGTH" \
    --gpu-memory-utilization 0.50 \
    --lora-dtype bfloat16 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj \
    --kv-cache-dtype fp8 \
    --moe-backend marlin \
    --mamba-backend flashinfer \
    --mamba-cache-mode align \
    --trust-remote-code \
    --prompt-style concise \
    --score-mode strict \
    --temperature 0 \
    --top-p 1 \
    --seed 42 \
    --bootstrap-samples 10000 \
    --bootstrap-seed 42 \
    --include-text \
    --output "/workspace/$OUTPUT_ROOT_REL/$(basename "$output")"
