#!/usr/bin/env bash
# Evaluate one completed Qwen2.5-7B TinyLoRA canary on the full official GSM8K
# test split.  The same exact zero-valued PEFT directory is loaded under two
# request IDs so zero-vs-zero_repeat measures same-path inference repeatability.
set -Eeuo pipefail

REPO="${REPO:-/home/nimitz/projects/RLtests}"
IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
RUN_SLUG="${RUN_SLUG:-}"
TRAIN_CONTAINER="${TRAIN_CONTAINER:-spark-tinylora-rl}"
SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
EVAL_CONTAINER="${EVAL_CONTAINER:-qwen25-7b-tinylora-gsm8k-eval}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
DATASET_REVISION="${DATASET_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
EXPECTED_FACTOR_SHA256="${EXPECTED_FACTOR_SHA256:-ab49b94216b6d2d46760ea44620766afc30417ee5ed626af620a182cd86344d7}"
FACTOR_CACHE_REL="${FACTOR_CACHE_REL:-cache/Qwen--Qwen2.5-7B-Instruct-r2-svd.safetensors}"
SAMPLES="${SAMPLES:-1319}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-1024}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

refuse() {
  printf 'Refusing: %s\n' "$*" >&2
  exit 1
}

container_is_running() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true
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

require_file() {
  [[ -s "$1" ]] || refuse "missing or empty file $1"
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

[[ -n "$RUN_SLUG" ]] || refuse 'set RUN_SLUG to the completed Qwen canary output directory name'
[[ "$RUN_SLUG" =~ ^[A-Za-z0-9._-]+$ ]] || refuse 'RUN_SLUG must be one safe path segment'
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] ||
  refuse 'PREFLIGHT_ONLY must be 0 or 1'
[[ "$SAMPLES" == 1319 ]] || refuse 'the canonical full-test protocol requires SAMPLES=1319'
[[ "$MAX_TOKENS" == 512 && "$MAX_MODEL_LENGTH" == 1024 ]] ||
  refuse 'the canonical canary evaluation requires MAX_TOKENS=512 and MAX_MODEL_LENGTH=1024'
[[ "$FACTOR_CACHE_REL" != /* && "/$FACTOR_CACHE_REL/" != *'/../'* ]] ||
  refuse 'FACTOR_CACHE_REL must be relative and may not contain parent traversal'

cd "$REPO"
run_root="$REPO/outputs/$RUN_SLUG"
container_root="/workspace/outputs/$RUN_SLUG"
artifact_root="$run_root/evaluation_adapters"
container_artifact_root="$container_root/evaluation_adapters"
evaluation="$run_root/gsm8k-test-n1319-verl-strict-greedy-zero-control.json"
factor_cache="$REPO/$FACTOR_CACHE_REL"
steps=(16 32 48 64)

[[ -d "$run_root" ]] || refuse "missing canary output root $run_root"
container_is_running "$TRAIN_CONTAINER" && refuse "$TRAIN_CONTAINER is still running"
container_is_running "$EVAL_CONTAINER" && refuse "$EVAL_CONTAINER is already running"
for gpu_job in nemotron35-tinylora-sweep nemotron35-tinylora-heldout-eval \
  nemotron35-tinylora-recovered-eval; do
  container_is_running "$gpu_job" && refuse "$gpu_job is still running"
done
if pgrep -f '[r]un_qwen25_7b_tinylora_canary\.sh' >/dev/null 2>&1; then
  refuse 'the Qwen training wrapper is still active (server restore may be pending)'
fi
assert_no_competing_gpu_owners "$SERVER_CONTAINER"
[[ ! -e "$evaluation" ]] || refuse "evaluation output already exists: $evaluation"

for relative in run_manifest.json metrics.jsonl trajectories.jsonl \
  final_adapter/adapter_config.json final_adapter/adapter.safetensors; do
  require_file "$run_root/$relative"
done
for step in "${steps[@]}"; do
  require_file "$run_root/checkpoint-$step/adapter_config.json"
  require_file "$run_root/checkpoint-$step/adapter.safetensors"
done
require_file "$factor_cache"

# Hash inside the research image so root-owned cache/checkpoint permissions do
# not weaken the provenance check.
factor_sha=$(docker run --rm --entrypoint sha256sum \
  -v "$REPO:/workspace:ro" "$IMAGE" "/workspace/$FACTOR_CACHE_REL" | awk '{print $1}')
[[ "$factor_sha" == "$EXPECTED_FACTOR_SHA256" ]] ||
  refuse "factor-cache hash mismatch: $factor_sha"

# Prove completion, the exact training design, checkpoint topology, and that
# checkpoint-64 and final_adapter are duplicate representations of step 64.
docker run --rm -i \
  -v "$REPO:/workspace:ro" \
  -w /workspace \
  "$IMAGE" \
  python3 - \
    "$container_root" "$MODEL" "$MODEL_REVISION" "$DATASET_REVISION" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


root_arg, model, model_revision, dataset_revision = sys.argv[1:]
root = Path(root_arg)


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    result = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            fail(f"expected an object at {path}:{line_number}")
        result.append(value)
    return result


manifest = read_json(root / "run_manifest.json")
learner = manifest.get("learner")
rollout = manifest.get("rollout")
dataset = manifest.get("dataset")
train = manifest.get("train_config")
adapter = manifest.get("adapter_config")
if manifest.get("schema_version") != 3:
    fail("run manifest is not schema 3")
if not all(isinstance(value, dict) for value in (learner, rollout, dataset, train, adapter)):
    fail("run manifest lacks structured provenance")
if learner.get("model_id") != model or learner.get("requested_revision") != model_revision:
    fail("learner model/revision differs from the canary protocol")
if learner.get("resolved_revision") not in (None, model_revision):
    fail("learner resolved revision conflicts with the pinned model commit")
if rollout.get("model_id") != model or rollout.get("requested_revision") != model_revision:
    fail("rollout model/revision differs from the canary protocol")
if rollout.get("resolved_revision") not in (None, model_revision) or rollout.get("sync_mode") != "lora":
    fail("rollout did not use the pinned model with LoRA synchronization")
if any((
    dataset.get("id") != "openai/gsm8k",
    dataset.get("config") != "main",
    dataset.get("split") != "train[:-512]",
    dataset.get("revision") != dataset_revision,
    dataset.get("selected_rows") != 512,
)):
    fail("training dataset provenance differs from the canary protocol")
expected_train = {
    "steps": 64,
    "prompts_per_step": 16,
    "generations_per_prompt": 4,
    "max_completion_length": 512,
    "temperature": 1.0,
    "top_p": 1.0,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "ppo_epochs": 1,
    "clip_epsilon": 0.2,
    "micro_batch_size": 2,
    "tis_mode": "token_clip",
    "tis_minimum": 0.1,
    "tis_maximum": 10.0,
    "loss_reduction": "token_mean",
    "seed": 42,
    "save_every": 16,
    "prompt_style": "verl",
    "reward_mode": "strict",
}
for key, expected in expected_train.items():
    if train.get(key) != expected:
        fail(f"training field {key!r} is {train.get(key)!r}, expected {expected!r}")
expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
if any((
    adapter.get("rank") != 2,
    adapter.get("projection_dim") != 1,
    adapter.get("modules_per_group") != 16,
    adapter.get("num_groups") is not None,
    set(adapter.get("target_modules", [])) != expected_targets,
    adapter.get("target_layer_indices") is not None,
    adapter.get("grouping") != "tiled",
    adapter.get("projection_seed") != 42,
    adapter.get("projection_std") is not None,
    adapter.get("scaling") != 1.0,
    adapter.get("parameter_dtype") != "bfloat16",
    adapter.get("svd_niter") != 2,
    manifest.get("target_layers") != 196,
    manifest.get("trainable_parameters") != 13,
)):
    fail("adapter provenance is not the 13-parameter all-linear TinyLoRA canary")
if manifest.get("prompt") != {"style": "verl"}:
    fail("training prompt was not the VERL prompt")
if manifest.get("reward") != {
    "mode": "strict",
    "implementation": "verl_gsm8k_strict_last_300_chars_v1",
}:
    fail("training reward was not the canonical VERL strict scorer")

metrics = read_jsonl(root / "metrics.jsonl")
if len(metrics) != 64 or [row.get("step") for row in metrics] != list(range(1, 65)):
    fail("metrics.jsonl does not prove exactly 64 completed optimizer steps")
for row in metrics:
    for key in ("grad_norm", "adapter_norm", "step_seconds"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            fail(f"step {row.get('step')} has invalid {key}")
    if row["adapter_norm"] <= 0 or row["step_seconds"] <= 0:
        fail(f"step {row.get('step')} lacks a completed adapter update")
    if row.get("synced_weights") != 196:
        fail(f"step {row.get('step')} did not synchronize 196 target matrices")

trajectories = read_jsonl(root / "trajectories.jsonl")
if len(trajectories) != 64 * 16 * 4:
    fail("trajectories.jsonl does not contain 4096 canary generations")
for step in range(1, 65):
    selected = [row for row in trajectories if row.get("step") == step]
    groups = [row.get("group_id") for row in selected]
    if len(selected) != 64 or sorted(groups) != [group for group in range(16) for _ in range(4)]:
        fail(f"step {step} does not contain four generations for each of 16 prompts")


def validate_native(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    config = read_json(path / "adapter_config.json")
    modules = config.get("modules")
    if config.get("config") != adapter or not isinstance(modules, list) or len(modules) != 196:
        fail(f"native checkpoint {path.name} has incompatible metadata")
    if len({item.get("name") for item in modules if isinstance(item, dict)}) != 196:
        fail(f"native checkpoint {path.name} has duplicate or malformed modules")
    tensors = load_file(str(path / "adapter.safetensors"))
    bank = tensors.get("bank.v")
    if bank is None or tuple(bank.shape) != (13, 1) or bank.dtype != torch.bfloat16:
        fail(f"native checkpoint {path.name} has the wrong TinyLoRA bank")
    return config, tensors


for step in (16, 32, 48, 64):
    validate_native(root / f"checkpoint-{step}")
final_config, final_tensors = validate_native(root / "final_adapter")
step64_config, step64_tensors = validate_native(root / "checkpoint-64")
if final_config != step64_config or final_tensors.keys() != step64_tensors.keys():
    fail("checkpoint-64 and final_adapter metadata/tensor sets differ")
for name in final_tensors:
    if not torch.equal(final_tensors[name], step64_tensors[name]):
        fail(f"checkpoint-64 and final_adapter differ at tensor {name}")
print("validated completed 64-step Qwen canary; final_adapter duplicates checkpoint-64")
PY

for step in "${steps[@]}"; do
  peft_host="$artifact_root/step$step"
  peft_container="$container_artifact_root/step$step"
  if ! require_complete_peft_or_absent "$peft_host" "step-$step PEFT export"; then
    docker run --rm \
      -v "$REPO:/workspace" \
      -w /workspace \
      "$IMAGE" \
      python3 /workspace/export_lora_adapter.py \
        --adapter "$container_root/checkpoint-$step" \
        --output "$peft_container" \
        --base-model "$MODEL"
  fi
done
zero_host="$artifact_root/zero"
zero_container="$container_artifact_root/zero"
if ! require_complete_peft_or_absent "$zero_host" 'zero-control PEFT export'; then
  docker run --rm \
    -v "$REPO:/workspace" \
    -w /workspace \
    "$IMAGE" \
    python3 /workspace/export_lora_adapter.py \
      --adapter "$container_root/checkpoint-16" \
      --output "$zero_container" \
      --base-model "$MODEL" \
      --multiplier 0
fi

# Verify every native/PEFT pair and prove that the zero control shares the
# checkpoint-16 right factors while all materialized left factors are zero.
docker run --rm -i \
  -v "$REPO:/workspace:ro" \
  -w /workspace \
  "$IMAGE" \
  python3 - \
    "$container_root" "$container_artifact_root" "$MODEL" <<'PY'
import sys
from pathlib import Path

import torch

from tinylora_rl.adapter_analysis import load_adapter_updates
from tinylora_rl.registry import _verify_peft_companion


root_arg, artifacts_arg, model = sys.argv[1:]
root = Path(root_arg)
artifacts = Path(artifacts_arg)
for step in (16, 32, 48, 64):
    _verify_peft_companion(
        root / f"checkpoint-{step}",
        artifacts / f"step{step}",
        base_model=model,
    )
source = {update.name: update for update in load_adapter_updates(root / "checkpoint-16")}
zero = {update.name: update for update in load_adapter_updates(artifacts / "zero")}
if len(source) != 196 or source.keys() != zero.keys():
    raise RuntimeError("zero control and checkpoint-16 module sets differ")
for name in source:
    if not torch.equal(source[name].right, zero[name].right):
        raise RuntimeError(f"zero-control right factor differs for {name}")
    if torch.count_nonzero(zero[name].left).item() != 0:
        raise RuntimeError(f"zero-control left factor is nonzero for {name}")
print("verified four native/PEFT bindings and exact zero-LoRA control")
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  printf 'Preflight passed; GPU evaluation was not launched. Output will be %s\n' "$evaluation"
  exit 0
fi

adapter_args=(
  --adapter "zero=$zero_container"
  --adapter "zero_repeat=$zero_container"
)
holm_args=()
for step in "${steps[@]}"; do
  adapter_args+=(--adapter "step$step=$container_artifact_root/step$step")
  holm_args+=(--holm-label "step$step")
done

server_was_running=0
if container_is_running "$SERVER_CONTAINER"; then
  server_was_running=1
fi

restore_server() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if container_is_running "$EVAL_CONTAINER"; then
    printf 'Not restoring %s: evaluation container %s is still running.\n' \
      "$SERVER_CONTAINER" "$EVAL_CONTAINER" >&2
    [[ "$rc" != 0 ]] || rc=1
  elif [[ "$server_was_running" == 1 ]]; then
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
trap 'exit 129' HUP

if [[ "$server_was_running" == 1 ]]; then
  docker stop --timeout 60 "$SERVER_CONTAINER" >/dev/null
fi
assert_no_competing_gpu_owners

docker run --rm --gpus all --ipc=host \
  --name "$EVAL_CONTAINER" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /home/nimitz/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nimitz/.cache/vllm:/root/.cache/vllm \
  -v "$REPO:/workspace" \
  "$IMAGE" \
  python3 /workspace/evaluate_gsm8k_adapters.py \
    --model "$MODEL" \
    --revision "$MODEL_REVISION" \
    "${adapter_args[@]}" \
    --comparison-baseline zero \
    "${holm_args[@]}" \
    --split test \
    --dataset-revision "$DATASET_REVISION" \
    --samples "$SAMPLES" \
    --require-exact-samples \
    --max-tokens "$MAX_TOKENS" \
    --max-model-length "$MAX_MODEL_LENGTH" \
    --gpu-memory-utilization 0.50 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --kv-cache-dtype auto \
    --prompt-style verl \
    --score-mode strict \
    --temperature 0 \
    --top-p 1 \
    --seed 42 \
    --bootstrap-samples 10000 \
    --bootstrap-seed 42 \
    --include-text \
    --output "/workspace/outputs/$RUN_SLUG/$(basename "$evaluation")"
