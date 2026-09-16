#!/usr/bin/env bash
# Evaluate the two scientifically usable checkpoints from the externally
# interrupted Nemotron 3.5 Lightning TinyLoRA sweep.  The 1e-5 checkpoint has
# two optimizer steps; the 5e-5 checkpoint has exactly one optimizer step.
set -Eeuo pipefail

REPO="${REPO:-/home/nimitz/projects/RLtests}"
IMAGE="${IMAGE:-spark-vllm-tinylora:0.1}"
SWEEP_CONTAINER="${SWEEP_CONTAINER:-nemotron35-tinylora-sweep}"
SERVER_CONTAINER="${SERVER_CONTAINER:-nemotron35_lightning_vllm}"
OUTPUT_ROOT_REL="${OUTPUT_ROOT_REL:-outputs/nemotron35-tinylora-lr-scale-canary-20260915}"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-bee7596271d1495f6992ae224aefde4410e816b8}"
DATASET_REVISION="${DATASET_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
SAMPLES="${SAMPLES:-512}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

host_root="$REPO/$OUTPUT_ROOT_REL"
container_root="/workspace/$OUTPUT_ROOT_REL"
output="$host_root/heldout-gsm8k-${SAMPLES}-canonical-zero-control.json"
zero_host="$host_root/zero_control"
zero_container="$container_root/zero_control"
recovery_manifest_container="$container_root/recovery_manifest.json"
c1_native="$container_root/lr1e-5_s1/final_adapter"
c1_peft="$container_root/lr1e-5_s1/peft_adapter"
c2_peft="$container_root/rollout_adapters/step-000005"
expected_zero_start_sha256="ecc8a92451e4bc1b52b44407e6262a25bb6994b2e0311132e7cc28de02114000"
expected_c1_step2_sha256="959b6d50895fb5e176d3c85828bd53d1bb855ea03eb8b18fd622ace64cde6173"
expected_c2_step1_sha256="8b8c1d545e6f07e07e7a18c94931bb14007fb17aba63dea0ea66febcd3681824"

refuse() {
  printf 'Refusing: %s\n' "$*" >&2
  exit 1
}

container_is_running() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true
}

require_complete_peft_or_absent() {
  local path=$1
  local description=$2
  local config="$path/adapter_config.json"
  local weights="$path/adapter_model.safetensors"
  if [[ -e "$path" ]]; then
    [[ -s "$config" && -s "$weights" ]] ||
      refuse "$description is partial; expected both $config and $weights"
    return 0
  fi
  return 1
}

cd "$REPO"
[[ "$SAMPLES" == 512 ]] ||
  refuse 'this recovered protocol fixes the held-out evaluation at 512 examples'
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] ||
  refuse 'PREFLIGHT_ONLY must be 0 or 1'
[[ -d "$host_root" ]] || refuse "missing sweep output root $host_root"
container_is_running "$SWEEP_CONTAINER" && refuse "$SWEEP_CONTAINER is still running"
container_is_running spark-tinylora-rl && refuse 'spark-tinylora-rl is still running'
container_is_running nemotron35-tinylora-heldout-eval &&
  refuse 'the regular held-out evaluation is already running'
container_is_running nemotron35-tinylora-recovered-eval &&
  refuse 'the recovered held-out evaluation is already running'
if pgrep -f '[r]un_nemotron35_tinylora_sweep\.sh' >/dev/null 2>&1; then
  refuse 'the sweep wrapper is still active (the serving-container restore may be pending)'
fi
[[ ! -e "$output" ]] || refuse "evaluation output already exists: $output"

[[ -s "$host_root/sweep_status.json" ]] || refuse 'missing sweep_status.json'
jq -e '.status == "failed"' "$host_root/sweep_status.json" >/dev/null ||
  refuse 'recovery requires the externally interrupted failed sweep'

# Bind the recovery to the exact experiment, actor revision, and candidate
# hyperparameters.  These checks prevent this one-off path being reused for an
# unrelated partial run.
jq -e \
  --arg model "$MODEL" \
  --arg revision "$MODEL_REVISION" '
    .rollout_model == $model and
    .rollout_revision == $revision and
    .common.steps == 2 and
    .common.train_split == "train[:-512]" and
    .common.prompt_style == "concise" and
    .common.reward_mode == "strict" and
    ([.candidates[] | select(.label == "lr1e-5_s1")]
      | length == 1 and .[0].learning_rate == 0.00001 and .[0].scaling == 1) and
    ([.candidates[] | select(.label == "lr5e-5_s1")]
      | length == 1 and .[0].learning_rate == 0.00005 and .[0].scaling == 1)
  ' "$host_root/sweep_manifest.json" >/dev/null ||
  refuse 'sweep_manifest.json does not match this recovery protocol'

# Candidate one completed both requested updates and the normal final save.
c1="$host_root/lr1e-5_s1"
jq -s -e '
  length == 2 and
  (map(.step) == [1, 2]) and
  all(.[]; (.grad_norm | type == "number") and
           (.adapter_norm | type == "number"))
' "$c1/metrics.jsonl" >/dev/null ||
  refuse 'lr1e-5_s1 must contain exactly two metric rows for steps 1 and 2'
[[ -s "$c1/final_adapter/adapter_config.json" ]] ||
  refuse 'lr1e-5_s1 is missing its native final adapter config'
[[ -s "$c1/final_adapter/adapter.safetensors" ]] ||
  refuse 'lr1e-5_s1 is missing its native final adapter weights'
[[ -s "$c1/peft_adapter/adapter_config.json" ]] ||
  refuse 'lr1e-5_s1 is missing its two-step PEFT config'
[[ -s "$c1/peft_adapter/adapter_model.safetensors" ]] ||
  refuse 'lr1e-5_s1 is missing its two-step PEFT weights'

# Candidate two completed only step 1.  step-000005 is the PEFT adapter synced
# immediately after that optimizer update and used for the step-2 rollouts.
# SIGINT arrived before step-2 backward, so those rollouts are explicitly
# discarded and no final/native adapter is expected or synthesized.
c2="$host_root/lr5e-5_s1"
jq -s -e '
  length == 1 and
  (map(.step) == [1]) and
  all(.[]; (.grad_norm | type == "number") and
           (.adapter_norm | type == "number") and
           .synced_weights == 24)
' "$c2/metrics.jsonl" >/dev/null ||
  refuse 'lr5e-5_s1 must contain exactly one valid metric row for step 1'
jq -s -e '
  (group_by(.step) | map({step: .[0].step, rows: length})) ==
  [{"step": 1, "rows": 32}, {"step": 2, "rows": 32}]
' "$c2/trajectories.jsonl" >/dev/null ||
  refuse 'lr5e-5_s1 must have 32 step-1 and 32 unevaluated step-2 trajectories'
[[ ! -e "$c2/final_adapter" ]] ||
  refuse 'lr5e-5_s1 unexpectedly has a final adapter; this protocol expects an interrupted run'
[[ -s "$host_root/rollout_adapters/step-000005/adapter_config.json" ]] ||
  refuse 'missing lr5e-5_s1 one-step rollout PEFT config (step-000005)'
[[ -s "$host_root/rollout_adapters/step-000005/adapter_model.safetensors" ]] ||
  refuse 'missing lr5e-5_s1 one-step rollout PEFT weights (step-000005)'

created_zero=0
if ! require_complete_peft_or_absent "$zero_host" 'zero-control PEFT adapter'; then
  docker run --rm \
    -v "$REPO:/workspace" \
    -w /workspace \
    "$IMAGE" \
    python3 /workspace/export_lora_adapter.py \
      --adapter "$c1_native" \
      --output "$zero_container" \
      --base-model "$MODEL" \
      --multiplier 0
  created_zero=1
fi

# Validate both checkpoint sources, prove the candidate-one PEFT transport
# equals its native adapter, pin candidate two to the known post-step-1 hash,
# and verify the no-op control reuses candidate-one's right factors with zero B.
docker run --rm -i \
  -v "$REPO:/workspace" \
  -w /workspace \
  "$IMAGE" \
  python3 - \
    "$c1_native" \
    "$c1_peft" \
    "$c2_peft" \
    "$zero_container" \
    "$MODEL" \
    "$expected_c1_step2_sha256" \
    "$expected_c2_step1_sha256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

from tinylora_rl.adapter_analysis import load_adapter_updates
from tinylora_rl.registry import _verify_peft_companion


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def updates_by_name(path: str):
    updates = load_adapter_updates(path)
    result = {update.name: update for update in updates}
    if len(result) != 24 or len(result) != len(updates):
        raise RuntimeError(f"{path}: expected 24 unique attention updates")
    return result


def validate_config(path: str, expected_base: str) -> None:
    config = json.loads((Path(path) / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != expected_base:
        raise RuntimeError(f"{path}: unexpected base model declaration")
    if config.get("r") != 2 or config.get("lora_alpha") != 2:
        raise RuntimeError(f"{path}: expected rank 2 with alpha 2")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise RuntimeError(f"{path}: unexpected target modules")


(
    native_path,
    c1_path,
    c2_path,
    zero_path,
    expected_base,
    expected_c1_sha,
    expected_c2_sha,
) = sys.argv[1:]
for path in (c1_path, c2_path, zero_path):
    validate_config(path, expected_base)

# The actor export was materialized on GPU while the native checkpoint is
# reconstructed on CPU.  Use the registry's bounded, per-matrix effective-
# delta comparison rather than demanding bit identity from the two GEMMs.
_verify_peft_companion(
    Path(native_path),
    Path(c1_path),
    base_model=expected_base,
)
actual_c1_sha = sha256(Path(c1_path) / "adapter_model.safetensors")
if actual_c1_sha != expected_c1_sha:
    raise RuntimeError(f"candidate-one hash {actual_c1_sha} != {expected_c1_sha}")

native = updates_by_name(native_path)
c1 = updates_by_name(c1_path)
c2 = updates_by_name(c2_path)
zero = updates_by_name(zero_path)
if not (native.keys() == c1.keys() == c2.keys() == zero.keys()):
    raise RuntimeError("native, candidate, and zero-control module sets differ")
for name in native:
    if not torch.equal(native[name].right, c1[name].right):
        raise RuntimeError(f"candidate-one right factor differs for {name}")
    if not torch.equal(native[name].right, zero[name].right):
        raise RuntimeError(f"zero-control right factor differs for {name}")
    if torch.count_nonzero(zero[name].left).item() != 0:
        raise RuntimeError(f"zero control is nonzero for {name}")
if all(torch.count_nonzero(update.left).item() == 0 for update in c2.values()):
    raise RuntimeError("candidate-two one-step adapter is unexpectedly all zero")
c2_sha = sha256(Path(c2_path) / "adapter_model.safetensors")
if c2_sha != expected_c2_sha:
    raise RuntimeError(f"candidate-two hash {c2_sha} != {expected_c2_sha}")
print("verified step-2/step-1 candidates and the zero-LoRA control")
PY

# Add a recovery manifest without modifying the original status or results.
# A second invocation validates and reuses a matching manifest, allowing an
# evaluation retry without rewriting provenance.
docker run --rm -i \
  -v "$REPO:/workspace" \
  -w /workspace \
  "$IMAGE" \
  python3 - \
    "$container_root" \
    "$recovery_manifest_container" \
    "$MODEL" \
    "$MODEL_REVISION" \
    "$DATASET_REVISION" \
    "$SAMPLES" \
    "$expected_zero_start_sha256" \
    "$expected_c1_step2_sha256" \
    "$expected_c2_step1_sha256" \
    "$created_zero" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def trajectory_hash(path: Path, *, design_only: bool) -> str:
    digest = hashlib.sha256()
    count = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["step"]) != 1:
                continue
            if design_only:
                row = {
                    "step": int(row["step"]),
                    "group_id": int(row["group_id"]),
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                }
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
            count += 1
    if count != 32:
        raise RuntimeError(f"{path}: expected 32 first-step trajectories, found {count}")
    return digest.hexdigest()


(
    root_raw,
    manifest_raw,
    model,
    model_revision,
    dataset_revision,
    samples_raw,
    expected_zero_sha,
    expected_c1_sha,
    expected_c2_sha,
    created_zero_raw,
) = sys.argv[1:]
root = Path(root_raw)
manifest_path = Path(manifest_raw)
source_candidates = ["lr1e-5_s1", "lr5e-5_s1"]
evaluation_labels = ["lr1e-5_s1_step2", "lr5e-5_s1_step1"]

zero_starts = {
    "lr1e-5_s1": file_record(root / "rollout_adapters/step-000001/adapter_model.safetensors"),
    "lr5e-5_s1": file_record(root / "rollout_adapters/step-000004/adapter_model.safetensors"),
}
for label, record in zero_starts.items():
    if record["sha256"] != expected_zero_sha:
        raise RuntimeError(
            f"{label}: zero-start adapter hash {record['sha256']} != {expected_zero_sha}"
        )

c2_checkpoint = file_record(root / "rollout_adapters/step-000005/adapter_model.safetensors")
c1_checkpoint = file_record(root / "lr1e-5_s1/peft_adapter/adapter_model.safetensors")
if c1_checkpoint["sha256"] != expected_c1_sha:
    raise RuntimeError("lr1e-5_s1 two-step checkpoint hash changed")
if c2_checkpoint["sha256"] != expected_c2_sha:
    raise RuntimeError("lr5e-5_s1 one-step checkpoint hash changed")

full_hashes = {
    label: trajectory_hash(root / label / "trajectories.jsonl", design_only=False)
    for label in source_candidates
}
design_hashes = {
    label: trajectory_hash(root / label / "trajectories.jsonl", design_only=True)
    for label in source_candidates
}
if len(set(design_hashes.values())) != 1:
    raise RuntimeError("candidate first-step prompt designs differ")

artifacts = {
    "sweep_manifest": file_record(root / "sweep_manifest.json"),
    "sweep_results": file_record(root / "sweep_results.json"),
    "sweep_status": file_record(root / "sweep_status.json"),
    "zero_control": {
        "config": file_record(root / "zero_control/adapter_config.json"),
        "weights": file_record(root / "zero_control/adapter_model.safetensors"),
    },
    "lr1e-5_s1_step2": {
        "source_candidate": "lr1e-5_s1",
        "completed_optimizer_steps": 2,
        "metrics": file_record(root / "lr1e-5_s1/metrics.jsonl"),
        "trajectories": file_record(root / "lr1e-5_s1/trajectories.jsonl"),
        "native_config": file_record(root / "lr1e-5_s1/final_adapter/adapter_config.json"),
        "native_weights": file_record(root / "lr1e-5_s1/final_adapter/adapter.safetensors"),
        "peft_config": file_record(root / "lr1e-5_s1/peft_adapter/adapter_config.json"),
        "peft_weights": c1_checkpoint,
    },
    "lr5e-5_s1_step1": {
        "source_candidate": "lr5e-5_s1",
        "completed_optimizer_steps": 1,
        "metrics": file_record(root / "lr5e-5_s1/metrics.jsonl"),
        "trajectories": file_record(root / "lr5e-5_s1/trajectories.jsonl"),
        "native_checkpoint": None,
        "peft_config": file_record(root / "rollout_adapters/step-000005/adapter_config.json"),
        "peft_weights": c2_checkpoint,
    },
}

protocol_identity = {
    "recovered_candidates": evaluation_labels,
    "actor_model": model,
    "actor_revision": model_revision,
    "dataset": "openai/gsm8k",
    "dataset_revision": dataset_revision,
    "held_out_split": "train[-512:]",
    "samples": int(samples_raw),
    "comparison_baseline": "zero",
    "evaluation_adapter_order": ["zero", "zero_repeat", *evaluation_labels],
    "zero_repeat": {
        "path_alias_of": "zero",
        "purpose": "measure repeat-request evaluator and LoRA-kernel determinism with a distinct LoRA ID",
    },
}

if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    if existing.get("protocol") != protocol_identity:
        raise RuntimeError(f"existing {manifest_path} has a different recovery protocol")
    if existing.get("zero_start_adapters") != zero_starts:
        raise RuntimeError(f"existing {manifest_path} has different zero-start artifacts")
    print(f"verified existing recovery manifest {manifest_path}")
    raise SystemExit(0)

manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "kind": "externally_interrupted_sweep_recovery",
    "original_status_and_results_preserved": True,
    "termination": {
        "cause": "external_sigint",
        "docker_event_evidence": {
            "container_id": "3b0cc92bf0f18cd4f97c383d3537c974c08649b53c85e0b9b05181e1a82903a8",
            "container_name": "nemotron35-tinylora-sweep",
            "kill": {
                "signal": 2,
                "unix_time": 1789518172,
                "time_nano": 1789518172621930843,
                "timestamp_utc": "2026-09-16T00:22:52.621930843Z",
            },
            "die": {
                "exit_code": 1,
                "unix_time": 1789518177,
                "time_nano": 1789518177308893701,
                "timestamp_utc": "2026-09-16T00:22:57.308893701Z",
            },
            "source": "docker events for nemotron35-tinylora-sweep",
        },
        "state_at_interrupt": (
            "lr5e-5_s1 step-1 optimizer update and PEFT sync completed. All 32 step-2 "
            "rollouts were written, but SIGINT arrived before step-2 backward/optimizer, "
            "metrics, final native save, or sweep candidate export."
        ),
        "discarded_data": {
            "candidate": "lr5e-5_s1",
            "step": 2,
            "trajectory_rows": 32,
            "reason": "no corresponding backward pass or optimizer step",
            "used_for_evaluation": False,
        },
    },
    "trajectory_replay_diagnostic": {
        "note": "This diagnostic did not terminate the run; termination was external SIGINT.",
        "first_step_trajectory_sha256": full_hashes,
        "first_step_design_sha256": design_hashes,
        "designs_match": len(set(design_hashes.values())) == 1,
        "sampled_trajectories_match": len(set(full_hashes.values())) == 1,
    },
    "training_scorer_caveat": {
        "manifest_mode": "strict",
        "actual_loaded_implementation": "legacy_numeric_strict",
        "note": (
            "This training process predates the canonical VERL scorer patch. Its numeric "
            "strict reward is more permissive than VERL's literal #### marker, last-300-"
            "characters, string-equality implementation. Held-out evaluation uses the "
            "canonical VERL implementation."
        ),
    },
    "protocol": protocol_identity,
    "zero_start_adapter_expected_sha256": expected_zero_sha,
    "zero_start_adapters": zero_starts,
    "actions_this_initial_recovery": {
        "candidate_peft_exports_created": [],
        "zero_control_created_by_recovery": True,
        "zero_control_created_this_invocation": created_zero_raw == "1",
    },
    "commands": {
        "candidate_checkpoint_sources": {
            "lr1e-5_s1_step2": f"{root}/lr1e-5_s1/peft_adapter",
            "lr5e-5_s1_step1": f"{root}/rollout_adapters/step-000005",
        },
        "zero_control_export": [
            "python3", "/workspace/export_lora_adapter.py",
            "--adapter", f"{root}/lr1e-5_s1/final_adapter",
            "--output", f"{root}/zero_control",
            "--base-model", model,
            "--multiplier", "0",
        ],
    },
    "artifacts": artifacts,
}
temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n")
temporary.replace(manifest_path)
print(f"wrote recovery manifest {manifest_path}")
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  printf 'Preflight passed; evaluation was not launched. Output will be %s\n' "$output"
  exit 0
fi

# Submit the same no-op adapter twice with distinct LoRA IDs.  Any paired flips
# between zero and zero_repeat measure evaluator/kernel repeatability, not
# learned-adapter effectiveness.  `zero` remains the comparison baseline.
adapter_args=(
  --adapter "zero=$zero_container"
  --adapter "zero_repeat=$zero_container"
  --adapter "lr1e-5_s1_step2=$c1_peft"
  --adapter "lr5e-5_s1_step1=$c2_peft"
)
holm_args=(
  --holm-label lr1e-5_s1_step2
  --holm-label lr5e-5_s1_step1
)

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

docker run --rm --gpus all --ipc=host \
  --name nemotron35-tinylora-recovered-eval \
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
    --split 'train[-512:]' \
    --dataset-revision "$DATASET_REVISION" \
    --samples "$SAMPLES" \
    --max-tokens 512 \
    --max-model-length 768 \
    --gpu-memory-utilization 0.50 \
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
