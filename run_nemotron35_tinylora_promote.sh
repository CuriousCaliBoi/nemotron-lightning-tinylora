#!/usr/bin/env bash
# Promote the completed canonical Nemotron 3.5 Lightning replay sweep into the
# immutable adapter registry. This script is intentionally CPU-only and is
# meant to run inside the research image after the held-out screen and disjoint
# confirmation both exist.
set -Eeuo pipefail

REPO="${REPO:-/workspace}"
REGISTRY="${REGISTRY:-$REPO/adapter_registry}"
OUTPUT_ROOT_REL="outputs/nemotron35-tinylora-lr-scale-replay-canary-20260916"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

LEARNER_MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
LEARNER_REVISION="a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
ACTOR_MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
ACTOR_REVISION="bee7596271d1495f6992ae224aefde4410e816b8"
DATASET_REVISION="740312add88f781978c0658806c59bc2815b9866"
readonly OUTPUT_ROOT_REL LEARNER_MODEL LEARNER_REVISION ACTOR_MODEL ACTOR_REVISION
readonly DATASET_REVISION

output_root="$REPO/$OUTPUT_ROOT_REL"
screen_evaluation="$output_root/screen-consumed128-max1024.json"
confirmation_evaluation="$output_root/confirmatory-untouched384-lr1e-4-s0p5-abba-bf16.json"
zero_adapter="$output_root/zero_control"
screen_eval_name="gsm8k/exploratory-screen-no-evidence-n128-concise-verl-strict-greedy-s42"
confirmation_eval_name="gsm8k/primary-confirmatory-abba-n384-lr1e-4-scale0p5-concise-verl-strict-greedy-s42"
evaluation_paths=("$screen_evaluation" "$confirmation_evaluation")
evaluation_names=("$screen_eval_name" "$confirmation_eval_name")

candidates=(
  lr1e-5_s1
  lr5e-5_s1
  lr1e-4_s1
  lr1e-4_s0p5
)
label_hparams=(
  lr1e-5-scale1
  lr5e-5-scale1
  lr1e-4-scale1
  lr1e-4-scale0p5
)

label_prefix="nemotron3.5-lightning/gsm8k"
protocol_label="tinylora13-concise-verl-strict-exact-step1-replay-v1"
zero_label="$label_prefix/zero-lora-r2-attn24-s42-nvfp4"
peft_labels=()
native_labels=()
for index in "${!candidates[@]}"; do
  stem="$label_prefix/$protocol_label-${label_hparams[$index]}-s42-step2"
  peft_labels+=("$stem-nvfp4")
  native_labels+=("$stem-bf16-native")
done

refuse() {
  printf 'Refusing: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || refuse "missing or empty file $1"
}

[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] ||
  refuse 'PREFLIGHT_ONLY must be 0 or 1'
[[ -f /.dockerenv ]] ||
  refuse 'run this promotion inside the research Docker image as documented'
[[ "$OUTPUT_ROOT_REL" != /* && "/$OUTPUT_ROOT_REL/" != *'/../'* ]] ||
  refuse 'OUTPUT_ROOT_REL must be relative and may not contain parent traversal'
[[ -f "$REPO/tinylora_rl/registry.py" ]] ||
  refuse "REPO does not contain tinylora_rl/registry.py: $REPO"
[[ -d "$output_root" ]] || refuse "missing output root $output_root"

for path in \
  "$output_root/sweep_manifest.json" \
  "$output_root/sweep_results.json" \
  "$output_root/sweep_status.json" \
  "$screen_evaluation" \
  "$confirmation_evaluation" \
  "$zero_adapter/adapter_config.json" \
  "$zero_adapter/adapter_model.safetensors"; do
  require_file "$path"
done
for candidate in "${candidates[@]}"; do
  for relative in \
    run_manifest.json \
    metrics.jsonl \
    trajectories.jsonl \
    final_adapter/adapter_config.json \
    final_adapter/adapter.safetensors \
    peft_adapter/adapter_config.json \
    peft_adapter/adapter_model.safetensors; do
    require_file "$output_root/$candidate/$relative"
  done
done

# Validate the completed sweep, every native/PEFT pair, the exact zero control,
# and the evaluation-to-source artifact bindings before touching the registry.
python3 - \
  "$REPO" \
  "$output_root" \
  "$screen_evaluation" \
  "$confirmation_evaluation" \
  "$LEARNER_MODEL" \
  "$LEARNER_REVISION" \
  "$ACTOR_MODEL" \
  "$ACTOR_REVISION" \
  "$DATASET_REVISION" <<'PY'
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from tinylora_rl.adapter_analysis import load_adapter_updates
from tinylora_rl.registry import _verify_peft_companion


(
    source_root_arg,
    output_root_arg,
    screen_evaluation_arg,
    confirmation_evaluation_arg,
    learner_model,
    learner_revision,
    actor_model,
    actor_revision,
    dataset_revision,
) = sys.argv[1:]
source_root = Path(source_root_arg).resolve()
root = Path(output_root_arg)
screen_evaluation_path = Path(screen_evaluation_arg)
confirmation_evaluation_path = Path(confirmation_evaluation_arg)
candidates = {
    "lr1e-5_s1": (1e-5, 1.0),
    "lr5e-5_s1": (5e-5, 1.0),
    "lr1e-4_s1": (1e-4, 1.0),
    "lr1e-4_s0p5": (1e-4, 0.5),
}
container_root = "/workspace/outputs/nemotron35-tinylora-lr-scale-replay-canary-20260916"
selected_candidate = "lr1e-4_s0p5"
selection_rationale = (
    "selected after the 128-row screen: highest trained strict accuracy "
    "(99/128), +2/128 versus the mean of the two zero repeats, with "
    "conservative 0.5 deployment scale"
)
hex64 = re.compile(r"[0-9a-f]{64}")
expected_runtime_hashes = {
    "train_nemotron35_tinylora_sweep.py": "838d39721ccc9485632193943909cdc18667273ba445b01b5dfda56f48237b4c",
    "tinylora_rl/__init__.py": "f3a61f654eec56064a051ff9a21f74537662fc6089e20cdb6ecb0bcc83cad805",
    "tinylora_rl/adapters.py": "858f9c36a32ae5563ba3717b1a6351af070e11d0f116c1da3c33e698e21413fa",
    "tinylora_rl/objectives.py": "cd52ef5227ee7aef68b53373f1435646ba286636810abb78c0a7a6bfa89edd7a",
    "tinylora_rl/profiling.py": "2637cd3fb667a87311d9380e7621dadf5e7dad49180420eafff9413753f1aff1",
    "tinylora_rl/prompts.py": "4f9e330cd9b65d7f1ea7a329d328093bf1bbcd652c7745612ee1b5e216510c50",
    "tinylora_rl/rewards.py": "0283a2ea178bea6a647960df99ca08f3a9dc02acb4bdb0f222740261906f0e72",
    "tinylora_rl/rollout.py": "95f4bdda043e6af49ebf3cc0d73cb2fe569d7c80ddef658ed08566d1f685dbba",
    "tinylora_rl/trainer.py": "8ddc8050351cae9835d93de4c0e10c9901b7b50034b3b0b131918bab3b2d0d2f",
}
expected_factor_hash = "da2cccb61f2745aec631ca94c5ec4a0e764386e9d0c21c789f1d83c1ced95594"


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON at {path}:{number}: {exc}")
        if not isinstance(value, dict):
            fail(f"expected a JSON object at {path}:{number}")
        records.append(value)
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_binding(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        artifact = path / filename
        result[filename] = {
            "sha256": sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
    return result


def step_hash(records: list[dict], step: int = 1) -> str:
    selected = [row for row in records if int(row.get("step", -1)) == step]
    if not selected:
        fail(f"no trajectories found for step {step}")
    digest = hashlib.sha256()
    for row in selected:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


manifest = read_json(root / "sweep_manifest.json")
results = read_json(root / "sweep_results.json")
status = read_json(root / "sweep_status.json")
expected_candidate_rows = [
    {"label": label, "learning_rate": lr, "scaling": scaling}
    for label, (lr, scaling) in candidates.items()
]
if manifest.get("model") != learner_model or manifest.get("model_revision") != learner_revision:
    fail("sweep manifest does not bind the expected BF16 learner revision")
if manifest.get("rollout_model") != actor_model or manifest.get("rollout_revision") != actor_revision:
    fail("sweep manifest does not bind the expected NVFP4 actor revision")
if manifest.get("candidates") != expected_candidate_rows:
    fail("sweep candidate order or hyperparameters differ from the promotion plan")
common = manifest.get("common")
if not isinstance(common, dict) or any(
    (
        common.get("steps") != 2,
        common.get("train_split") != "train[:-512]",
        common.get("dataset_revision") != dataset_revision,
        common.get("samples") != 256,
        common.get("prompts_per_step") != 4,
        common.get("generations") != 8,
        common.get("max_completion_length") != 768,
        common.get("max_model_length") != 1024,
        common.get("micro_batch_size") != 1,
        common.get("temperature") != 1.2,
        common.get("top_p") != 1.0,
        common.get("prompt_style") != "concise",
        common.get("reward_mode") != "strict",
        common.get("ppo_epochs") != 1,
        common.get("clip_epsilon") != 0.2,
        common.get("tis_mode") != "token_clip",
        common.get("tis_minimum") != 0.1,
        common.get("tis_maximum") != 10.0,
        common.get("loss_reduction") != "sample_mean",
        common.get("seed") != 42,
    )
):
    fail("sweep common settings differ from the canonical replay protocol")
pairing = manifest.get("pairing")
if not isinstance(pairing, dict) or pairing.get("protocol") != "exact_first_step_token_text_logprob_replay_v1" or pairing.get("later_steps_replayed") is not False:
    fail("sweep is not the exact-first-step replay-v1 protocol")
adapter_config = manifest.get("adapter_config")
if not isinstance(adapter_config, dict) or any(
    (
        adapter_config.get("rank") != 2,
        adapter_config.get("projection_dim") != 1,
        adapter_config.get("num_groups") != 13,
        adapter_config.get("modules_per_group") != 16,
        adapter_config.get("grouping") != "tiled",
        adapter_config.get("projection_seed") != 42,
        adapter_config.get("parameter_dtype") != "float32",
        set(adapter_config.get("target_modules", []))
        != {"q_proj", "k_proj", "v_proj", "o_proj"},
    )
):
    fail("sweep adapter configuration is not the 13-scalar TinyLoRA protocol")
runtime_hashes = manifest.get("runtime_source_sha256")
if runtime_hashes != expected_runtime_hashes:
    fail("sweep manifest runtime-source hashes differ from the pinned launch")
for relative, expected_hash in runtime_hashes.items():
    source_path = (source_root / relative).resolve()
    if source_root not in source_path.parents or not source_path.is_file():
        fail(f"runtime source path is missing or unsafe: {relative!r}")
    if sha256(source_path) != expected_hash:
        fail(f"runtime source SHA-256 differs from sweep provenance: {relative}")
factor_cache = Path(str(manifest.get("factor_cache", "")))
if manifest.get("factor_cache_sha256") != expected_factor_hash or not factor_cache.is_file():
    fail("the hash-pinned factor cache is unavailable")
if sha256(factor_cache) != expected_factor_hash:
    fail("factor-cache SHA-256 differs from the sweep manifest")
if status.get("status") != "complete" or status.get("results") != results:
    fail("sweep status is not complete or disagrees with sweep_results.json")
if list(results) != list(candidates):
    fail("sweep results do not contain exactly the four planned candidates in order")

reference_request_hash = None
reference_trajectory_hash = None
source_bindings = {}
native_updates_by_candidate = {}
for label, (learning_rate, scaling) in candidates.items():
    candidate = root / label
    run = read_json(candidate / "run_manifest.json")
    if run.get("schema_version", 0) < 3:
        fail(f"{label}: run manifest predates schema 3")
    learner = run.get("learner")
    rollout = run.get("rollout")
    dataset = run.get("dataset")
    train = run.get("train_config")
    tiny = run.get("adapter_config")
    sweep = run.get("sweep")
    if not all(isinstance(item, dict) for item in (learner, rollout, dataset, train, tiny, sweep)):
        fail(f"{label}: incomplete structured provenance")
    if learner.get("model_id") != learner_model or learner.get("requested_revision") != learner_revision or learner.get("load_dtype") != "bfloat16":
        fail(f"{label}: learner provenance mismatch")
    if rollout.get("model_id") != actor_model or rollout.get("requested_revision") != actor_revision or rollout.get("sync_mode") != "lora":
        fail(f"{label}: actor provenance mismatch")
    if dataset.get("id") != "openai/gsm8k" or dataset.get("config") != "main" or dataset.get("split") != "train[:-512]" or dataset.get("revision") != dataset_revision:
        fail(f"{label}: dataset provenance mismatch")
    if train.get("steps") != 2 or train.get("learning_rate") != learning_rate or train.get("seed") != 42 or train.get("prompt_style") != "concise" or train.get("reward_mode") != "strict":
        fail(f"{label}: training hyperparameter mismatch")
    if tiny.get("scaling") != scaling or tiny.get("rank") != 2 or tiny.get("projection_dim") != 1 or tiny.get("num_groups") != 13 or tiny.get("parameter_dtype") != "float32":
        fail(f"{label}: TinyLoRA hyperparameter mismatch")
    if run.get("target_layers") != 24 or run.get("trainable_parameters") != 13:
        fail(f"{label}: unexpected target or trainable-parameter count")
    if run.get("reward") != {"mode": "strict", "implementation": "verl_gsm8k_strict_last_300_chars_v1"}:
        fail(f"{label}: non-canonical reward implementation")
    if sweep.get("label") != label or sweep.get("runtime_source_sha256") != runtime_hashes or sweep.get("first_step_rollout_protocol") != "exact_token_logprob_replay_v1":
        fail(f"{label}: sweep provenance mismatch")

    metrics = read_jsonl(candidate / "metrics.jsonl")
    if len(metrics) != 2 or [row.get("step") for row in metrics] != [1, 2]:
        fail(f"{label}: expected exactly optimizer steps 1 and 2")
    for row in metrics:
        if row.get("synced_weights") != 24 or not isinstance(row.get("grad_norm"), (int, float)) or not math.isfinite(row["grad_norm"]) or not isinstance(row.get("adapter_norm"), (int, float)) or not math.isfinite(row["adapter_norm"]) or row["adapter_norm"] <= 0:
            fail(f"{label}: invalid optimizer-step metrics")
    trajectories = read_jsonl(candidate / "trajectories.jsonl")
    counts = {step: sum(int(row.get("step", -1)) == step for row in trajectories) for step in (1, 2)}
    if len(trajectories) != 64 or counts != {1: 32, 2: 32}:
        fail(f"{label}: expected 32 trajectories for each of two steps")

    result = results[label]
    actual_step_hash = step_hash(trajectories)
    if result.get("learning_rate") != learning_rate or result.get("scaling") != scaling or result.get("last_training_metrics") != metrics[-1]:
        fail(f"{label}: sweep result does not describe the completed run")
    if result.get("first_step_trajectory_sha256") != actual_step_hash or result.get("first_step_trajectory_matches_reference") is not True:
        fail(f"{label}: first-step trajectory hash/replay invariant failed")
    request_hash = result.get("first_step_request_sha256")
    if not isinstance(request_hash, str) or hex64.fullmatch(request_hash) is None:
        fail(f"{label}: invalid first-step request hash")
    if reference_request_hash is None:
        reference_request_hash = request_hash
        reference_trajectory_hash = actual_step_hash
        if result.get("first_step_rollout_replayed") is not False:
            fail(f"{label}: reference candidate incorrectly marked replayed")
    else:
        if request_hash != reference_request_hash or actual_step_hash != reference_trajectory_hash:
            fail(f"{label}: exact first-step replay differs from the reference")
        if result.get("first_step_rollout_replayed") is not True:
            fail(f"{label}: candidate was not marked as a first-step replay")

    native_path = candidate / "final_adapter"
    peft_path = candidate / "peft_adapter"
    native_config = read_json(native_path / "adapter_config.json")
    modules = native_config.get("modules")
    if not isinstance(modules, list) or len(modules) != 24:
        fail(f"{label}: native adapter does not contain 24 modules")
    if native_config.get("config", {}).get("scaling") != scaling:
        fail(f"{label}: native adapter scaling mismatch")
    peft_config = read_json(peft_path / "adapter_config.json")
    if peft_config.get("base_model_name_or_path") != actor_model or peft_config.get("r") != 2 or peft_config.get("lora_alpha") != 2 or set(peft_config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        fail(f"{label}: PEFT actor adapter metadata mismatch")
    _verify_peft_companion(native_path, peft_path, base_model=actor_model)
    source_bindings[label] = artifact_binding(peft_path)
    native_updates = {update.name: update for update in load_adapter_updates(native_path)}
    if len(native_updates) != 24:
        fail(f"{label}: native adapter update set is not 24 unique modules")
    native_updates_by_candidate[label] = native_updates

zero_config = read_json(root / "zero_control" / "adapter_config.json")
if zero_config.get("base_model_name_or_path") != actor_model or zero_config.get("r") != 2 or zero_config.get("lora_alpha") != 2 or set(zero_config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
    fail("zero control is not the expected rank-2 NVFP4 PEFT adapter")
zero_updates = {update.name: update for update in load_adapter_updates(root / "zero_control")}
reference_updates = native_updates_by_candidate["lr1e-5_s1"]
if len(zero_updates) != 24 or zero_updates.keys() != reference_updates.keys():
    fail("zero control does not share the reference candidate module universe")
for name, update in zero_updates.items():
    if torch.count_nonzero(update.left).item() != 0:
        fail(f"zero control has a nonzero left factor for {name}")
    if not torch.equal(update.right, reference_updates[name].right):
        fail(f"zero control right factor differs from the reference for {name}")
source_bindings["zero"] = artifact_binding(root / "zero_control")

# The already-generated control was exported from FP32 native factors, whereas
# rollout PEFT artifacts were saved in BF16. vLLM casts every loaded adapter to
# its configured LoRA dtype; prove that an explicit BF16 cast makes the frozen
# A factors bit-identical and leaves every control B factor exactly zero. This
# makes the on-disk dtype difference transparent instead of silently assuming
# that equal mathematical updates imply an equal runtime path.
zero_weight_path = root / "zero_control" / "adapter_model.safetensors"
zero_tensors = load_file(str(zero_weight_path), device="cpu")
zero_a_keys = sorted(key for key in zero_tensors if ".lora_A." in key)
zero_b_keys = sorted(key for key in zero_tensors if ".lora_B." in key)
if (
    len(zero_tensors) != 48
    or len(zero_a_keys) != 24
    or len(zero_b_keys) != 24
    or sum(tensor.numel() for tensor in zero_tensors.values()) != 233472
    or sum(zero_tensors[key].numel() for key in zero_a_keys) != 145920
    or sum(zero_tensors[key].numel() for key in zero_b_keys) != 87552
    or {tensor.dtype for tensor in zero_tensors.values()} != {torch.float32}
    or zero_weight_path.stat().st_size != 939912
):
    fail("zero control does not have the audited 48-tensor FP32 structure")
if any(torch.count_nonzero(zero_tensors[key]).item() != 0 for key in zero_b_keys):
    fail("zero control contains a nonzero PEFT B tensor")
for label in candidates:
    learned_weight_path = root / label / "peft_adapter" / "adapter_model.safetensors"
    learned_tensors = load_file(str(learned_weight_path), device="cpu")
    if (
        learned_tensors.keys() != zero_tensors.keys()
        or len(learned_tensors) != 48
        or sum(tensor.numel() for tensor in learned_tensors.values()) != 233472
        or {tensor.dtype for tensor in learned_tensors.values()} != {torch.bfloat16}
        or learned_weight_path.stat().st_size != 473008
    ):
        fail(f"{label}: learned PEFT structure is not the audited BF16 layout")
    if any(
        not torch.equal(zero_tensors[key].to(torch.bfloat16), learned_tensors[key])
        for key in zero_a_keys
    ):
        fail(f"{label}: zero-control A factors differ after the runtime BF16 cast")

def require_close(observed: object, expected: float, context: str) -> None:
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(observed))
        or not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-15)
    ):
        fail(f"{context} is {observed!r}, expected {expected!r}")


def bootstrap_interval(
    row_effects: list[float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> list[float]:
    generator = random.Random(seed)
    count = len(row_effects)
    draws = sorted(
        sum(row_effects[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(bootstrap_samples)
    )
    return [
        draws[int(0.025 * (bootstrap_samples - 1))],
        draws[int(0.975 * (bootstrap_samples - 1))],
    ]


def expected_pair(before: list[bool], after: list[bool]) -> dict[str, object]:
    deltas = [int(new) - int(old) for old, new in zip(before, after, strict=True)]
    wrong_to_right = sum(value == 1 for value in deltas)
    right_to_wrong = sum(value == -1 for value in deltas)
    discordant = wrong_to_right + right_to_wrong
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(wrong_to_right, right_to_wrong) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "accuracy_delta": sum(deltas) / len(deltas),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "mcnemar_exact_p": p_value,
    }


def validate_pair_fields(value: object, expected: dict[str, object], context: str) -> None:
    if not isinstance(value, dict):
        fail(f"{context} is not an object")
    for key in ("wrong_to_right", "right_to_wrong"):
        if value.get(key) != expected[key]:
            fail(f"{context}.{key} disagrees with detailed correctness")
    for key in ("accuracy_delta", "mcnemar_exact_p"):
        require_close(value.get(key), float(expected[key]), f"{context}.{key}")
    interval = value.get("paired_bootstrap_95_ci")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not -1.0 <= float(item) <= 1.0
            for item in interval
        )
        or float(interval[0]) > float(interval[1])
    ):
        fail(f"{context} has an invalid paired bootstrap interval")


def validate_exact_pair_bootstrap(
    value: dict,
    before: list[bool],
    after: list[bool],
    context: str,
) -> None:
    expected = bootstrap_interval(
        [float(int(new) - int(old)) for old, new in zip(before, after, strict=True)]
    )
    observed = value.get("paired_bootstrap_95_ci")
    if not isinstance(observed, list) or len(observed) != 2:
        fail(f"{context} lacks its paired bootstrap interval")
    for index, expected_value in enumerate(expected):
        require_close(
            observed[index],
            expected_value,
            f"{context}.paired_bootstrap_95_ci[{index}]",
        )


def validate_evaluation_record(
    protocol: str,
    name: str,
    record: object,
    samples: int,
) -> None:
    context = f"{protocol} evaluation record {name!r}"
    if not isinstance(record, dict):
        fail(f"{context} is not an object")
    details = record.get("details")
    if record.get("samples") != samples or not isinstance(details, list) or len(details) != samples:
        fail(f"{context} lacks {samples} detailed results")
    if any(not isinstance(detail, dict) for detail in details):
        fail(f"{context} contains a malformed detail")
    if [detail.get("index") for detail in details] != list(range(samples)):
        fail(f"{context} has invalid detail ordering")
    if any(
        not isinstance(detail.get("completion"), str)
        or not isinstance(detail.get("completion_sha256"), str)
        or hex64.fullmatch(detail["completion_sha256"]) is None
        or hashlib.sha256(detail["completion"].encode()).hexdigest()
        != detail["completion_sha256"]
        or not isinstance(detail.get("completion_token_ids_sha256"), str)
        or hex64.fullmatch(detail["completion_token_ids_sha256"]) is None
        or isinstance(detail.get("completion_tokens"), bool)
        or not isinstance(detail.get("completion_tokens"), int)
        or detail["completion_tokens"] < 0
        or not isinstance(detail.get("strict_correct"), bool)
        or not isinstance(detail.get("flexible_correct"), bool)
        for detail in details
    ):
        fail(f"{context} has invalid completion/correctness details")
    strict_correct = sum(detail["strict_correct"] for detail in details)
    flexible_correct = sum(detail["flexible_correct"] for detail in details)
    if (
        record.get("score_mode") != "strict"
        or record.get("correct") != strict_correct
        or record.get("strict_correct") != strict_correct
        or record.get("flexible_correct") != flexible_correct
    ):
        fail(f"{context} summary counts disagree with its details")
    for key, expected in (
        ("accuracy", strict_correct / samples),
        ("strict_accuracy", strict_correct / samples),
        ("flexible_accuracy", flexible_correct / samples),
    ):
        require_close(record.get(key), expected, f"{context}.{key}")


def validate_evaluation(
    path: Path,
    *,
    protocol: str,
    split: str,
    samples: int,
    require_attestation: bool,
    require_explicit_bf16: bool,
) -> dict:
    evaluation = read_json(path)
    expected_values = {
        "schema_version": 3,
        "metric": "gsm8k_greedy_exact_match",
        "model": actor_model,
        "revision": actor_revision,
        "requested_revision": actor_revision,
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": dataset_revision,
        "split": split,
        "source_dataset_rows": samples,
        "selected_dataset_rows": samples,
        "require_exact_samples": True,
        "shuffle_seed": None,
        "prompt_style": "concise",
        "score_mode": "strict",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 42,
        "max_tokens": 1024,
        "max_model_length": 1280,
        "gpu_memory_utilization": 0.5,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "kv_cache_dtype": "fp8",
        "moe_backend": "marlin",
        "mamba_backend": "flashinfer",
        "mamba_cache_mode": "align",
        "trust_remote_code": True,
        "comparison_baseline": "zero",
    }
    for key, expected in expected_values.items():
        if evaluation.get(key) != expected:
            fail(
                f"{protocol} evaluation field {key!r} is "
                f"{evaluation.get(key)!r}, expected {expected!r}"
            )
    # The screen started before explicit dtype provenance was added. Its vLLM
    # 0.27.1 engine used LoRA dtype=auto on a BF16 model; the confirmation must
    # pin BF16 explicitly, and the structural cast proof above covers both.
    if require_explicit_bf16:
        if evaluation.get("lora_dtype") != "bfloat16":
            fail(f"{protocol} evaluation did not explicitly use BF16 LoRA runtime buffers")
    elif evaluation.get("lora_dtype") not in (None, "auto", "bfloat16"):
        fail(f"{protocol} evaluation records an unexpected LoRA dtype")
    if evaluation.get("score_implementations", {}).get("strict") != "verl_gsm8k_strict_last_300_chars_v1":
        fail(f"{protocol} evaluation does not identify the canonical VERL strict scorer")
    for key in ("source_dataset_fingerprint", "selected_dataset_fingerprint", "created_at"):
        if not isinstance(evaluation.get(key), str) or not evaluation[key]:
            fail(f"{protocol} evaluation has invalid provenance field {key!r}")
    versions = evaluation.get("software_versions")
    if not isinstance(versions, dict) or versions.get("vllm") != "0.27.1":
        fail(f"{protocol} evaluation lacks the audited vLLM 0.27.1 runtime")
    provenance = evaluation.get("evaluator_provenance")
    if provenance is not None:
        evaluator_source = source_root / "evaluate_gsm8k_adapters.py"
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema_version") != 1
            or provenance.get("program") != "evaluate_gsm8k_adapters.py"
            or provenance.get("source_sha256") != sha256(evaluator_source)
            or provenance.get("source_size_bytes") != evaluator_source.stat().st_size
            or not isinstance(provenance.get("arguments"), list)
            or any(not isinstance(value, str) for value in provenance["arguments"])
        ):
            fail(f"{protocol} evaluation source attestation is invalid")
    elif require_attestation:
        fail(f"{protocol} evaluation lacks required evaluator source attestation")
    for key in ("question_sha256", "answer_sha256"):
        values = evaluation.get(key)
        if (
            not isinstance(values, list)
            or len(values) != samples
            or any(not isinstance(value, str) or hex64.fullmatch(value) is None for value in values)
        ):
            fail(f"{protocol} evaluation has invalid {key}")
    if len(set(evaluation["question_sha256"])) != samples:
        fail(f"{protocol} evaluation question hashes are not unique")
    if not isinstance(evaluation.get("prompt_set_sha256"), str) or hex64.fullmatch(evaluation["prompt_set_sha256"]) is None:
        fail(f"{protocol} evaluation has an invalid prompt-set hash")

    validate_evaluation_record(protocol, "base", evaluation.get("base"), samples)
    eval_candidates = evaluation.get("candidates")
    expected_eval_candidates = ["zero", *candidates, "zero_repeat"]
    if not isinstance(eval_candidates, dict) or list(eval_candidates) != expected_eval_candidates:
        fail(f"{protocol} evaluation candidates/order differ from the evaluator plan")
    for selector in expected_eval_candidates:
        record = eval_candidates[selector]
        source_selector = "zero" if selector == "zero_repeat" else selector
        validate_evaluation_record(protocol, selector, record, samples)
        expected_adapter_path = (
            f"{container_root}/zero_control"
            if source_selector == "zero"
            else f"{container_root}/{source_selector}/peft_adapter"
        )
        if record.get("adapter") != expected_adapter_path:
            fail(f"{protocol} candidate {selector!r} has the wrong adapter path")
        if record.get("adapter_artifacts") != source_bindings[source_selector]:
            fail(f"{protocol} candidate {selector!r} is not hash-bound to its PEFT adapter")
    comparisons = evaluation.get("comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != {"base", "zero_repeat", *candidates}:
        fail(f"{protocol} comparisons do not cover every non-baseline path")
    if evaluation.get("holm_family_labels") != list(candidates):
        fail(f"{protocol} Holm family is not exactly the four trained candidates")

    reference_details = eval_candidates["zero"]["details"]
    expected_holm_inputs = []
    for selector, comparison in comparisons.items():
        context = f"{protocol} comparison {selector!r}"
        if comparison.get("score_mode") != "strict":
            fail(f"{context} lacks canonical paired statistics")
        candidate_record = evaluation["base"] if selector == "base" else eval_candidates[selector]
        candidate_details = candidate_record["details"]
        paired_by_mode = comparison.get("paired_by_score_mode")
        if not isinstance(paired_by_mode, dict) or set(paired_by_mode) != {"flexible", "strict"}:
            fail(f"{context} lacks both score-mode comparisons")
        expected_by_mode = {}
        for mode in ("flexible", "strict"):
            key = f"{mode}_correct"
            expected = expected_pair(
                [detail[key] for detail in reference_details],
                [detail[key] for detail in candidate_details],
            )
            expected_by_mode[mode] = expected
            validate_pair_fields(paired_by_mode[mode], expected, f"{context}.{mode}")
        validate_pair_fields(comparison, expected_by_mode["strict"], context)
        exact_generation_matches = sum(
            before["completion_sha256"] == after["completion_sha256"]
            and before["completion_token_ids_sha256"]
            == after["completion_token_ids_sha256"]
            and before.get("finish_reason") == after.get("finish_reason")
            for before, after in zip(reference_details, candidate_details, strict=True)
        )
        if comparison.get("identical_generations") != exact_generation_matches:
            fail(f"{context} has an invalid generation-match count")
        require_close(
            comparison.get("identical_generation_rate"),
            exact_generation_matches / samples,
            f"{context}.identical_generation_rate",
        )
        if selector in candidates:
            if (
                comparison.get("holm_family_member") is not True
                or comparison.get("multiple_comparison_family_size") != 4
                or not isinstance(comparison.get("mcnemar_holm_p"), (int, float))
            ):
                fail(f"{context} lacks four-way Holm correction")
            expected_holm_inputs.append(
                (float(expected_by_mode["strict"]["mcnemar_exact_p"]), selector)
            )
        elif (
            comparison.get("holm_family_member") is not False
            or "multiple_comparison_family_size" in comparison
            or "mcnemar_holm_p" in comparison
        ):
            fail(f"diagnostic {context} was incorrectly included in Holm correction")

    # Repeatability is a measured diagnostic, not a promotion gate. The
    # screen demonstrated that this vLLM/MoE path can produce different greedy
    # generations for two IDs backed by identical zero weights. The exact
    # match count and paired flips were recomputed above and remain attached to
    # every PEFT object so an inconclusive result cannot masquerade as a gain.
    running_holm = 0.0
    for index, (p_value, selector) in enumerate(sorted(expected_holm_inputs)):
        running_holm = max(running_holm, min(1.0, (4 - index) * p_value))
        require_close(
            comparisons[selector].get("mcnemar_holm_p"),
            running_holm,
            f"{protocol} comparison {selector!r}.mcnemar_holm_p",
        )
    return evaluation


def recompute_repeatability(first: dict, second: dict) -> dict[str, object]:
    first_details = first["details"]
    second_details = second["details"]
    if [row["index"] for row in first_details] != [
        row["index"] for row in second_details
    ]:
        fail("ABBA repeatability rows are not aligned")
    count = len(first_details)
    correctness = {}
    prediction_identity = {}
    for mode in ("strict", "flexible"):
        first_only = sum(
            bool(left[f"{mode}_correct"])
            and not bool(right[f"{mode}_correct"])
            for left, right in zip(first_details, second_details, strict=True)
        )
        second_only = sum(
            not bool(left[f"{mode}_correct"])
            and bool(right[f"{mode}_correct"])
            for left, right in zip(first_details, second_details, strict=True)
        )
        same_prediction = sum(
            left[f"{mode}_prediction"] == right[f"{mode}_prediction"]
            for left, right in zip(first_details, second_details, strict=True)
        )
        disagreement = first_only + second_only
        correctness[mode] = {
            "disagreement_count": disagreement,
            "disagreement_rate": disagreement / count,
            "first_correct_second_wrong": first_only,
            "first_wrong_second_correct": second_only,
        }
        prediction_identity[mode] = {
            "same_count": same_prediction,
            "same_rate": same_prediction / count,
            "different_count": count - same_prediction,
        }
    text_matches = sum(
        left["completion_sha256"] == right["completion_sha256"]
        for left, right in zip(first_details, second_details, strict=True)
    )
    token_matches = sum(
        left["completion_token_ids_sha256"]
        == right["completion_token_ids_sha256"]
        for left, right in zip(first_details, second_details, strict=True)
    )
    finish_matches = sum(
        left.get("finish_reason") == right.get("finish_reason")
        for left, right in zip(first_details, second_details, strict=True)
    )
    joint_matches = sum(
        left["completion_sha256"] == right["completion_sha256"]
        and left["completion_token_ids_sha256"]
        == right["completion_token_ids_sha256"]
        and left.get("finish_reason") == right.get("finish_reason")
        for left, right in zip(first_details, second_details, strict=True)
    )
    return {
        "samples": count,
        "exact_text": {"count": text_matches, "rate": text_matches / count},
        "exact_token_ids": {
            "count": token_matches,
            "rate": token_matches / count,
        },
        "exact_finish_reason": {
            "count": finish_matches,
            "rate": finish_matches / count,
        },
        "exact_text_token_finish": {
            "count": joint_matches,
            "rate": joint_matches / count,
        },
        "correctness": correctness,
        "prediction_identity": prediction_identity,
    }


def validate_confirmation_evaluation(path: Path) -> dict:
    protocol = "ABBA primary confirmation"
    samples = 384
    evaluation = read_json(path)
    expected_values = {
        "schema_version": 3,
        "metric": "gsm8k_greedy_exact_match",
        "model": actor_model,
        "revision": actor_revision,
        "requested_revision": actor_revision,
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": dataset_revision,
        "split": "train[-384:]",
        "source_dataset_rows": samples,
        "selected_dataset_rows": samples,
        "require_exact_samples": True,
        "shuffle_seed": None,
        "prompt_style": "concise",
        "score_mode": "strict",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 42,
        "max_tokens": 1024,
        "max_model_length": 1280,
        "gpu_memory_utilization": 0.5,
        "enforce_eager": True,
        "lora_dtype": "bfloat16",
        "max_loras": 1,
        "max_cpu_loras": 1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "kv_cache_dtype": "fp8",
        "moe_backend": "marlin",
        "mamba_backend": "flashinfer",
        "mamba_cache_mode": "align",
        "trust_remote_code": True,
        "comparison_baseline": "zero_a",
        "required_runtime_environment": {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_BATCH_INVARIANT": "0",
        },
        "adapter_execution_mode": "sequential_reused_lora_slot",
        "holm_family_labels": [],
    }
    for key, expected in expected_values.items():
        if evaluation.get(key) != expected:
            fail(
                f"{protocol} field {key!r} is {evaluation.get(key)!r}, "
                f"expected {expected!r}"
            )
    if evaluation.get("score_implementations") != {
        "strict": "verl_gsm8k_strict_last_300_chars_v1",
        "flexible": "normalized_last_answer_v1",
    }:
        fail(f"{protocol} scorer provenance is incomplete")
    for key in (
        "source_dataset_fingerprint",
        "selected_dataset_fingerprint",
        "created_at",
    ):
        if not isinstance(evaluation.get(key), str) or not evaluation[key]:
            fail(f"{protocol} has invalid provenance field {key!r}")
    versions = evaluation.get("software_versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != {"torch", "transformers", "datasets", "vllm", "safetensors"}
        or versions.get("vllm") != "0.27.1"
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        fail(f"{protocol} lacks complete execution-package provenance")

    zero_path = f"{container_root}/zero_control"
    trained_path = f"{container_root}/{selected_candidate}/peft_adapter"
    screen_path = f"{container_root}/screen-consumed128-max1024.json"
    output_path = (
        f"{container_root}/"
        "confirmatory-untouched384-lr1e-4-s0p5-abba-bf16.json"
    )
    expected_arguments = [
        "--model", actor_model,
        "--revision", actor_revision,
        "--adapter", f"zero_a={zero_path}",
        "--adapter", f"trained_a={trained_path}",
        "--adapter", f"trained_b={trained_path}",
        "--adapter", f"zero_b={zero_path}",
        "--comparison-baseline", "zero_a",
        "--no-holm",
        "--repeated-contrast", "zero_a,trained_a,trained_b,zero_b",
        "--selection-rationale", selection_rationale,
        "--reuse-lora-slot",
        "--enforce-eager",
        "--split", "train[-384:]",
        "--dataset-revision", dataset_revision,
        "--disjoint-evaluation", f"selection_screen={screen_path}",
        "--require-environment", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
        "--require-environment", "VLLM_BATCH_INVARIANT=0",
        "--samples", "384",
        "--require-exact-samples",
        "--max-tokens", "1024",
        "--max-model-length", "1280",
        "--gpu-memory-utilization", "0.50",
        "--lora-dtype", "bfloat16",
        "--lora-target-modules", "q_proj,k_proj,v_proj,o_proj",
        "--kv-cache-dtype", "fp8",
        "--moe-backend", "marlin",
        "--mamba-backend", "flashinfer",
        "--mamba-cache-mode", "align",
        "--trust-remote-code",
        "--prompt-style", "concise",
        "--score-mode", "strict",
        "--temperature", "0",
        "--top-p", "1",
        "--seed", "42",
        "--bootstrap-samples", "10000",
        "--bootstrap-seed", "42",
        "--include-text",
        "--output", output_path,
    ]
    provenance = evaluation.get("evaluator_provenance")
    evaluator_source = source_root / "evaluate_gsm8k_adapters.py"
    expected_provenance = {
        "schema_version": 1,
        "program": "evaluate_gsm8k_adapters.py",
        "source_sha256": sha256(evaluator_source),
        "source_size_bytes": evaluator_source.stat().st_size,
        "arguments": expected_arguments,
    }
    if provenance != expected_provenance:
        fail(f"{protocol} source or invocation attestation is not exact")

    for key in ("question_sha256", "answer_sha256"):
        values = evaluation.get(key)
        if (
            not isinstance(values, list)
            or len(values) != samples
            or any(
                not isinstance(value, str) or hex64.fullmatch(value) is None
                for value in values
            )
        ):
            fail(f"{protocol} has invalid {key}")
    if len(set(evaluation["question_sha256"])) != samples:
        fail(f"{protocol} question hashes are not unique")
    if (
        not isinstance(evaluation.get("prompt_set_sha256"), str)
        or hex64.fullmatch(evaluation["prompt_set_sha256"]) is None
    ):
        fail(f"{protocol} has an invalid prompt-set hash")

    validate_evaluation_record(protocol, "base", evaluation.get("base"), samples)
    arms = evaluation.get("candidates")
    arm_order = ["zero_a", "trained_a", "trained_b", "zero_b"]
    if not isinstance(arms, dict) or list(arms) != arm_order:
        fail(f"{protocol} candidate order is not exact ABBA")
    expected_arm_sources = {
        "zero_a": (zero_path, source_bindings["zero"]),
        "trained_a": (trained_path, source_bindings[selected_candidate]),
        "trained_b": (trained_path, source_bindings[selected_candidate]),
        "zero_b": (zero_path, source_bindings["zero"]),
    }
    for arm in arm_order:
        record = arms[arm]
        validate_evaluation_record(protocol, arm, record, samples)
        expected_path, expected_artifacts = expected_arm_sources[arm]
        if record.get("adapter") != expected_path:
            fail(f"{protocol} arm {arm!r} has the wrong adapter path")
        if record.get("adapter_artifacts") != expected_artifacts:
            fail(f"{protocol} arm {arm!r} has the wrong adapter artifact hashes")
    if (
        arms["zero_a"]["adapter"] != arms["zero_b"]["adapter"]
        or arms["zero_a"]["adapter_artifacts"]
        != arms["zero_b"]["adapter_artifacts"]
        or arms["trained_a"]["adapter"] != arms["trained_b"]["adapter"]
        or arms["trained_a"]["adapter_artifacts"]
        != arms["trained_b"]["adapter_artifacts"]
    ):
        fail(f"{protocol} repeated arms are not exact path-and-hash duplicates")

    comparisons = evaluation.get("comparisons")
    expected_comparison_order = ["base", "trained_a", "trained_b", "zero_b"]
    if not isinstance(comparisons, dict) or list(comparisons) != expected_comparison_order:
        fail(f"{protocol} ordinary diagnostic comparisons are incomplete")
    reference_details = arms["zero_a"]["details"]
    for selector in expected_comparison_order:
        comparison = comparisons[selector]
        if comparison.get("score_mode") != "strict":
            fail(f"{protocol} comparison {selector!r} has the wrong score mode")
        candidate_record = evaluation["base"] if selector == "base" else arms[selector]
        paired_by_mode = comparison.get("paired_by_score_mode")
        if not isinstance(paired_by_mode, dict) or set(paired_by_mode) != {
            "flexible",
            "strict",
        }:
            fail(f"{protocol} comparison {selector!r} lacks both score modes")
        expected_by_mode = {}
        for mode in ("flexible", "strict"):
            key = f"{mode}_correct"
            expected = expected_pair(
                [detail[key] for detail in reference_details],
                [detail[key] for detail in candidate_record["details"]],
            )
            expected_by_mode[mode] = expected
            validate_pair_fields(
                paired_by_mode[mode],
                expected,
                f"{protocol} comparison {selector!r}.{mode}",
            )
        validate_pair_fields(
            comparison,
            expected_by_mode["strict"],
            f"{protocol} comparison {selector!r}",
        )
        exact_generation_matches = sum(
            before["completion_sha256"] == after["completion_sha256"]
            and before["completion_token_ids_sha256"]
            == after["completion_token_ids_sha256"]
            and before.get("finish_reason") == after.get("finish_reason")
            for before, after in zip(
                reference_details, candidate_record["details"], strict=True
            )
        )
        if comparison.get("identical_generations") != exact_generation_matches:
            fail(f"{protocol} comparison {selector!r} has a bad exact-match count")
        require_close(
            comparison.get("identical_generation_rate"),
            exact_generation_matches / samples,
            f"{protocol} comparison {selector!r}.identical_generation_rate",
        )
        if (
            comparison.get("holm_family_member") is not False
            or "multiple_comparison_family_size" in comparison
            or "mcnemar_holm_p" in comparison
        ):
            fail(f"{protocol} comparison {selector!r} was put in a Holm family")

    contrast = evaluation.get("repeated_contrast")
    expected_contrast_header = {
        "schema_version": 1,
        "design": "two_by_two_repeated_abba_v1",
        "adapter_order": arm_order,
        "roles": {
            "zero_a": "zero_a",
            "trained_a": "trained_a",
            "trained_b": "trained_b",
            "zero_b": "zero_b",
        },
        "row_effect_formula": "((trained_a + trained_b) - (zero_a + zero_b)) / 2",
        "bootstrap_unit": "gsm8k_row_with_all_four_repeated_outcomes",
        "bootstrap_samples": 10000,
        "bootstrap_seed": 42,
        "primary_score_mode": "strict",
        "secondary_score_mode": "flexible",
        "predeclared_hypotheses": 1,
        "ordinary_pairwise_tests_are_diagnostic_only": True,
        "selection_rationale": selection_rationale,
        "selection_evaluation_labels": ["selection_screen"],
    }
    if not isinstance(contrast, dict):
        fail(f"{protocol} lacks its repeated contrast")
    for key, expected in expected_contrast_header.items():
        if contrast.get(key) != expected:
            fail(f"{protocol} repeated_contrast.{key} is invalid")
    expected_identity = {
        "zero": {"same_path": True, "same_artifact_hashes": True},
        "trained": {"same_path": True, "same_artifact_hashes": True},
    }
    if contrast.get("within_arm_identity") != expected_identity:
        fail(f"{protocol} does not attest identical artifacts within both arms")
    expected_repeatability = {
        "zero": recompute_repeatability(arms["zero_a"], arms["zero_b"]),
        "trained": recompute_repeatability(arms["trained_a"], arms["trained_b"]),
    }
    if contrast.get("within_arm_repeatability") != expected_repeatability:
        fail(f"{protocol} within-arm repeatability disagrees with detailed rows")

    by_mode = contrast.get("by_score_mode")
    if not isinstance(by_mode, dict) or list(by_mode) != ["strict", "flexible"]:
        fail(f"{protocol} contrast must report strict then flexible")
    recomputed_modes = {}
    for mode in ("strict", "flexible"):
        values = {
            arm: [bool(row[f"{mode}_correct"]) for row in arms[arm]["details"]]
            for arm in arm_order
        }
        row_effects = [
            (int(ta) + int(tb) - int(za) - int(zb)) / 2.0
            for za, ta, tb, zb in zip(
                values["zero_a"],
                values["trained_a"],
                values["trained_b"],
                values["zero_b"],
                strict=True,
            )
        ]
        if len(row_effects) != samples:
            fail(f"{protocol} {mode} contrast does not contain 384 complete rows")
        observed = by_mode.get(mode)
        if not isinstance(observed, dict):
            fail(f"{protocol} lacks {mode} contrast statistics")
        if observed.get("row_effects") != row_effects:
            fail(f"{protocol} {mode} row effects disagree with the declared formula")
        mean_effect = sum(row_effects) / samples
        require_close(
            observed.get("mean_effect"),
            mean_effect,
            f"{protocol} {mode}.mean_effect",
        )
        expected_counts = {
            str(value): row_effects.count(value)
            for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
        }
        if observed.get("row_effect_counts") != expected_counts or sum(
            expected_counts.values()
        ) != samples:
            fail(f"{protocol} {mode} row-effect counts are invalid")
        expected_ci = bootstrap_interval(row_effects)
        observed_ci = observed.get("row_cluster_bootstrap_95_ci")
        if not isinstance(observed_ci, list) or len(observed_ci) != 2:
            fail(f"{protocol} {mode} lacks its 10k row-cluster interval")
        for index, expected_value in enumerate(expected_ci):
            require_close(
                observed_ci[index],
                expected_value,
                f"{protocol} {mode}.row_cluster_bootstrap_95_ci[{index}]",
            )
        repeat_specific = observed.get("repeat_specific")
        if not isinstance(repeat_specific, dict) or set(repeat_specific) != {
            "a_trained_minus_zero",
            "b_trained_minus_zero",
        }:
            fail(f"{protocol} {mode} repeat-specific effects are incomplete")
        repeat_a = expected_pair(values["zero_a"], values["trained_a"])
        repeat_b = expected_pair(values["zero_b"], values["trained_b"])
        validate_pair_fields(
            repeat_specific["a_trained_minus_zero"],
            repeat_a,
            f"{protocol} {mode}.repeat_a",
        )
        validate_exact_pair_bootstrap(
            repeat_specific["a_trained_minus_zero"],
            values["zero_a"],
            values["trained_a"],
            f"{protocol} {mode}.repeat_a",
        )
        validate_pair_fields(
            repeat_specific["b_trained_minus_zero"],
            repeat_b,
            f"{protocol} {mode}.repeat_b",
        )
        validate_exact_pair_bootstrap(
            repeat_specific["b_trained_minus_zero"],
            values["zero_b"],
            values["trained_b"],
            f"{protocol} {mode}.repeat_b",
        )
        recomputed_modes[mode] = {
            "mean_effect": mean_effect,
            "ci": expected_ci,
            "repeat_a": repeat_a["accuracy_delta"],
            "repeat_b": repeat_b["accuracy_delta"],
        }

    strict = recomputed_modes["strict"]
    expected_gate_conditions = {
        "aggregate_strict_mean_gt_zero": strict["mean_effect"] > 0,
        "aggregate_strict_ci_lower_gt_zero": strict["ci"][0] > 0,
        "repeat_a_strict_estimate_gt_zero": strict["repeat_a"] > 0,
        "repeat_b_strict_estimate_gt_zero": strict["repeat_b"] > 0,
    }
    expected_gate = {
        "conditions": expected_gate_conditions,
        "passed": all(expected_gate_conditions.values()),
    }
    if contrast.get("primary_efficacy_gate") != expected_gate:
        fail(f"{protocol} efficacy gate was not recomputed from detailed strict rows")
    return evaluation


screen_evaluation = validate_evaluation(
    screen_evaluation_path,
    protocol="selection screen",
    split="train[-512:-384]",
    samples=128,
    require_attestation=False,
    require_explicit_bf16=False,
)
if screen_evaluation.get("disjoint_evaluations") != {}:
    fail("selection screen unexpectedly declares a prior disjoint evaluation")
screen_candidates = screen_evaluation["candidates"]
if (
    screen_candidates[selected_candidate].get("strict_correct") != 99
    or screen_candidates[selected_candidate].get("flexible_correct") != 115
    or 2 * screen_candidates[selected_candidate]["strict_correct"]
    - screen_candidates["zero"]["strict_correct"]
    - screen_candidates["zero_repeat"]["strict_correct"]
    != 4
    or max(screen_candidates[label]["strict_correct"] for label in candidates) != 99
    or sum(
        screen_candidates[label]["strict_correct"] == 99 for label in candidates
    )
    != 1
):
    fail("selection screen does not reproduce the predeclared c4 selection record")
if any(
    screen_evaluation["comparisons"][label]["mcnemar_holm_p"] < 0.05
    for label in candidates
):
    fail("selection screen unexpectedly contains Holm-adjusted efficacy evidence")
confirmation_evaluation = validate_confirmation_evaluation(
    confirmation_evaluation_path
)
if set(screen_evaluation["question_sha256"]) & set(confirmation_evaluation["question_sha256"]):
    fail("selection-screen and confirmation question hashes overlap")
disjoint = confirmation_evaluation.get("disjoint_evaluations")
if not isinstance(disjoint, dict) or set(disjoint) != {"selection_screen"}:
    fail("confirmation does not name exactly the selection screen as its disjoint set")
screen_binding = {
    "sha256": sha256(screen_evaluation_path),
    "size_bytes": screen_evaluation_path.stat().st_size,
}
expected_disjoint = {
    "path": f"{container_root}/screen-consumed128-max1024.json",
    "artifact": screen_binding,
    "question_count": 128,
    "unique_question_count": 128,
    "question_overlap_count": 0,
    "split": "train[-512:-384]",
    "selected_dataset_rows": 128,
    "dataset_revision": dataset_revision,
    "model": actor_model,
    "requested_revision": actor_revision,
}
if disjoint["selection_screen"] != expected_disjoint:
    fail("confirmation is not content-bound to the exact validated selection screen")

print("validated replay sweep, nine adapters, exploratory screen, and ABBA confirmation")
PY

registry_cli=(python3 -m tinylora_rl.registry)

register_all() {
  local destination_registry=$1
  "${registry_cli[@]}" --registry "$destination_registry" add "$zero_adapter" \
    --label "$zero_label" \
    --base-model "$ACTOR_MODEL" \
    --base-revision "$ACTOR_REVISION" >/dev/null

  for index in "${!candidates[@]}"; do
    candidate_path="$output_root/${candidates[$index]}"
    "${registry_cli[@]}" --registry "$destination_registry" add \
      "$candidate_path/peft_adapter" \
      --label "${peft_labels[$index]}" \
      --base-model "$ACTOR_MODEL" \
      --base-revision "$ACTOR_REVISION" >/dev/null
    "${registry_cli[@]}" --registry "$destination_registry" add \
      "$candidate_path" \
      --label "${native_labels[$index]}" \
      --base-model "$LEARNER_MODEL" \
      --base-revision "$LEARNER_REVISION" >/dev/null
  done

  for eval_index in "${!evaluation_paths[@]}"; do
    "${registry_cli[@]}" --registry "$destination_registry" add-eval \
      "$zero_label" "${evaluation_paths[$eval_index]}" \
      --name "${evaluation_names[$eval_index]}" --candidate zero >/dev/null
    for index in "${!candidates[@]}"; do
      "${registry_cli[@]}" --registry "$destination_registry" add-eval \
        "${peft_labels[$index]}" "${evaluation_paths[$eval_index]}" \
        --name "${evaluation_names[$eval_index]}" \
        --candidate "${candidates[$index]}" >/dev/null
    done
  done
}

temporary_parent="${TMPDIR:-/tmp}"
temporary_registry=$(mktemp -d "$temporary_parent/nemotron35-registry-preflight.XXXXXX")
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "${temporary_registry:-}" && -d "$temporary_registry" &&
        "$(basename "$temporary_registry")" == nemotron35-registry-preflight.* ]]; then
    chmod -R u+rwX "$temporary_registry" 2>/dev/null || true
    rm -rf -- "$temporary_registry"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# A complete dry promotion computes the exact content IDs and evaluation IDs.
# Comparing those IDs lets preflight distinguish a safe retry from a label
# collision without ever passing a registry replacement flag.
register_all "$temporary_registry"

all_labels=("$zero_label" "${peft_labels[@]}" "${native_labels[@]}")

check_registry_state() {
  local require_present=$1
  python3 - \
    "$temporary_registry" \
    "$REGISTRY" \
    "$require_present" \
    "${all_labels[@]}" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath


expected_root = Path(sys.argv[1])
actual_root = Path(sys.argv[2])
require_present = sys.argv[3] == "1"
labels = sys.argv[4:]


def ref_path(root: Path, label: str) -> Path:
    parts = PurePosixPath(label).parts
    return root / "refs" / Path(*parts[:-1], f"{parts[-1]}.json")


for label in labels:
    expected_ref = json.loads(ref_path(expected_root, label).read_text())
    expected_id = expected_ref["object_id"]
    actual_ref_path = ref_path(actual_root, label)
    if not actual_ref_path.exists():
        if require_present:
            raise RuntimeError(f"promotion did not create label {label!r}")
    else:
        actual_ref = json.loads(actual_ref_path.read_text())
        if actual_ref.get("label") != label:
            raise RuntimeError(
                f"registry reference at {actual_ref_path} declares label "
                f"{actual_ref.get('label')!r}, expected {label!r}"
            )
        if actual_ref.get("object_id") != expected_id:
            raise RuntimeError(
                f"label {label!r} already points to {actual_ref.get('object_id')}, "
                f"but this run has content ID {expected_id}; refusing to move it"
            )
        actual_object = actual_root / "objects" / "sha256" / expected_id.removeprefix("sha256:")
        if not actual_object.is_dir():
            raise RuntimeError(
                f"label {label!r} points to missing registry object {expected_id}"
            )

    digest = expected_id.removeprefix("sha256:")
    expected_eval_root = expected_root / "evaluation_refs" / digest
    if not expected_eval_root.exists():
        continue
    for expected_eval_path in expected_eval_root.rglob("*.json"):
        relative = expected_eval_path.relative_to(expected_eval_root)
        expected_eval = json.loads(expected_eval_path.read_text())
        evaluation_name = expected_eval.get("name")
        actual_eval_path = actual_root / "evaluation_refs" / digest / relative
        if not actual_eval_path.exists():
            if require_present:
                raise RuntimeError(
                    f"promotion did not attach {evaluation_name!r} to {label!r}"
                )
            continue
        actual_eval = json.loads(actual_eval_path.read_text())
        if (
            actual_eval.get("adapter_object_id") != expected_id
            or actual_eval.get("name") != evaluation_name
        ):
            raise RuntimeError(f"malformed evaluation reference at {actual_eval_path}")
        if actual_eval.get("evaluation_id") != expected_eval.get("evaluation_id"):
            raise RuntimeError(
                f"evaluation name {evaluation_name!r} on {label!r} already points to "
                f"{actual_eval.get('evaluation_id')}, expected "
                f"{expected_eval.get('evaluation_id')}; refusing to move it"
            )

print("registry labels/evaluation names are collision-free and content-identical where present")
PY
}

check_registry_state 0
declare -A seen_expected_ids=()
for label in "${all_labels[@]}"; do
  "${registry_cli[@]}" --registry "$temporary_registry" verify "$label" >/dev/null
  expected_manifest=$("${registry_cli[@]}" --registry "$temporary_registry" show "$label")
  object_id=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["object_id"])' \
    <<<"$expected_manifest")
  [[ "$object_id" =~ ^sha256:[0-9a-f]{64}$ ]] ||
    refuse "temporary registry returned an invalid object ID for $label"
  [[ -z "${seen_expected_ids[$object_id]:-}" ]] || continue
  seen_expected_ids[$object_id]=1
  digest=${object_id#sha256:}
  if [[ -d "$REGISTRY/objects/sha256/$digest" ]]; then
    "${registry_cli[@]}" --registry "$REGISTRY" verify "$object_id" >/dev/null
  fi
done

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  printf 'Preflight passed; registry was not changed. Planned labels:\n'
  printf '  %s\n' "${all_labels[@]}"
  printf 'Evaluation names:\n'
  printf '  %s\n' "${evaluation_names[@]}"
  exit 0
fi

register_all "$REGISTRY"
check_registry_state 1
for label in "${all_labels[@]}"; do
  "${registry_cli[@]}" --registry "$REGISTRY" verify "$label" >/dev/null
done

printf 'Promoted and hash-verified %d immutable adapter labels.\n' "${#all_labels[@]}"
printf 'Attached and verified two evaluations on the zero control and four NVFP4 adapters.\n'
