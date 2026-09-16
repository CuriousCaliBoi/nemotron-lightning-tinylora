#!/usr/bin/env bash
# Promote a fully evaluated Qwen2.5-7B TinyLoRA canary into the immutable local
# registry.  This is CPU-only.  PREFLIGHT_ONLY=1 validates everything and
# computes the exact planned object/evaluation IDs without changing REGISTRY.
set -Eeuo pipefail

REPO="${REPO:-/workspace}"
REGISTRY="${REGISTRY:-$REPO/adapter_registry}"
RUN_SLUG="${RUN_SLUG:-}"
REGISTRY_RUN_TAG="${REGISTRY_RUN_TAG:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
DATASET_REVISION="${DATASET_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
EXPECTED_FACTOR_SHA256="${EXPECTED_FACTOR_SHA256:-ab49b94216b6d2d46760ea44620766afc30417ee5ed626af620a182cd86344d7}"
FACTOR_CACHE_REL="${FACTOR_CACHE_REL:-cache/Qwen--Qwen2.5-7B-Instruct-r2-svd.safetensors}"

refuse() {
  printf 'Refusing: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || refuse "missing or empty file $1"
}

[[ -n "$RUN_SLUG" ]] || refuse 'set RUN_SLUG to the evaluated Qwen canary directory name'
[[ "$RUN_SLUG" =~ ^[A-Za-z0-9._-]+$ ]] || refuse 'RUN_SLUG must be one safe path segment'
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] ||
  refuse 'PREFLIGHT_ONLY must be 0 or 1'
[[ -f /.dockerenv ]] ||
  refuse 'run promotion inside the research Docker image so root-owned artifacts are readable'
[[ "$FACTOR_CACHE_REL" != /* && "/$FACTOR_CACHE_REL/" != *'/../'* ]] ||
  refuse 'FACTOR_CACHE_REL must be relative and may not contain parent traversal'
[[ -f "$REPO/tinylora_rl/registry.py" ]] ||
  refuse "REPO does not contain tinylora_rl/registry.py: $REPO"

run_root="$REPO/outputs/$RUN_SLUG"
artifact_root="$run_root/evaluation_adapters"
evaluation="$run_root/gsm8k-test-n1319-verl-strict-greedy-zero-control.json"
factor_cache="$REPO/$FACTOR_CACHE_REL"
steps=(16 32 48 64)

if [[ -z "$REGISTRY_RUN_TAG" ]]; then
  REGISTRY_RUN_TAG=${RUN_SLUG,,}
fi
[[ "$REGISTRY_RUN_TAG" =~ ^[a-z0-9][a-z0-9._-]*$ ]] ||
  refuse 'REGISTRY_RUN_TAG must be one lowercase registry-label segment'

label_prefix="qwen2.5-7b/gsm8k/tinylora13-verl-strict-canary-lr1e-4-s42/$REGISTRY_RUN_TAG"
zero_label="$label_prefix/zero-lora-r2-all196"
step_labels=()
for step in "${steps[@]}"; do
  step_labels+=("$label_prefix/step$step")
done
eval_name="gsm8k/test-n1319-verl-strict-greedy-zero-control-s42-canary"

[[ -d "$run_root" ]] || refuse "missing canary output root $run_root"
for relative in run_manifest.json metrics.jsonl trajectories.jsonl \
  final_adapter/adapter_config.json final_adapter/adapter.safetensors; do
  require_file "$run_root/$relative"
done
require_file "$evaluation"
require_file "$factor_cache"
for step in "${steps[@]}"; do
  require_file "$run_root/checkpoint-$step/adapter_config.json"
  require_file "$run_root/checkpoint-$step/adapter.safetensors"
  require_file "$artifact_root/step$step/adapter_config.json"
  require_file "$artifact_root/step$step/adapter_model.safetensors"
done
require_file "$artifact_root/zero/adapter_config.json"
require_file "$artifact_root/zero/adapter_model.safetensors"

# Validate the completed training run, native/PEFT equivalence, exact zero,
# full-test evaluation provenance, per-example summaries, paired transitions,
# repeatability diagnostics, Holm family, and every evaluation artifact hash.
python3 - \
  "$run_root" "$evaluation" "$factor_cache" "$EXPECTED_FACTOR_SHA256" \
  "$MODEL" "$MODEL_REVISION" "$DATASET_REVISION" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from tinylora_rl.adapter_analysis import load_adapter_updates
from tinylora_rl.registry import _verify_peft_companion
from tinylora_rl.rewards import predicted_answer, strict_predicted_answer


(
    root_arg,
    evaluation_arg,
    factor_arg,
    expected_factor_hash,
    model,
    model_revision,
    dataset_revision,
) = sys.argv[1:]
root = Path(root_arg)
evaluation_path = Path(evaluation_arg)
factor_path = Path(factor_arg)
steps = (16, 32, 48, 64)
selectors = [f"step{step}" for step in steps]
hex64 = re.compile(r"[0-9a-f]{64}")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        artifact = path / filename
        result[filename] = {
            "sha256": sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
    return result


if sha256(factor_path) != expected_factor_hash:
    fail("factor cache no longer matches the canary's pinned SHA-256")

manifest = read_json(root / "run_manifest.json")
learner = manifest.get("learner")
rollout = manifest.get("rollout")
dataset = manifest.get("dataset")
train = manifest.get("train_config")
adapter = manifest.get("adapter_config")
if manifest.get("schema_version") != 3 or not all(
    isinstance(value, dict) for value in (learner, rollout, dataset, train, adapter)
):
    fail("training run lacks schema-3 structured provenance")
if learner.get("model_id") != model or learner.get("requested_revision") != model_revision or learner.get("resolved_revision") not in (None, model_revision):
    fail("learner is not bound to the pinned Qwen commit")
if rollout.get("model_id") != model or rollout.get("requested_revision") != model_revision or rollout.get("resolved_revision") not in (None, model_revision) or rollout.get("sync_mode") != "lora":
    fail("rollout is not bound to the pinned Qwen LoRA path")
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
        fail(f"training field {key!r} differs from the canary protocol")
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
    adapter.get("scaling") != 1.0,
    adapter.get("parameter_dtype") != "bfloat16",
    manifest.get("target_layers") != 196,
    manifest.get("trainable_parameters") != 13,
)):
    fail("training adapter is not the 13-parameter all-linear canary")
if manifest.get("prompt") != {"style": "verl"} or manifest.get("reward") != {
    "mode": "strict",
    "implementation": "verl_gsm8k_strict_last_300_chars_v1",
}:
    fail("training did not use the VERL prompt and canonical strict scorer")

metrics = read_jsonl(root / "metrics.jsonl")
if len(metrics) != 64 or [row.get("step") for row in metrics] != list(range(1, 65)):
    fail("training metrics do not prove exactly 64 optimizer steps")
for row in metrics:
    if row.get("synced_weights") != 196:
        fail(f"step {row.get('step')} did not synchronize all 196 matrices")
    for key in ("grad_norm", "adapter_norm", "step_seconds"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            fail(f"step {row.get('step')} has invalid {key}")
    if row["adapter_norm"] <= 0 or row["step_seconds"] <= 0:
        fail(f"step {row.get('step')} lacks a completed update")
trajectories = read_jsonl(root / "trajectories.jsonl")
if len(trajectories) != 4096:
    fail("training trajectory count is not 64*16*4")
for step in range(1, 65):
    selected = [row for row in trajectories if row.get("step") == step]
    groups = [row.get("group_id") for row in selected]
    if len(selected) != 64 or sorted(groups) != [group for group in range(16) for _ in range(4)]:
        fail(f"training step {step} has an invalid prompt/generation design")

native_configs = {}
native_tensors = {}
source_bindings = {}
for step in steps:
    native = root / f"checkpoint-{step}"
    peft = root / "evaluation_adapters" / f"step{step}"
    native_config = read_json(native / "adapter_config.json")
    modules = native_config.get("modules")
    if native_config.get("config") != adapter or not isinstance(modules, list) or len(modules) != 196:
        fail(f"checkpoint-{step} has incompatible native metadata")
    tensors = load_file(str(native / "adapter.safetensors"))
    bank = tensors.get("bank.v")
    if bank is None or tuple(bank.shape) != (13, 1) or bank.dtype != torch.bfloat16:
        fail(f"checkpoint-{step} has the wrong parameter bank")
    _verify_peft_companion(native, peft, base_model=model)
    native_configs[step] = native_config
    native_tensors[step] = tensors
    source_bindings[f"step{step}"] = binding(peft)

final_config = read_json(root / "final_adapter" / "adapter_config.json")
final_tensors = load_file(str(root / "final_adapter" / "adapter.safetensors"))
if final_config != native_configs[64] or final_tensors.keys() != native_tensors[64].keys():
    fail("final_adapter and checkpoint-64 metadata/tensor sets differ")
for name in final_tensors:
    if not torch.equal(final_tensors[name], native_tensors[64][name]):
        fail(f"final_adapter and checkpoint-64 differ at {name}")

zero_path = root / "evaluation_adapters" / "zero"
zero_updates = {update.name: update for update in load_adapter_updates(zero_path)}
source_updates = {
    update.name: update for update in load_adapter_updates(root / "checkpoint-16")
}
if len(zero_updates) != 196 or zero_updates.keys() != source_updates.keys():
    fail("zero control and checkpoint-16 have different module sets")
for name, update in zero_updates.items():
    if torch.count_nonzero(update.left).item() != 0:
        fail(f"zero control has a nonzero left factor for {name}")
    if not torch.equal(update.right, source_updates[name].right):
        fail(f"zero control right factor differs for {name}")
source_bindings["zero"] = binding(zero_path)

evaluation = read_json(evaluation_path)
expected_top = {
    "schema_version": 3,
    "metric": "gsm8k_greedy_exact_match",
    "model": model,
    "revision": model_revision,
    "requested_revision": model_revision,
    "dataset": "openai/gsm8k",
    "dataset_config": "main",
    "dataset_revision": dataset_revision,
    "source_dataset_rows": 1319,
    "selected_dataset_rows": 1319,
    "require_exact_samples": True,
    "split": "test",
    "shuffle_seed": None,
    "prompt_style": "verl",
    "score_mode": "strict",
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "bootstrap_samples": 10000,
    "bootstrap_seed": 42,
    "max_tokens": 512,
    "max_model_length": 1024,
    "gpu_memory_utilization": 0.5,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    "kv_cache_dtype": "auto",
    "moe_backend": None,
    "mamba_backend": None,
    "mamba_cache_mode": None,
    "trust_remote_code": False,
    "comparison_baseline": "zero",
    "holm_family_labels": selectors,
}
for key, expected in expected_top.items():
    if evaluation.get(key) != expected:
        fail(f"evaluation field {key!r} is {evaluation.get(key)!r}, expected {expected!r}")
if evaluation.get("score_implementations", {}).get("strict") != "verl_gsm8k_strict_last_300_chars_v1":
    fail("evaluation does not name the canonical VERL strict scorer")
for key in ("source_dataset_fingerprint", "selected_dataset_fingerprint", "created_at"):
    if not isinstance(evaluation.get(key), str) or not evaluation[key]:
        fail(f"evaluation lacks provenance field {key!r}")
if not isinstance(evaluation.get("software_versions"), dict) or not evaluation["software_versions"]:
    fail("evaluation lacks software-version provenance")
for key in ("question_sha256", "answer_sha256"):
    values = evaluation.get(key)
    if not isinstance(values, list) or len(values) != 1319 or any(
        not isinstance(value, str) or hex64.fullmatch(value) is None for value in values
    ):
        fail(f"evaluation has invalid {key}")
if not isinstance(evaluation.get("prompt_set_sha256"), str) or hex64.fullmatch(evaluation["prompt_set_sha256"]) is None:
    fail("evaluation has an invalid prompt-set hash")


def validate_record(name: str, record: object) -> dict[str, list[bool]]:
    if not isinstance(record, dict):
        fail(f"evaluation record {name!r} is not an object")
    details = record.get("details")
    if record.get("samples") != 1319 or not isinstance(details, list) or len(details) != 1319:
        fail(f"evaluation record {name!r} lacks 1319 details")
    strict_vector = []
    flexible_vector = []
    for index, detail in enumerate(details):
        if not isinstance(detail, dict) or detail.get("index") != index:
            fail(f"evaluation record {name!r} has malformed detail ordering")
        completion = detail.get("completion")
        if not isinstance(completion, str) or hashlib.sha256(completion.encode()).hexdigest() != detail.get("completion_sha256"):
            fail(f"evaluation record {name!r} has an invalid completion hash")
        if not isinstance(detail.get("completion_token_ids_sha256"), str) or hex64.fullmatch(detail["completion_token_ids_sha256"]) is None:
            fail(f"evaluation record {name!r} has an invalid token-ID hash")
        strict_prediction = strict_predicted_answer(completion)
        flexible_prediction = predicted_answer(completion)
        if detail.get("strict_prediction") != strict_prediction or detail.get("flexible_prediction") != flexible_prediction:
            fail(f"evaluation record {name!r} prediction extraction disagrees at row {index}")
        strict_correct = strict_prediction is not None and strict_prediction == detail.get("strict_gold")
        flexible_correct = flexible_prediction is not None and flexible_prediction == detail.get("flexible_gold")
        if detail.get("strict_correct") is not strict_correct or detail.get("flexible_correct") is not flexible_correct:
            fail(f"evaluation record {name!r} correctness disagrees at row {index}")
        strict_vector.append(strict_correct)
        flexible_vector.append(flexible_correct)
    strict_count = sum(strict_vector)
    flexible_count = sum(flexible_vector)
    if any((
        record.get("score_mode") != "strict",
        record.get("correct") != strict_count,
        record.get("strict_correct") != strict_count,
        record.get("flexible_correct") != flexible_count,
        not math.isclose(float(record.get("accuracy", -1)), strict_count / 1319, abs_tol=1e-15),
        not math.isclose(float(record.get("strict_accuracy", -1)), strict_count / 1319, abs_tol=1e-15),
        not math.isclose(float(record.get("flexible_accuracy", -1)), flexible_count / 1319, abs_tol=1e-15),
    )):
        fail(f"evaluation record {name!r} summary disagrees with details")
    return {"strict": strict_vector, "flexible": flexible_vector}


correctness = {"base": validate_record("base", evaluation.get("base"))}
candidates = evaluation.get("candidates")
expected_candidates = ["zero", "zero_repeat", *selectors]
if not isinstance(candidates, dict) or list(candidates) != expected_candidates:
    fail("evaluation candidate order differs from zero, zero_repeat, step16/32/48/64")
for selector in expected_candidates:
    record = candidates[selector]
    source = "zero" if selector == "zero_repeat" else selector
    correctness[selector] = validate_record(selector, record)
    if record.get("adapter_artifacts") != source_bindings[source]:
        fail(f"evaluation candidate {selector!r} is not hash-bound to its PEFT source")

comparisons = evaluation.get("comparisons")
expected_comparisons = {"base", "zero_repeat", *selectors}
if not isinstance(comparisons, dict) or set(comparisons) != expected_comparisons:
    fail("evaluation comparisons do not cover every non-baseline path")
zero_correct = correctness["zero"]


def expected_pair(before: list[bool], after: list[bool]) -> dict[str, float | int]:
    wrong_to_right = sum(
        not old and new for old, new in zip(before, after, strict=True)
    )
    right_to_wrong = sum(
        old and not new for old, new in zip(before, after, strict=True)
    )
    discordant = wrong_to_right + right_to_wrong
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(wrong_to_right, right_to_wrong) + 1)
        ) / (2**discordant)
        mcnemar = min(1.0, 2.0 * tail)
    else:
        mcnemar = 1.0
    return {
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "accuracy_delta": (wrong_to_right - right_to_wrong) / len(before),
        "mcnemar_exact_p": mcnemar,
    }


strict_p_values = {}
for selector, comparison in comparisons.items():
    candidate_correct = correctness[selector]
    paired_by_mode = comparison.get("paired_by_score_mode")
    if comparison.get("score_mode") != "strict" or not isinstance(paired_by_mode, dict):
        fail(f"paired comparison {selector!r} lacks score-mode statistics")
    for mode in ("strict", "flexible"):
        recorded = paired_by_mode.get(mode)
        if not isinstance(recorded, dict):
            fail(f"paired comparison {selector!r} lacks {mode} statistics")
        expected = expected_pair(zero_correct[mode], candidate_correct[mode])
        for key in ("wrong_to_right", "right_to_wrong"):
            if recorded.get(key) != expected[key]:
                fail(f"paired comparison {selector!r} has the wrong {mode} {key}")
        for key in ("accuracy_delta", "mcnemar_exact_p"):
            if not isinstance(recorded.get(key), (int, float)) or not math.isclose(
                float(recorded[key]), float(expected[key]), abs_tol=1e-15
            ):
                fail(f"paired comparison {selector!r} has the wrong {mode} {key}")
        interval = recorded.get("paired_bootstrap_95_ci")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) for value in interval)
            or not (-1.0 <= interval[0] <= interval[1] <= 1.0)
        ):
            fail(f"comparison {selector!r} lacks a valid {mode} bootstrap interval")
    strict_expected = expected_pair(zero_correct["strict"], candidate_correct["strict"])
    for key in ("wrong_to_right", "right_to_wrong"):
        if comparison.get(key) != strict_expected[key]:
            fail(f"top-level paired comparison {selector!r} has the wrong {key}")
    for key in ("accuracy_delta", "mcnemar_exact_p"):
        if not isinstance(comparison.get(key), (int, float)) or not math.isclose(
            float(comparison[key]), float(strict_expected[key]), abs_tol=1e-15
        ):
            fail(f"top-level paired comparison {selector!r} has the wrong {key}")
    if comparison.get("paired_bootstrap_95_ci") != paired_by_mode["strict"].get(
        "paired_bootstrap_95_ci"
    ):
        fail(f"top-level paired comparison {selector!r} disagrees with strict statistics")
    strict_p_values[selector] = float(comparison["mcnemar_exact_p"])
    reference_details = candidates["zero"]["details"]
    candidate_details = (evaluation["base"] if selector == "base" else candidates[selector])["details"]
    identical = sum(
        before["completion_sha256"] == after["completion_sha256"]
        and before["completion_token_ids_sha256"] == after["completion_token_ids_sha256"]
        and before["finish_reason"] == after["finish_reason"]
        for before, after in zip(reference_details, candidate_details, strict=True)
    )
    if comparison.get("identical_generations") != identical or not math.isclose(float(comparison.get("identical_generation_rate", -1)), identical / 1319, abs_tol=1e-15):
        fail(f"generation-match diagnostic {selector!r} disagrees with details")
    if selector in selectors:
        if comparison.get("holm_family_member") is not True or comparison.get("multiple_comparison_family_size") != 4 or not isinstance(comparison.get("mcnemar_holm_p"), (int, float)):
            fail(f"trained comparison {selector!r} is outside the four-way Holm family")
    elif comparison.get("holm_family_member") is not False or "mcnemar_holm_p" in comparison:
        fail(f"control comparison {selector!r} was incorrectly included in Holm correction")

ordered = sorted((strict_p_values[selector], selector) for selector in selectors)
running = 0.0
for index, (p_value, selector) in enumerate(ordered):
    running = max(running, min(1.0, (len(ordered) - index) * p_value))
    if not math.isclose(float(comparisons[selector]["mcnemar_holm_p"]), running, abs_tol=1e-15):
        fail(f"comparison {selector!r} has an invalid Holm-adjusted p-value")

print("validated Qwen canary, checkpoint/final dedup, full-test evaluation, and artifact bindings")
PY

registry_cli=(python3 -m tinylora_rl.registry)

register_all() {
  local destination_registry=$1
  "${registry_cli[@]}" --registry "$destination_registry" add \
    "$artifact_root/zero" \
    --label "$zero_label" \
    --base-model "$MODEL" \
    --base-revision "$MODEL_REVISION" >/dev/null

  for index in "${!steps[@]}"; do
    step=${steps[$index]}
    "${registry_cli[@]}" --registry "$destination_registry" add \
      "$run_root/checkpoint-$step" \
      --peft "$artifact_root/step$step" \
      --label "${step_labels[$index]}" \
      --base-model "$MODEL" \
      --base-revision "$MODEL_REVISION" >/dev/null
  done

  "${registry_cli[@]}" --registry "$destination_registry" add-eval \
    "$zero_label" "$evaluation" --name "$eval_name" --candidate zero >/dev/null
  for index in "${!steps[@]}"; do
    step=${steps[$index]}
    "${registry_cli[@]}" --registry "$destination_registry" add-eval \
      "${step_labels[$index]}" "$evaluation" \
      --name "$eval_name" --candidate "step$step" >/dev/null
  done
}

temporary_parent="${TMPDIR:-/tmp}"
temporary_registry=$(mktemp -d "$temporary_parent/qwen25-registry-preflight.XXXXXX")
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${temporary_registry:-}" && -d "$temporary_registry" &&
        "$(basename "$temporary_registry")" == qwen25-registry-preflight.* ]]; then
    chmod -R u+rwX "$temporary_registry" 2>/dev/null || true
    rm -rf -- "$temporary_registry"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Dry promotion computes exact immutable IDs.  Comparing them makes retries
# idempotent while refusing to move any existing human label or eval name.
register_all "$temporary_registry"
all_labels=("$zero_label" "${step_labels[@]}")

check_registry_state() {
  local require_present=$1
  python3 - \
    "$temporary_registry" "$REGISTRY" "$require_present" "$eval_name" \
    "${all_labels[@]}" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath


expected_root = Path(sys.argv[1])
actual_root = Path(sys.argv[2])
require_present = sys.argv[3] == "1"
evaluation_name = sys.argv[4]
labels = sys.argv[5:]


def ref_path(root: Path, label: str) -> Path:
    parts = PurePosixPath(label).parts
    return root / "refs" / Path(*parts[:-1], f"{parts[-1]}.json")


def eval_ref_path(root: Path, object_id: str) -> Path:
    parts = PurePosixPath(evaluation_name).parts
    digest = object_id.removeprefix("sha256:")
    return root / "evaluation_refs" / digest / Path(*parts[:-1], f"{parts[-1]}.json")


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
            raise RuntimeError(f"malformed registry reference at {actual_ref_path}")
        if actual_ref.get("object_id") != expected_id:
            raise RuntimeError(
                f"label {label!r} already points to {actual_ref.get('object_id')}, "
                f"but this run has {expected_id}; refusing to move it"
            )
        actual_object = actual_root / "objects" / "sha256" / expected_id.removeprefix("sha256:")
        if not actual_object.is_dir():
            raise RuntimeError(f"label {label!r} points to missing object {expected_id}")

    expected_eval_path = eval_ref_path(expected_root, expected_id)
    if not expected_eval_path.exists():
        continue
    expected_eval = json.loads(expected_eval_path.read_text())
    actual_eval_path = eval_ref_path(actual_root, expected_id)
    if not actual_eval_path.exists():
        if require_present:
            raise RuntimeError(f"promotion did not attach {evaluation_name!r} to {label!r}")
        continue
    actual_eval = json.loads(actual_eval_path.read_text())
    if actual_eval.get("adapter_object_id") != expected_id or actual_eval.get("name") != evaluation_name:
        raise RuntimeError(f"malformed evaluation reference at {actual_eval_path}")
    if actual_eval.get("evaluation_id") != expected_eval.get("evaluation_id"):
        raise RuntimeError(
            f"evaluation name {evaluation_name!r} on {label!r} already points to "
            f"{actual_eval.get('evaluation_id')}, expected {expected_eval.get('evaluation_id')}; "
            "refusing to move it"
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
  printf 'Evaluation name: %s\n' "$eval_name"
  printf 'checkpoint-64 is the canonical step-64 object; duplicate final_adapter is not separately registered.\n'
  exit 0
fi

register_all "$REGISTRY"
check_registry_state 1
for label in "${all_labels[@]}"; do
  "${registry_cli[@]}" --registry "$REGISTRY" verify "$label" >/dev/null
done

printf 'Promoted and hash-verified five immutable Qwen canary adapter labels.\n'
printf 'Attached and verified %s on the exact zero control and four checkpoints.\n' "$eval_name"
printf 'checkpoint-64 and final_adapter were proven equal; only checkpoint-64 was registered.\n'
