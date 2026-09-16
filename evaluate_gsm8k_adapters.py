#!/usr/bin/env python3
"""Evaluate a base model and multiple PEFT-compatible adapters identically."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tinylora_rl.prompts import gsm8k_messages
from tinylora_rl.rewards import (
    predicted_answer,
    reference_answer,
    strict_predicted_answer,
    strict_reference_answer,
)


EVALUATOR_PROGRAM = "evaluate_gsm8k_adapters.py"
EVALUATOR_PROVENANCE_SCHEMA_VERSION = 1


def parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("adapters must use LABEL=PATH")
    return label, Path(raw_path)


def parse_repeated_contrast(value: str) -> tuple[str, str, str, str]:
    labels = tuple(item.strip() for item in value.split(","))
    if len(labels) != 4 or any(not label for label in labels):
        raise argparse.ArgumentTypeError(
            "repeated contrast must be ZERO_A,TRAINED_A,TRAINED_B,ZERO_B"
        )
    if len(set(labels)) != 4:
        raise argparse.ArgumentTypeError("repeated contrast labels must be unique")
    return labels[0], labels[1], labels[2], labels[3]


def parse_environment_requirement(value: str) -> tuple[str, str]:
    name, separator, expected = value.partition("=")
    if (
        not separator
        or re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is None
        or not expected
    ):
        raise argparse.ArgumentTypeError(
            "environment requirements must use UPPER_CASE_NAME=VALUE"
        )
    return name, expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", action="append", type=parse_labelled_path, default=[])
    parser.add_argument(
        "--comparison-baseline",
        default="base",
        help=(
            "Correctness vector used for top-level paired comparisons. Use the label "
            "of a zero-valued adapter to control for vLLM's LoRA execution path."
        ),
    )
    parser.add_argument(
        "--holm-label",
        action="append",
        help=(
            "Adapter label to include in the Holm multiple-comparison family. "
            "Repeat for every trained candidate; controls such as base and "
            "zero_repeat should be omitted. By default every comparison is included."
        ),
    )
    parser.add_argument(
        "--no-holm",
        action="store_true",
        help="Declare that no ordinary pairwise comparison belongs to a Holm family.",
    )
    parser.add_argument(
        "--repeated-contrast",
        type=parse_repeated_contrast,
        metavar="ZERO_A,TRAINED_A,TRAINED_B,ZERO_B",
        help=(
            "Declare a two-by-two repeated contrast. Adapter order must exactly "
            "match these four labels and paths must repeat within each arm."
        ),
    )
    parser.add_argument(
        "--selection-rationale",
        help="Predeclared reason the trained arm was selected for confirmation.",
    )
    parser.add_argument(
        "--require-environment",
        action="append",
        type=parse_environment_requirement,
        default=[],
        metavar="NAME=VALUE",
        help="Require and record an exact runtime environment value.",
    )
    parser.add_argument(
        "--interleave-adapters",
        action="store_true",
        help=(
            "Generate every adapter for each question in one shared vLLM call. "
            "This balances scheduler/kernel history and places duplicate-zero "
            "stability controls in the same batch as trained candidates."
        ),
    )
    parser.add_argument(
        "--reuse-lora-slot",
        action="store_true",
        help=(
            "Evaluate adapters sequentially through one fixed LoRA ID, explicitly "
            "unloading it between arms. This controls slot-specific kernel effects."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--disjoint-evaluation",
        action="append",
        type=parse_labelled_path,
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Require the selected questions to be disjoint from a prior evaluation "
            "artifact and record that artifact's content hash. Repeat as needed."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--require-exact-samples",
        action="store_true",
        help="Fail unless --samples equals the total number of rows in the selected split.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-length", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--lora-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="Runtime LoRA dtype; 'auto' follows the base model dtype.",
    )
    parser.add_argument("--lora-target-modules")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--moe-backend")
    parser.add_argument("--mamba-backend")
    parser.add_argument("--mamba-cache-mode")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-style", choices=("concise", "verl"), default="concise")
    parser.add_argument("--score-mode", choices=("flexible", "strict"), default="flexible")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    return parser.parse_args()


def file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def evaluator_source_provenance(
    source_path: Path,
    arguments: list[str],
) -> dict[str, object]:
    """Describe the evaluator bytes loaded at entry without storing their path."""

    source = file_record(source_path)
    return {
        "schema_version": EVALUATOR_PROVENANCE_SCHEMA_VERSION,
        "program": EVALUATOR_PROGRAM,
        "source_sha256": source["sha256"],
        "source_size_bytes": source["size_bytes"],
        "arguments": list(arguments),
    }


def adapter_artifacts(path: Path) -> dict[str, dict[str, object]]:
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"adapter {path} is missing required files: {missing}")
    return {name: file_record(path / name) for name in required}


def disjoint_evaluation_record(
    path: Path,
    selected_question_sha256: list[str],
) -> dict[str, object]:
    """Validate and bind a prior evaluation used to establish sample independence."""

    if not path.is_file():
        raise FileNotFoundError(f"disjoint evaluation does not exist: {path}")
    try:
        source = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not parse disjoint evaluation {path}: {error}") from error
    prior_hashes = source.get("question_sha256")
    if not isinstance(prior_hashes, list) or not prior_hashes:
        raise ValueError(
            f"disjoint evaluation {path} has no non-empty question_sha256 list"
        )
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in prior_hashes
    ):
        raise ValueError(f"disjoint evaluation {path} contains invalid question hashes")
    overlap = sorted(set(prior_hashes) & set(selected_question_sha256))
    if overlap:
        raise ValueError(
            f"selected questions overlap disjoint evaluation {path}: "
            f"{len(overlap)} shared hashes"
        )
    return {
        "path": str(path),
        "artifact": file_record(path),
        "question_count": len(prior_hashes),
        "unique_question_count": len(set(prior_hashes)),
        "question_overlap_count": 0,
        "split": source.get("split"),
        "selected_dataset_rows": source.get("selected_dataset_rows"),
        "dataset_revision": source.get("dataset_revision"),
        "model": source.get("model"),
        "requested_revision": source.get("requested_revision"),
    }


def resolve_model_revision(model: str, requested: str | None) -> str | None:
    """Resolve the immutable Hub snapshot used by Transformers when possible."""

    config_path = cached_file(model, "config.json", revision=requested)
    match = re.search(r"[/\\]snapshots[/\\]([0-9a-f]{7,64})[/\\]", str(config_path))
    return match.group(1) if match else requested


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "datasets", "vllm", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def adapter_rank(path: Path) -> int:
    config = json.loads((path / "adapter_config.json").read_text())
    ranks = [int(config.get("r", 1))]
    ranks.extend(int(value) for value in (config.get("rank_pattern") or {}).values())
    return max(ranks)


def supported_max_lora_rank(required: int) -> int:
    for candidate in (1, 8, 16, 32, 64, 128, 256, 320, 512):
        if required <= candidate:
            return candidate
    raise ValueError(f"adapter rank {required} exceeds vLLM's supported maximum")


def row_cluster_bootstrap_interval(
    row_effects: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[float] | None:
    """Bootstrap rows, keeping every repeated outcome for a row in one cluster."""

    if not row_effects:
        raise ValueError("row effects must be non-empty")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap samples must be nonnegative")
    if bootstrap_samples == 0:
        return None
    generator = random.Random(seed)
    count = len(row_effects)
    samples = sorted(
        sum(row_effects[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(bootstrap_samples)
    )
    return [
        samples[int(0.025 * (bootstrap_samples - 1))],
        samples[int(0.975 * (bootstrap_samples - 1))],
    ]


def paired_statistics(
    before: list[bool],
    after: list[bool],
    *,
    bootstrap_samples: int,
    seed: int = 0,
) -> dict[str, object]:
    """Compute paired transitions, a bootstrap CI, and exact McNemar p-value."""

    if len(before) != len(after) or not before:
        raise ValueError("paired correctness vectors must be non-empty and equal-sized")
    deltas = [int(new) - int(old) for old, new in zip(before, after, strict=True)]
    wrong_to_right = sum(value == 1 for value in deltas)
    right_to_wrong = sum(value == -1 for value in deltas)
    discordant = wrong_to_right + right_to_wrong
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(wrong_to_right, right_to_wrong) + 1)
        ) / (2**discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0

    interval = row_cluster_bootstrap_interval(
        [float(value) for value in deltas],
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "accuracy_delta": sum(deltas) / len(deltas),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "paired_bootstrap_95_ci": interval,
        "mcnemar_exact_p": mcnemar_p,
    }


def compare_correctness(
    reference: dict[str, list[bool]],
    candidate: dict[str, list[bool]],
    *,
    score_mode: str,
    bootstrap_samples: int,
    seed: int = 0,
) -> dict[str, object]:
    """Summarize one candidate against an explicitly selected baseline."""

    by_mode = {
        mode: paired_statistics(
            reference[mode],
            candidate[mode],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for mode in ("flexible", "strict")
    }
    return {
        **by_mode[score_mode],
        "score_mode": score_mode,
        "paired_by_score_mode": by_mode,
    }


def add_holm_adjustment(
    comparisons: dict[str, dict[str, object]],
    labels: list[str] | None = None,
) -> None:
    """Add Holm-adjusted p-values for an explicit score-mode family in place."""

    family = list(comparisons) if labels is None else list(labels)
    if len(family) != len(set(family)):
        raise ValueError("Holm family labels must be unique")
    unknown = set(family) - set(comparisons)
    if unknown:
        raise ValueError(f"Holm family contains unknown comparisons: {sorted(unknown)}")
    for label, comparison in comparisons.items():
        comparison["holm_family_member"] = label in family

    ordered = sorted(
        (
            (float(comparison["mcnemar_exact_p"]), label)
            for label, comparison in comparisons.items()
            if label in family
        ),
        key=lambda item: item[0],
    )
    running = 0.0
    family_size = len(ordered)
    for index, (p_value, label) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - index) * p_value))
        comparisons[label]["mcnemar_holm_p"] = running
        comparisons[label]["multiple_comparison_family_size"] = family_size


def add_generation_match(
    comparison: dict[str, object],
    reference: dict[str, object],
    candidate: dict[str, object],
) -> None:
    """Record exact token/text/finish matches for repeatability diagnostics."""

    reference_details = reference["details"]
    candidate_details = candidate["details"]
    if not isinstance(reference_details, list) or not isinstance(candidate_details, list):
        raise TypeError("evaluation summaries must contain detail lists")
    if len(reference_details) != len(candidate_details) or not reference_details:
        raise ValueError("generation detail lists must be non-empty and equal-sized")
    matches = [
        (
            before["completion_sha256"] == after["completion_sha256"]
            and before["completion_token_ids_sha256"]
            == after["completion_token_ids_sha256"]
            and before["finish_reason"] == after["finish_reason"]
        )
        for before, after in zip(reference_details, candidate_details, strict=True)
    ]
    comparison["identical_generations"] = sum(matches)
    comparison["identical_generation_rate"] = sum(matches) / len(matches)


def within_arm_repeatability(
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, object]:
    """Compare two sequential evaluations of one exact adapter artifact."""

    first_details = first["details"]
    second_details = second["details"]
    if not isinstance(first_details, list) or not isinstance(second_details, list):
        raise TypeError("evaluation summaries must contain detail lists")
    if len(first_details) != len(second_details) or not first_details:
        raise ValueError("repeatability detail lists must be non-empty and equal-sized")
    if any(
        left.get("index") != right.get("index")
        for left, right in zip(first_details, second_details, strict=True)
    ):
        raise ValueError("repeatability details are not aligned to the same rows")
    count = len(first_details)
    text_matches = 0
    token_matches = 0
    finish_matches = 0
    joint_matches = 0
    correctness: dict[str, object] = {}
    prediction_identity: dict[str, object] = {}
    for mode in ("strict", "flexible"):
        first_only = 0
        second_only = 0
        same_prediction = 0
        for left, right in zip(first_details, second_details, strict=True):
            left_correct = bool(left[f"{mode}_correct"])
            right_correct = bool(right[f"{mode}_correct"])
            first_only += int(left_correct and not right_correct)
            second_only += int(right_correct and not left_correct)
            same_prediction += int(
                left[f"{mode}_prediction"] == right[f"{mode}_prediction"]
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
    for left, right in zip(first_details, second_details, strict=True):
        text_equal = left["completion_sha256"] == right["completion_sha256"]
        token_equal = (
            left["completion_token_ids_sha256"]
            == right["completion_token_ids_sha256"]
        )
        finish_equal = left["finish_reason"] == right["finish_reason"]
        text_matches += int(text_equal)
        token_matches += int(token_equal)
        finish_matches += int(finish_equal)
        joint_matches += int(text_equal and token_equal and finish_equal)
    return {
        "samples": count,
        "exact_text": {"count": text_matches, "rate": text_matches / count},
        "exact_token_ids": {"count": token_matches, "rate": token_matches / count},
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


def repeated_contrast_statistics(
    correctness_by_label: dict[str, dict[str, list[bool]]],
    summaries_by_label: dict[str, dict[str, object]],
    artifact_records: dict[str, dict[str, dict[str, object]]],
    adapter_paths: dict[str, Path],
    labels: tuple[str, str, str, str],
    *,
    bootstrap_samples: int,
    seed: int,
    selection_rationale: str,
) -> dict[str, object]:
    """Analyze one ABBA two-by-two repeated efficacy contrast."""

    zero_a, trained_a, trained_b, zero_b = labels
    by_score_mode: dict[str, object] = {}
    for mode in ("strict", "flexible"):
        za = correctness_by_label[zero_a][mode]
        ta = correctness_by_label[trained_a][mode]
        tb = correctness_by_label[trained_b][mode]
        zb = correctness_by_label[zero_b][mode]
        lengths = {len(values) for values in (za, ta, tb, zb)}
        if len(lengths) != 1 or not za:
            raise ValueError("repeated contrast vectors must be non-empty and equal-sized")
        row_effects = [
            (int(t_a) + int(t_b) - int(z_a) - int(z_b)) / 2.0
            for z_a, t_a, t_b, z_b in zip(za, ta, tb, zb, strict=True)
        ]
        by_score_mode[mode] = {
            "mean_effect": sum(row_effects) / len(row_effects),
            "row_effects": row_effects,
            "row_cluster_bootstrap_95_ci": row_cluster_bootstrap_interval(
                row_effects,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "repeat_specific": {
                "a_trained_minus_zero": paired_statistics(
                    za,
                    ta,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
                "b_trained_minus_zero": paired_statistics(
                    zb,
                    tb,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
            },
            "row_effect_counts": {
                str(value): row_effects.count(value)
                for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
            },
        }

    strict = by_score_mode["strict"]
    strict_ci = strict["row_cluster_bootstrap_95_ci"]
    repeat_a = strict["repeat_specific"]["a_trained_minus_zero"]["accuracy_delta"]
    repeat_b = strict["repeat_specific"]["b_trained_minus_zero"]["accuracy_delta"]
    gate_conditions = {
        "aggregate_strict_mean_gt_zero": strict["mean_effect"] > 0,
        "aggregate_strict_ci_lower_gt_zero": (
            strict_ci is not None and strict_ci[0] > 0
        ),
        "repeat_a_strict_estimate_gt_zero": repeat_a > 0,
        "repeat_b_strict_estimate_gt_zero": repeat_b > 0,
    }
    return {
        "schema_version": 1,
        "design": "two_by_two_repeated_abba_v1",
        "adapter_order": list(labels),
        "roles": {
            "zero_a": zero_a,
            "trained_a": trained_a,
            "trained_b": trained_b,
            "zero_b": zero_b,
        },
        "row_effect_formula": "((trained_a + trained_b) - (zero_a + zero_b)) / 2",
        "bootstrap_unit": "gsm8k_row_with_all_four_repeated_outcomes",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "primary_score_mode": "strict",
        "secondary_score_mode": "flexible",
        "predeclared_hypotheses": 1,
        "ordinary_pairwise_tests_are_diagnostic_only": True,
        "selection_rationale": selection_rationale,
        "within_arm_identity": {
            "zero": {
                "same_path": adapter_paths[zero_a] == adapter_paths[zero_b],
                "same_artifact_hashes": artifact_records[zero_a]
                == artifact_records[zero_b],
            },
            "trained": {
                "same_path": adapter_paths[trained_a] == adapter_paths[trained_b],
                "same_artifact_hashes": artifact_records[trained_a]
                == artifact_records[trained_b],
            },
        },
        "within_arm_repeatability": {
            "zero": within_arm_repeatability(
                summaries_by_label[zero_a], summaries_by_label[zero_b]
            ),
            "trained": within_arm_repeatability(
                summaries_by_label[trained_a], summaries_by_label[trained_b]
            ),
        },
        "by_score_mode": by_score_mode,
        "primary_efficacy_gate": {
            "conditions": gate_conditions,
            "passed": all(gate_conditions.values()),
        },
    }


def summarize(
    rows: object,
    outputs: object,
    *,
    base_correct: dict[str, list[bool]] | None,
    include_text: bool,
    score_mode: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, list[bool]]]:
    details = []
    correctness: dict[str, list[bool]] = {"flexible": [], "strict": []}
    for index, (row, request_output) in enumerate(zip(rows, outputs, strict=True)):
        completion = request_output.outputs[0]
        text = completion.text
        flexible_prediction = predicted_answer(text)
        strict_prediction = strict_predicted_answer(text)
        flexible_gold = reference_answer(row["answer"])
        strict_gold = strict_reference_answer(row["answer"])
        flexible_correct = (
            flexible_prediction is not None
            and flexible_gold is not None
            and flexible_prediction == flexible_gold
        )
        strict_correct = (
            strict_prediction is not None
            and strict_gold is not None
            and strict_prediction == strict_gold
        )
        correctness["flexible"].append(flexible_correct)
        correctness["strict"].append(strict_correct)
        prediction = flexible_prediction if score_mode == "flexible" else strict_prediction
        gold = flexible_gold if score_mode == "flexible" else strict_gold
        correct = flexible_correct if score_mode == "flexible" else strict_correct
        detail = {
            "index": index,
            "gold": gold,
            "flexible_gold": flexible_gold,
            "strict_gold": strict_gold,
            "prediction": prediction,
            "correct": correct,
            "flexible_prediction": flexible_prediction,
            "strict_prediction": strict_prediction,
            "flexible_correct": flexible_correct,
            "strict_correct": strict_correct,
            "strict_format": strict_prediction is not None,
            "finish_reason": completion.finish_reason,
            "completion_tokens": len(completion.token_ids),
            "completion_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "completion_token_ids_sha256": hashlib.sha256(
                json.dumps(list(completion.token_ids), separators=(",", ":")).encode()
            ).hexdigest(),
        }
        if include_text:
            detail["completion"] = text
        details.append(detail)
    summary: dict[str, object] = {
        "score_mode": score_mode,
        "correct": sum(correctness[score_mode]),
        "samples": len(correctness[score_mode]),
        "accuracy": sum(correctness[score_mode]) / len(correctness[score_mode]),
        "flexible_correct": sum(correctness["flexible"]),
        "flexible_accuracy": sum(correctness["flexible"]) / len(details),
        "strict_correct": sum(correctness["strict"]),
        "strict_accuracy": sum(correctness["strict"]) / len(details),
        "strict_format_rate": sum(bool(item["strict_format"]) for item in details) / len(details),
        "no_answer_rate": sum(item["prediction"] is None for item in details) / len(details),
        "length_clipped_rate": sum(item["finish_reason"] == "length" for item in details) / len(details),
        "mean_completion_tokens": sum(int(item["completion_tokens"]) for item in details)
        / len(details),
    }
    if base_correct is not None:
        summary.update(
            paired_statistics(
                base_correct[score_mode],
                correctness[score_mode],
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )
        summary["paired_by_score_mode"] = {
            mode: paired_statistics(
                base_correct[mode],
                correctness[mode],
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            for mode in ("flexible", "strict")
        }
    summary["details"] = details
    return summary, correctness


def main() -> None:
    evaluator_provenance = evaluator_source_provenance(
        Path(__file__).resolve(),
        sys.argv[1:],
    )
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap samples must be nonnegative")
    if args.max_tokens <= 0 or args.max_model_length <= 0:
        raise ValueError("token limits must be positive")
    if args.interleave_adapters and args.reuse_lora_slot:
        raise ValueError("interleave-adapters and reuse-lora-slot are mutually exclusive")
    if args.no_holm and args.holm_label is not None:
        raise ValueError("no-holm and holm-label are mutually exclusive")
    required_environment: dict[str, str] = {}
    for name, expected in args.require_environment:
        if name in required_environment:
            raise ValueError(f"duplicate required environment variable {name}")
        observed = os.environ.get(name)
        if observed != expected:
            raise ValueError(
                f"required environment {name}={expected!r}, observed {observed!r}"
            )
        required_environment[name] = observed
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing evaluation {output}; pass --overwrite explicitly"
        )
    labels = [label for label, _ in args.adapter]
    adapter_paths = dict(args.adapter)
    if len(labels) != len(set(labels)):
        raise ValueError("adapter labels must be unique")
    if "base" in labels:
        raise ValueError("'base' is reserved for the unadapted model")
    if args.comparison_baseline != "base" and args.comparison_baseline not in labels:
        raise ValueError(
            "comparison baseline must be 'base' or one of the supplied adapter labels"
        )
    if args.repeated_contrast is not None:
        zero_a, trained_a, trained_b, zero_b = args.repeated_contrast
        if labels != list(args.repeated_contrast):
            raise ValueError(
                "adapter order must exactly match the declared repeated contrast"
            )
        if args.comparison_baseline != zero_a:
            raise ValueError("repeated contrast baseline must be its zero_a label")
        if adapter_paths[zero_a] != adapter_paths[zero_b]:
            raise ValueError("repeated contrast zero arms must use the same path")
        if adapter_paths[trained_a] != adapter_paths[trained_b]:
            raise ValueError("repeated contrast trained arms must use the same path")
        if not args.reuse_lora_slot or args.interleave_adapters:
            raise ValueError(
                "repeated contrast requires sequential reuse of one LoRA slot"
            )
        if not args.no_holm:
            raise ValueError("repeated contrast is one hypothesis and requires --no-holm")
        if args.bootstrap_samples != 10_000:
            raise ValueError("repeated contrast requires exactly 10000 bootstrap samples")
        if args.score_mode != "strict":
            raise ValueError("repeated contrast primary score mode must be strict")
        if not args.selection_rationale or not args.selection_rationale.strip():
            raise ValueError("repeated contrast requires a selection rationale")
    disjoint_labels = [label for label, _ in args.disjoint_evaluation]
    if len(disjoint_labels) != len(set(disjoint_labels)):
        raise ValueError("disjoint evaluation labels must be unique")
    if args.holm_label is not None:
        if len(args.holm_label) != len(set(args.holm_label)):
            raise ValueError("Holm labels must be unique")
        invalid_holm_labels = set(args.holm_label) - set(labels)
        if invalid_holm_labels:
            raise ValueError(
                "Holm labels must be supplied adapter labels: "
                f"{sorted(invalid_holm_labels)}"
            )
        if args.comparison_baseline in args.holm_label:
            raise ValueError("the comparison baseline cannot be in the Holm family")
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=args.split,
        revision=args.dataset_revision,
    )
    source_dataset_rows = len(dataset)
    if args.require_exact_samples and source_dataset_rows != args.samples:
        raise ValueError(
            f"expected exactly {args.samples} rows in split {args.split!r}, "
            f"found {source_dataset_rows}"
        )
    source_dataset_fingerprint = dataset._fingerprint
    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed)
    rows = dataset.select(range(min(args.samples, len(dataset))))
    selected_dataset_fingerprint = rows._fingerprint
    question_sha256 = [
        hashlib.sha256(row["question"].encode()).hexdigest() for row in rows
    ]
    answer_sha256 = [
        hashlib.sha256(row["answer"].encode()).hexdigest() for row in rows
    ]
    disjoint_evaluations = {
        label: disjoint_evaluation_record(path, question_sha256)
        for label, path in args.disjoint_evaluation
    }
    resolved_revision = resolve_model_revision(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    prompts = [
        tokenizer.apply_chat_template(
            gsm8k_messages(row["question"], args.prompt_style),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    artifact_records: dict[str, dict[str, dict[str, object]]] = {}
    for label, path in args.adapter:
        config = json.loads((path / "adapter_config.json").read_text())
        declared_base = config.get("base_model_name_or_path")
        if declared_base and declared_base != args.model:
            raise ValueError(
                f"adapter {label!r} declares base {declared_base!r}, expected {args.model!r}"
            )
        artifact_records[label] = adapter_artifacts(path)
    if args.repeated_contrast is not None:
        zero_a, trained_a, trained_b, zero_b = args.repeated_contrast
        if artifact_records[zero_a] != artifact_records[zero_b]:
            raise RuntimeError("repeated zero arms do not have identical artifact hashes")
        if artifact_records[trained_a] != artifact_records[trained_b]:
            raise RuntimeError("repeated trained arms do not have identical artifact hashes")
    max_rank = supported_max_lora_rank(
        max([adapter_rank(path) for _, path in args.adapter], default=1)
    )
    engine_kwargs = {"kv_cache_dtype": args.kv_cache_dtype}
    for key, value in (
        ("moe_backend", args.moe_backend),
        ("mamba_backend", args.mamba_backend),
        ("mamba_cache_mode", args.mamba_cache_mode),
    ):
        if value is not None:
            engine_kwargs[key] = value
    if args.lora_target_modules:
        engine_kwargs["lora_target_modules"] = [
            item.strip() for item in args.lora_target_modules.split(",") if item.strip()
        ]
    max_loras = len(args.adapter) if args.interleave_adapters else 1
    max_cpu_loras = len(args.adapter) if args.interleave_adapters else 1
    engine = LLM(
        model=args.model,
        revision=args.revision,
        enable_lora=bool(args.adapter),
        max_lora_rank=max_rank,
        max_loras=max_loras,
        max_cpu_loras=max_cpu_loras,
        lora_dtype=args.lora_dtype,
        max_model_len=args.max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        **engine_kwargs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    base_outputs = engine.generate(prompts, sampling, use_tqdm=True)
    base, base_correct = summarize(
        rows,
        base_outputs,
        base_correct=None,
        include_text=args.include_text,
        score_mode=args.score_mode,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    candidates: dict[str, object] = {}
    correctness_by_label = {"base": base_correct}
    adapter_requests = []
    for adapter_id, (label, path) in enumerate(args.adapter, start=1):
        request_id = 1 if args.reuse_lora_slot else adapter_id
        request_name = "evaluation-slot" if args.reuse_lora_slot else label
        adapter_requests.append(
            (label, path, LoRARequest(request_name, request_id, str(path)))
        )
    outputs_by_label: dict[str, object] = {}
    if args.interleave_adapters:
        interleaved_prompts = []
        interleaved_requests = []
        for prompt in prompts:
            for _, _, request in adapter_requests:
                interleaved_prompts.append(prompt)
                interleaved_requests.append(request)
        interleaved_outputs = engine.generate(
            interleaved_prompts,
            sampling,
            lora_request=interleaved_requests,
            use_tqdm=True,
        )
        adapter_count = len(adapter_requests)
        for offset, (label, _, _) in enumerate(adapter_requests):
            outputs_by_label[label] = interleaved_outputs[offset::adapter_count]
    else:
        for request_index, (label, _, request) in enumerate(adapter_requests):
            outputs_by_label[label] = engine.generate(
                prompts,
                sampling,
                lora_request=request,
                use_tqdm=True,
            )
            if args.reuse_lora_slot and request_index + 1 < len(adapter_requests):
                removed = engine.llm_engine.remove_lora(request.lora_int_id)
                if not removed:
                    raise RuntimeError(
                        f"vLLM failed to unload reusable LoRA slot after {label!r}"
                    )

    for label, path, _ in adapter_requests:
        candidate, candidate_correct = summarize(
            rows,
            outputs_by_label[label],
            base_correct=base_correct,
            include_text=args.include_text,
            score_mode=args.score_mode,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        candidate["adapter"] = str(path)
        candidate["adapter_artifacts"] = artifact_records[label]
        candidates[label] = candidate
        correctness_by_label[label] = candidate_correct

    comparison_reference = correctness_by_label[args.comparison_baseline]
    comparisons = {
        label: compare_correctness(
            comparison_reference,
            correctness,
            score_mode=args.score_mode,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        for label, correctness in correctness_by_label.items()
        if label != args.comparison_baseline
    }
    summaries_by_label = {"base": base, **candidates}
    comparison_summary = summaries_by_label[args.comparison_baseline]
    for label, comparison in comparisons.items():
        add_generation_match(
            comparison,
            comparison_summary,
            summaries_by_label[label],
        )
    holm_labels = (
        []
        if args.no_holm
        else (list(comparisons) if args.holm_label is None else args.holm_label)
    )
    add_holm_adjustment(comparisons, holm_labels)
    repeated_contrast = None
    if args.repeated_contrast is not None:
        repeated_contrast = repeated_contrast_statistics(
            correctness_by_label,
            summaries_by_label,
            artifact_records,
            adapter_paths,
            args.repeated_contrast,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            selection_rationale=args.selection_rationale.strip(),
        )
        repeated_contrast["selection_evaluation_labels"] = list(
            disjoint_evaluations
        )

    result = {
        # Additive provenance/diagnostic fields remain backward-compatible with
        # schema 3 consumers used by the existing promotion tooling.
        "schema_version": 3,
        "evaluator_provenance": evaluator_provenance,
        "metric": (
            "gsm8k_greedy_exact_match"
            if args.temperature == 0.0
            else "gsm8k_sampled_exact_match"
        ),
        "model": args.model,
        "revision": resolved_revision,
        "requested_revision": args.revision,
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": args.dataset_revision,
        "disjoint_evaluations": disjoint_evaluations,
        "source_dataset_rows": source_dataset_rows,
        "selected_dataset_rows": len(rows),
        "require_exact_samples": args.require_exact_samples,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "selected_dataset_fingerprint": selected_dataset_fingerprint,
        "split": args.split,
        "shuffle_seed": args.shuffle_seed,
        "prompt_style": args.prompt_style,
        "score_mode": args.score_mode,
        "score_implementations": {
            "strict": "verl_gsm8k_strict_last_300_chars_v1",
            "flexible": "normalized_last_answer_v1",
        },
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "max_tokens": args.max_tokens,
        "max_model_length": args.max_model_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "lora_dtype": args.lora_dtype,
        "max_loras": max_loras,
        "max_cpu_loras": max_cpu_loras,
        "lora_target_modules": (
            [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
            if args.lora_target_modules
            else None
        ),
        "kv_cache_dtype": args.kv_cache_dtype,
        "moe_backend": args.moe_backend,
        "mamba_backend": args.mamba_backend,
        "mamba_cache_mode": args.mamba_cache_mode,
        "trust_remote_code": args.trust_remote_code,
        "comparison_baseline": args.comparison_baseline,
        "required_runtime_environment": required_environment,
        "adapter_execution_mode": (
            "question_interleaved_single_call"
            if args.interleave_adapters
            else (
                "sequential_reused_lora_slot"
                if args.reuse_lora_slot
                else "adapter_sequential_calls"
            )
        ),
        "holm_family_labels": holm_labels,
        "repeated_contrast": repeated_contrast,
        "question_sha256": question_sha256,
        "answer_sha256": answer_sha256,
        "prompt_set_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "software_versions": package_versions(),
        "base": base,
        "candidates": candidates,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    concise = {
        "base": {key: value for key, value in base.items() if key != "details"},
        "candidates": {
            label: {key: value for key, value in candidate.items() if key != "details"}
            for label, candidate in candidates.items()
        },
        "comparison_baseline": args.comparison_baseline,
        "comparisons": comparisons,
        "repeated_contrast": repeated_contrast,
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
