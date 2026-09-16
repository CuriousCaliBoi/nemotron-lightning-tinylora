#!/usr/bin/env python3
"""Compare one or more LoRA-style adapters with the paper's SVD metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from tinylora_rl.adapter_analysis import (
    LowRankUpdate,
    find_intruder_dimensions_from_reference,
    load_adapter_updates,
    prepare_spectral_reference,
)


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_MODEL_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("adapters must use LABEL=PATH")
    return label, Path(raw_path)


def adapter_artifacts(path: Path) -> dict[str, dict[str, object]]:
    """Hash the exact adapter metadata and tensor file used by the analysis."""

    config_path = path / "adapter_config.json"
    tensor_candidates = (path / "adapter.safetensors", path / "adapter_model.safetensors")
    tensor_paths = [candidate for candidate in tensor_candidates if candidate.is_file()]
    if not config_path.is_file() or len(tensor_paths) != 1:
        raise FileNotFoundError(
            f"{path} must contain adapter_config.json and exactly one supported tensor file"
        )
    records: dict[str, dict[str, object]] = {}
    for artifact in (config_path, tensor_paths[0]):
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records[artifact.name] = {
            "sha256": digest.hexdigest(),
            "size_bytes": artifact.stat().st_size,
        }
    return records


def comma_separated_strings(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def comma_separated_ints(value: str | None) -> set[int] | None:
    parsed = comma_separated_strings(value)
    return {int(item) for item in parsed} if parsed is not None else None


def resolve_model_dtype(value: str) -> torch.dtype:
    """Resolve the storage dtype used while holding the base model in memory."""

    try:
        return _MODEL_DTYPES[value]
    except KeyError as exc:
        choices = ", ".join(_MODEL_DTYPES)
        raise ValueError(f"unsupported model dtype {value!r}; choose one of: {choices}") from exc


def selected(name: str, module_types: set[str] | None, layer_indices: set[int] | None) -> bool:
    if module_types is not None and name.rsplit(".", 1)[-1] not in module_types:
        return False
    if layer_indices is not None:
        match = _LAYER_RE.search(name)
        if match is None or int(match.group(1)) not in layer_indices:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument(
        "--model-dtype",
        choices=tuple(_MODEL_DTYPES),
        default="float32",
        help=(
            "Storage dtype for the loaded base model. Each selected matrix is still "
            "converted to float32 before its SVD."
        ),
    )
    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_labelled_path,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--module-types", help="Optional comma-separated module suffixes")
    parser.add_argument("--layer-indices", help="Optional comma-separated transformer layers")
    parser.add_argument(
        "--universe",
        choices=("intersection", "union"),
        default="intersection",
        help="Fixed matrix universe used for every candidate",
    )
    parser.add_argument(
        "--require-identical-matrices",
        action="store_true",
        help="Fail unless every adapter contains exactly the same selected matrices",
    )
    parser.add_argument(
        "--expected-matrices",
        type=int,
        help="Fail unless every adapter contains exactly this many selected matrices",
    )
    parser.add_argument(
        "--svd-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def summary(per_matrix: list[dict[str, object]]) -> dict[str, object]:
    counts = [int(item["intruder_count"]) for item in per_matrix]
    examined = [int(item["examined"]) for item in per_matrix]
    similarities = [
        float(value)
        for item in per_matrix
        for value in item["max_similarities"]  # type: ignore[index]
    ]
    update_ratios = [float(item["update_frobenius_ratio"]) for item in per_matrix]
    total_examined = sum(examined)
    return {
        "matrices": len(per_matrix),
        "matrices_with_intruders": sum(count > 0 for count in counts),
        "intruder_dimensions": sum(counts),
        "examined_dimensions": total_examined,
        "intruder_rate": sum(counts) / total_examined if total_examined else 0.0,
        "mean_max_similarity": statistics.fmean(similarities) if similarities else None,
        "minimum_max_similarity": min(similarities) if similarities else None,
        "mean_update_frobenius_ratio": statistics.fmean(update_ratios) if update_ratios else None,
        "maximum_update_frobenius_ratio": max(update_ratios) if update_ratios else None,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing analysis {output}; pass --overwrite explicitly"
        )
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if args.expected_matrices is not None and args.expected_matrices < 1:
        raise ValueError("expected-matrices must be positive")
    labels = [label for label, _ in args.adapter]
    if len(labels) != len(set(labels)):
        raise ValueError("adapter labels must be unique")
    module_types = comma_separated_strings(args.module_types)
    layer_indices = comma_separated_ints(args.layer_indices)
    updates_by_label: dict[str, dict[str, LowRankUpdate]] = {}
    paths: dict[str, str] = {}
    artifacts: dict[str, dict[str, dict[str, object]]] = {}
    for label, path in args.adapter:
        selected_updates = [
            update
            for update in load_adapter_updates(path)
            if selected(update.name, module_types, layer_indices)
        ]
        updates = {update.name: update for update in selected_updates}
        if len(updates) != len(selected_updates):
            raise ValueError(f"duplicate selected matrix names found for {label}")
        if not updates:
            raise ValueError(f"no selected updates found for {label}")
        if args.expected_matrices is not None and len(updates) != args.expected_matrices:
            raise ValueError(
                f"{label} has {len(updates)} selected matrices, expected "
                f"{args.expected_matrices}"
            )
        updates_by_label[label] = updates
        paths[label] = str(path)
        artifacts[label] = adapter_artifacts(path)

    name_sets = [set(updates) for updates in updates_by_label.values()]
    if args.require_identical_matrices:
        reference_names = name_sets[0]
        for label, names in zip(labels[1:], name_sets[1:], strict=True):
            if names != reference_names:
                raise ValueError(
                    f"selected matrix set for {label} differs from {labels[0]}: "
                    f"missing={sorted(reference_names - names)}, "
                    f"extra={sorted(names - reference_names)}"
                )
    if args.universe == "intersection":
        matrix_names = set.intersection(*name_sets)
    else:
        matrix_names = set.union(*name_sets)
    if not matrix_names:
        raise ValueError("the selected fixed matrix universe is empty")

    model_dtype = resolve_model_dtype(args.model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        trust_remote_code=args.trust_remote_code,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    details: dict[str, list[dict[str, object]]] = {label: [] for label in labels}
    started = time.perf_counter()
    for matrix_index, name in enumerate(sorted(matrix_names), start=1):
        module = model.get_submodule(name)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError(f"{name} is not a matrix-bearing module")
        reference = prepare_spectral_reference(weight, device=args.svd_device)
        for label in labels:
            update = updates_by_label[label].get(name)
            if update is None:
                # Union mode treats an absent adapter matrix as an exact no-op,
                # making totals comparable over a fixed denominator.
                tuned = reference.weight
                rank = 0
                source = "missing/no-op"
            else:
                tuned = reference.weight + update.materialize(
                    device=reference.weight.device,
                    dtype=torch.float32,
                )
                rank = update.rank
                source = update.source
            result = find_intruder_dimensions_from_reference(
                reference,
                tuned,
                threshold=args.threshold,
                top_k=args.top_k,
            )
            details[label].append(
                {
                    "name": name,
                    "shape": list(weight.shape),
                    "adapter_rank": rank,
                    "source": source,
                    "intruder_count": result.count,
                    "examined": result.examined,
                    "max_similarities": list(result.max_similarities),
                    "base_singular_values": list(result.base_singular_values),
                    "tuned_singular_values": list(result.tuned_singular_values),
                    "update_frobenius_ratio": result.update_frobenius_ratio,
                }
            )
        print(
            json.dumps(
                {"matrix": matrix_index, "total": len(matrix_names), "name": name}
            ),
            flush=True,
        )

    result = {
        "schema_version": 2,
        "metric": "exact_left_singular_vector_intruder_dimensions",
        "metric_definition_matches_paper": True,
        "experiment_design_matches_paper": False,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "model_dtype": args.model_dtype,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "svd_dtype": "float32",
        "svd_device": args.svd_device,
        "universe": args.universe,
        "require_identical_matrices": args.require_identical_matrices,
        "expected_matrices": args.expected_matrices,
        "matrix_names": sorted(matrix_names),
        "module_types": sorted(module_types) if module_types is not None else None,
        "layer_indices": sorted(layer_indices) if layer_indices is not None else None,
        "elapsed_seconds": time.perf_counter() - started,
        "candidates": {
            label: {
                "adapter": paths[label],
                "adapter_artifacts": artifacts[label],
                "available_matrices": len(updates_by_label[label]),
                "summary": summary(details[label]),
                "per_matrix": details[label],
            }
            for label in labels
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({label: result["candidates"][label]["summary"] for label in labels}, indent=2))


if __name__ == "__main__":
    main()
