#!/usr/bin/env python3
"""Evaluate several compatible TinyLoRA checkpoints on a retention corpus.

The large BF16 learner and its frozen TinyLoRA factors are loaded once.  The
base score is obtained by zeroing the parameter bank, and candidates are
evaluated by swapping only their small bank and per-layer scaling values.  The
base weights are never merged or otherwise modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinylora_rl.adapters import iter_tinylora_layers, load_tinylora


@dataclass(frozen=True)
class CandidateState:
    """The mutable portion of one compatible TinyLoRA checkpoint."""

    path: Path
    bank: Tensor
    scalings: tuple[tuple[str, float], ...]
    factor_signature: str
    checkpoint_sha256: str

    @property
    def scaling_by_name(self) -> dict[str, float]:
        return dict(self.scalings)


@dataclass(frozen=True)
class CorpusSample:
    blocks: Tensor
    documents_consumed: int
    nonempty_documents: int
    token_sha256: str


def parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("adapters must use LABEL=PATH")
    return label, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    )
    parser.add_argument("--revision")
    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_labelled_path,
        required=True,
        help="Repeat LABEL=PATH for compatible native TinyLoRA checkpoints",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(tensor: Tensor) -> bytes:
    """Return dtype-agnostic raw bytes, including for CPU bfloat16 tensors."""

    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _factor_signature(metadata: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> str:
    config = dict(metadata.get("config") or {})
    config.pop("scaling", None)
    modules = metadata.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("TinyLoRA metadata must contain a non-empty modules list")
    immutable_modules = []
    for module in modules:
        if not isinstance(module, Mapping) or "name" not in module or "group_id" not in module:
            raise ValueError("invalid TinyLoRA module metadata")
        immutable = dict(module)
        immutable.pop("scaling", None)
        immutable_modules.append(immutable)

    digest = hashlib.sha256()
    canonical = {"config": config, "modules": immutable_modules}
    digest.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(key for key in tensors if key != "bank.v"):
        tensor = tensors[name]
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def load_candidate_state(path: str | Path) -> CandidateState:
    adapter_path = Path(path)
    config_path = adapter_path / "adapter_config.json"
    tensor_path = adapter_path / "adapter.safetensors"
    if not config_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError(
            f"{adapter_path} must contain adapter_config.json and adapter.safetensors"
        )
    metadata = json.loads(config_path.read_text())
    tensors = load_file(str(tensor_path), device="cpu")
    if "bank.v" not in tensors:
        raise ValueError(f"{tensor_path} has no bank.v tensor")
    bank = tensors["bank.v"].detach().clone()
    if bank.ndim != 2 or not torch.isfinite(bank.float()).all():
        raise ValueError(f"{tensor_path} contains an invalid parameter bank")

    config = metadata.get("config")
    modules = metadata.get("modules")
    if not isinstance(config, dict) or not isinstance(modules, list) or not modules:
        raise ValueError(f"{config_path} is not native TinyLoRA metadata")
    default_scaling = float(config.get("scaling", 1.0))
    scalings: list[tuple[str, float]] = []
    seen: set[str] = set()
    for module in modules:
        if not isinstance(module, Mapping) or "name" not in module or "group_id" not in module:
            raise ValueError(f"invalid TinyLoRA module metadata in {config_path}")
        name = str(module["name"])
        if name in seen:
            raise ValueError(f"duplicate TinyLoRA module {name!r} in {config_path}")
        seen.add(name)
        scaling = float(module.get("scaling", default_scaling))
        if not math.isfinite(scaling):
            raise ValueError(f"non-finite scaling for {name!r} in {config_path}")
        scalings.append((name, scaling))

    return CandidateState(
        path=adapter_path,
        bank=bank,
        scalings=tuple(scalings),
        factor_signature=_factor_signature(metadata, tensors),
        checkpoint_sha256=_sha256_files((config_path, tensor_path)),
    )


def validate_candidate_states(states: Iterable[CandidateState]) -> list[CandidateState]:
    candidates = list(states)
    if not candidates:
        raise ValueError("at least one TinyLoRA candidate is required")
    reference = candidates[0]
    for candidate in candidates[1:]:
        if candidate.factor_signature != reference.factor_signature:
            raise ValueError(
                f"{candidate.path} does not share TinyLoRA factors/config with "
                f"{reference.path}"
            )
        if candidate.bank.shape != reference.bank.shape:
            raise ValueError(
                f"parameter-bank shape mismatch: {candidate.path} has "
                f"{tuple(candidate.bank.shape)}, expected {tuple(reference.bank.shape)}"
            )
        if candidate.bank.dtype != reference.bank.dtype:
            raise ValueError(
                f"parameter-bank dtype mismatch: {candidate.path} has "
                f"{candidate.bank.dtype}, expected {reference.bank.dtype}"
            )
    return candidates


def _layer_map(model: nn.Module) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for _, layer in iter_tinylora_layers(model):
        if layer.original_name in layers:
            raise ValueError(f"duplicate attached TinyLoRA layer {layer.original_name!r}")
        layers[layer.original_name] = layer
    if not layers:
        raise ValueError("model has no attached TinyLoRA layers")
    return layers


@contextmanager
def temporary_tinylora_state(
    model: nn.Module,
    bank: Tensor,
    scalings: Mapping[str, float],
) -> Iterator[None]:
    """Temporarily replace a TinyLoRA bank/scalings and always restore them."""

    parameter_bank = getattr(getattr(model, "tinylora_bank", None), "v", None)
    if not isinstance(parameter_bank, nn.Parameter):
        raise ValueError("model has no TinyLoRA parameter bank")
    layers = _layer_map(model)
    if bank.shape != parameter_bank.shape:
        raise ValueError(
            f"candidate bank has shape {tuple(bank.shape)}, expected "
            f"{tuple(parameter_bank.shape)}"
        )
    expected_names = set(layers)
    observed_names = set(scalings)
    if observed_names != expected_names:
        raise ValueError(
            "candidate layer set differs from attached model: "
            f"missing={sorted(expected_names - observed_names)[:3]}, "
            f"extra={sorted(observed_names - expected_names)[:3]}"
        )
    if not torch.isfinite(bank.float()).all():
        raise ValueError("candidate bank contains non-finite values")
    for name, scaling in scalings.items():
        if not math.isfinite(float(scaling)):
            raise ValueError(f"candidate scaling for {name!r} is non-finite")

    saved_bank = parameter_bank.detach().clone()
    saved_scalings = {name: layer.scaling for name, layer in layers.items()}
    try:
        with torch.no_grad():
            parameter_bank.copy_(bank.to(parameter_bank))
            for name, layer in layers.items():
                layer.scaling = float(scalings[name])
        yield
    finally:
        with torch.no_grad():
            parameter_bank.copy_(saved_bank)
            for name, layer in layers.items():
                layer.scaling = saved_scalings[name]


def build_corpus_sample(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    text_column: str,
    sequence_length: int,
    max_tokens: int,
) -> CorpusSample:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    requested_blocks = max_tokens // sequence_length
    if requested_blocks < 1:
        raise ValueError("max_tokens must contain at least one complete sequence")

    blocks: list[list[int]] = []
    buffer: list[int] = []
    cursor = 0
    documents_consumed = 0
    nonempty_documents = 0
    separator = getattr(tokenizer, "eos_token_id", None)
    for record in records:
        documents_consumed += 1
        if text_column not in record:
            raise ValueError(f"corpus record has no {text_column!r} field")
        text = record[text_column]
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if isinstance(encoded, Tensor):
            encoded = encoded.reshape(-1).tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        token_ids = [int(value) for value in encoded]
        if not token_ids:
            continue
        if nonempty_documents and separator is not None:
            buffer.append(int(separator))
        nonempty_documents += 1
        buffer.extend(token_ids)
        while len(buffer) - cursor >= sequence_length:
            blocks.append(buffer[cursor : cursor + sequence_length])
            cursor += sequence_length
            if len(blocks) == requested_blocks:
                tensor = torch.tensor(blocks, dtype=torch.long)
                digest = hashlib.sha256(
                    tensor.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                return CorpusSample(
                    blocks=tensor,
                    documents_consumed=documents_consumed,
                    nonempty_documents=nonempty_documents,
                    token_sha256=digest,
                )
        if cursor >= 4 * sequence_length:
            buffer = buffer[cursor:]
            cursor = 0

    if not blocks:
        raise ValueError("the selected corpus produced no complete token blocks")
    tensor = torch.tensor(blocks, dtype=torch.long)
    digest = hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    return CorpusSample(
        blocks=tensor,
        documents_consumed=documents_consumed,
        nonempty_documents=nonempty_documents,
        token_sha256=digest,
    )


def _safe_exp(value: float) -> float:
    return math.exp(max(min(value, 80.0), -80.0))


@torch.inference_mode()
def evaluate_blocks(
    model: nn.Module,
    blocks: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if blocks.ndim != 2 or blocks.shape[1] < 2 or len(blocks) == 0:
        raise ValueError("blocks must be a non-empty [batch, sequence] tensor")

    total_nll = 0.0
    total_tokens = 0
    per_window_nll: list[float] = []
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size].to(device)
        logits = model(input_ids=batch, use_cache=False).logits[:, :-1].float()
        targets = batch[:, 1:]
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        window_sums = token_losses.double().sum(dim=1)
        tokens_per_window = int(targets.shape[1])
        total_nll += float(window_sums.sum().item())
        total_tokens += len(batch) * tokens_per_window
        per_window_nll.extend(
            float(value) for value in (window_sums / tokens_per_window).cpu()
        )
    mean_nll = total_nll / total_tokens
    return {
        "nll_sum": total_nll,
        "nll": mean_nll,
        "perplexity": _safe_exp(mean_nll),
        "tokens": total_tokens,
        "windows": len(blocks),
        "per_window_nll": per_window_nll,
    }


def _candidate_summary(
    state: CandidateState,
    base: Mapping[str, object],
    tuned: Mapping[str, object],
) -> dict[str, object]:
    base_windows = list(base["per_window_nll"])
    tuned_windows = list(tuned["per_window_nll"])
    deltas = [
        float(after) - float(before)
        for before, after in zip(base_windows, tuned_windows, strict=True)
    ]
    delta_nll = float(tuned["nll"]) - float(base["nll"])
    scaling_values = [value for _, value in state.scalings]
    return {
        "adapter": str(state.path.resolve()),
        "checkpoint_sha256": state.checkpoint_sha256,
        "bank_norm": float(state.bank.float().norm().item()),
        "scaling_min": min(scaling_values),
        "scaling_max": max(scaling_values),
        "metrics": dict(tuned),
        "delta_nll": delta_nll,
        "perplexity_ratio": _safe_exp(delta_nll),
        "mean_paired_window_delta": statistics.fmean(deltas),
        "median_paired_window_delta": statistics.median(deltas),
        "per_window_delta_nll": deltas,
    }


def main() -> None:
    args = parse_args()
    labels = [label for label, _ in args.adapter]
    if len(labels) != len(set(labels)):
        raise ValueError("adapter labels must be unique")

    states = validate_candidate_states(
        load_candidate_state(path) for _, path in args.adapter
    )
    state_by_label = dict(zip(labels, states, strict=True))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        revision=args.dataset_revision,
        streaming=args.streaming,
    )
    sample = build_corpus_sample(
        dataset,
        tokenizer,
        text_column=args.text_column,
        sequence_length=args.sequence_length,
        max_tokens=args.max_tokens,
    )

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    load_tinylora(model, states[0].path)
    model.eval()
    if model.tinylora_bank.v.dtype != torch.float32:
        raise TypeError(
            "Nemotron retention evaluation expects a float32 TinyLoRA bank, "
            f"found {model.tinylora_bank.v.dtype}"
        )
    resolved_revision = getattr(model.config, "_commit_hash", None)

    started = time.perf_counter()
    attached_scalings = {
        layer.original_name: layer.scaling for _, layer in iter_tinylora_layers(model)
    }
    with temporary_tinylora_state(
        model,
        torch.zeros_like(model.tinylora_bank.v),
        attached_scalings,
    ):
        base = evaluate_blocks(
            model,
            sample.blocks,
            batch_size=args.batch_size,
            device=device,
        )

    candidates: dict[str, object] = {}
    for label in labels:
        state = state_by_label[label]
        with temporary_tinylora_state(model, state.bank, state.scaling_by_name):
            tuned = evaluate_blocks(
                model,
                sample.blocks,
                batch_size=args.batch_size,
                device=device,
            )
        candidates[label] = _candidate_summary(state, base, tuned)

    result = {
        "schema_version": 1,
        "metric": "tinylora_causal_language_model_retention",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "model_dtype": str(next(model.parameters()).dtype),
        "bank_dtype": str(model.tinylora_bank.v.dtype),
        "shared_factor_sha256": states[0].factor_signature,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "split": args.split,
        "text_column": args.text_column,
        "sequence_length": args.sequence_length,
        "requested_max_tokens": args.max_tokens,
        "sample_input_tokens": int(sample.blocks.numel()),
        "sample_token_sha256": sample.token_sha256,
        "documents_consumed": sample.documents_consumed,
        "nonempty_documents": sample.nonempty_documents,
        "elapsed_seconds": time.perf_counter() - started,
        "base": base,
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)

    concise = {
        "base_nll": base["nll"],
        "base_perplexity": base["perplexity"],
        "candidates": {
            label: {
                "nll": value["metrics"]["nll"],
                "delta_nll": value["delta_nll"],
                "perplexity_ratio": value["perplexity_ratio"],
            }
            for label, value in candidates.items()
        },
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
