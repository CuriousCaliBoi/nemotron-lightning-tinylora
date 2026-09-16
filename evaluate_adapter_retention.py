#!/usr/bin/env python3
"""Measure base and adapter causal NLL on a fixed unrelated text sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinylora_rl.adapter_analysis import apply_updates_, load_adapter_updates


def parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("adapters must use LABEL=PATH")
    return label, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", action="append", type=parse_labelled_path, default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_blocks(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    total_nll = 0.0
    total_tokens = 0
    per_block_nll: list[float] = []
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size].to(device)
        logits = model(input_ids=batch, use_cache=False).logits[:, :-1].float()
        targets = batch[:, 1:]
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        block_sums = token_losses.sum(dim=1)
        block_counts = torch.full_like(block_sums, targets.shape[1], dtype=torch.long)
        total_nll += float(block_sums.sum().item())
        total_tokens += int(block_counts.sum().item())
        per_block_nll.extend(
            float(value)
            for value in (block_sums / block_counts).detach().cpu()
        )
    mean_nll = total_nll / total_tokens
    return {
        "nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
        "tokens": total_tokens,
        "blocks": len(blocks),
        "per_block_nll": per_block_nll,
    }


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2 or args.max_tokens < args.sequence_length:
        raise ValueError("max_tokens must contain at least one sequence")
    labels = [label for label, _ in args.adapter]
    if len(labels) != len(set(labels)):
        raise ValueError("adapter labels must be unique")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = [str(value) for value in dataset[args.text_column] if str(value).strip()]
    encoded = tokenizer(
        "\n\n".join(texts),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    usable = min(len(encoded), args.max_tokens)
    usable -= usable % args.sequence_length
    if usable == 0:
        raise ValueError("the selected corpus produced no complete token blocks")
    blocks = encoded[:usable].reshape(-1, args.sequence_length).contiguous()
    token_hash = hashlib.sha256(blocks.numpy().tobytes()).hexdigest()

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    started = time.perf_counter()
    base = evaluate_blocks(model, blocks, batch_size=args.batch_size, device=device)
    candidates: dict[str, object] = {}
    for label, path in args.adapter:
        updates = load_adapter_updates(path)
        applied = apply_updates_(model, updates)
        tuned = evaluate_blocks(model, blocks, batch_size=args.batch_size, device=device)
        apply_updates_(model, updates, multiplier=-1.0)
        base_per_block = base["per_block_nll"]
        tuned_per_block = tuned["per_block_nll"]
        deltas = [
            float(tuned_value) - float(base_value)
            for base_value, tuned_value in zip(base_per_block, tuned_per_block, strict=True)
        ]
        delta_nll = float(tuned["nll"]) - float(base["nll"])
        candidates[label] = {
            "adapter": str(path),
            "applied_matrices": applied,
            "metrics": tuned,
            "delta_nll": delta_nll,
            "perplexity_ratio": math.exp(max(min(delta_nll, 80.0), -80.0)),
            "mean_paired_block_delta": statistics.fmean(deltas),
            "median_paired_block_delta": statistics.median(deltas),
        }

    result = {
        "schema_version": 1,
        "metric": "causal_language_model_retention",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_column": args.text_column,
        "sequence_length": args.sequence_length,
        "sample_token_sha256": token_hash,
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
