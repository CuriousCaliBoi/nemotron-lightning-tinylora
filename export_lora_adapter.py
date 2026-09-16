#!/usr/bin/env python3
"""Convert a TinyLoRA checkpoint to a standard inference-only PEFT LoRA."""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from tinylora_rl.adapter_analysis import export_peft_adapter, load_adapter_updates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="Multiply the materialized adapter update (0 creates a no-op control).",
    )
    parser.add_argument(
        "--dtype",
        choices=("preserve", "float16", "bfloat16", "float32"),
        default="preserve",
        help="Storage dtype for exported LoRA tensors.",
    )
    args = parser.parse_args()
    updates = load_adapter_updates(args.adapter)
    updates = [
        replace(update, scaling=update.scaling * args.multiplier)
        for update in updates
    ]
    dtype = {
        "preserve": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    output = export_peft_adapter(
        updates,
        args.output,
        base_model_name=args.base_model,
        tensor_dtype=dtype,
    )
    print(
        f"exported {len(updates)} matrices to {output} "
        f"with multiplier={args.multiplier:g} dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
