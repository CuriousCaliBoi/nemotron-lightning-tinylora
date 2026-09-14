#!/usr/bin/env python3
"""Reproduce TinyLoRA RL on GSM8K without TRL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tinylora_rl.adapters import (
    DEFAULT_TARGET_MODULES,
    TinyLoRAConfig,
    apply_tinylora,
    trainable_parameter_count,
)
from tinylora_rl.rollout import (
    VLLMLoRARolloutBackend,
    VLLMRolloutBackend,
    destroy_process_group,
    ensure_single_gpu_process_group,
)
from tinylora_rl.trainer import TinyLoRAGRPOTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rollout-model", help="May be a quantized version of --model")
    parser.add_argument("--rollout-sync", choices=("merged", "lora"), default="merged")
    parser.add_argument("--output-dir", default="/workspace/outputs/tinylora-from-scratch")
    parser.add_argument("--factor-cache")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--tis-mode",
        choices=("none", "token_clip", "token_mask", "sequence_clip", "sequence_mask"),
        default="token_clip",
    )
    parser.add_argument("--tis-minimum", type=float, default=0.1)
    parser.add_argument("--tis-maximum", type=float, default=10.0)
    parser.add_argument("--loss-reduction", choices=("sample_mean", "token_mean"), default="sample_mean")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=1)
    parser.add_argument("--modules-per-group", type=int, default=16)
    parser.add_argument(
        "--num-groups",
        type=int,
        help="Tie matched modules across exactly this many trainable parameter groups",
    )
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated linear module suffixes",
    )
    parser.add_argument("--parameter-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--grouping", choices=("tiled", "structured"), default="tiled")
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--projection-std", type=float)
    parser.add_argument("--adapter-scaling", type=float, default=1.0)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--max-lora-rank", type=int, default=8)
    parser.add_argument("--vllm-moe-backend")
    parser.add_argument("--vllm-mamba-backend")
    parser.add_argument("--vllm-mamba-cache-mode")
    parser.add_argument("--vllm-kv-cache-dtype", default="auto")
    parser.add_argument("--max-model-length", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RL training entrypoint")
    set_seed(args.seed)
    ensure_single_gpu_process_group()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": "cuda"},
            trust_remote_code=args.trust_remote_code,
        )
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        adapter_config = TinyLoRAConfig(
            rank=args.rank,
            projection_dim=args.projection_dim,
            modules_per_group=args.modules_per_group,
            num_groups=args.num_groups,
            target_modules=tuple(
                item.strip() for item in args.target_modules.split(",") if item.strip()
            ),
            grouping=args.grouping,
            projection_seed=args.projection_seed,
            projection_std=args.projection_std,
            scaling=args.adapter_scaling,
            parameter_dtype=args.parameter_dtype,
            svd_niter=args.svd_niter,
        )
        factor_cache = args.factor_cache
        if factor_cache is None:
            model_slug = args.model.replace("/", "--")
            factor_cache = f"/workspace/cache/{model_slug}-r{args.rank}-svd.safetensors"
        apply_tinylora(model, adapter_config, factor_cache=factor_cache)
        print(
            json.dumps(
                {
                    "target_layers": sum(1 for module in model.modules() if module.__class__.__name__ == "TinyLoRALinear"),
                    "trainable_parameters": trainable_parameter_count(model),
                    "adapter_config": adapter_config.__dict__,
                    "factor_cache": factor_cache,
                },
                default=list,
                indent=2,
            ),
            flush=True,
        )
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.samples, len(dataset))))
        rollout_model = args.rollout_model or args.model
        engine_kwargs = {"kv_cache_dtype": args.vllm_kv_cache_dtype}
        for key, value in (
            ("moe_backend", args.vllm_moe_backend),
            ("mamba_backend", args.vllm_mamba_backend),
            ("mamba_cache_mode", args.vllm_mamba_cache_mode),
        ):
            if value is not None:
                engine_kwargs[key] = value
        if args.rollout_sync == "lora":
            rollout = VLLMLoRARolloutBackend(
                rollout_model,
                adapter_root=Path(args.output_dir) / "vllm_adapters",
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.max_model_length,
                max_lora_rank=args.max_lora_rank,
                seed=args.seed,
                trust_remote_code=args.trust_remote_code,
                engine_kwargs=engine_kwargs,
            )
        else:
            if rollout_model != args.model:
                raise ValueError(
                    "different learner/rollout models require --rollout-sync lora; "
                    "merged synchronization overwrites base weights"
                )
            rollout = VLLMRolloutBackend(
                rollout_model,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.max_model_length,
                seed=args.seed,
                trust_remote_code=args.trust_remote_code,
                engine_kwargs=engine_kwargs,
            )
        train_config = TrainConfig(
            steps=args.steps,
            prompts_per_step=args.prompts_per_step,
            generations_per_prompt=args.generations,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            micro_batch_size=args.micro_batch_size,
            tis_mode=args.tis_mode,
            tis_minimum=args.tis_minimum,
            tis_maximum=args.tis_maximum,
            loss_reduction=args.loss_reduction,
            seed=args.seed,
            save_every=args.save_every,
        )
        trainer = TinyLoRAGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            rollout=rollout,
            config=train_config,
            output_dir=Path(args.output_dir),
        )
        trainer.train(dataset)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
