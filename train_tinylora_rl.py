#!/usr/bin/env python3
"""Reproduce TinyLoRA RL on GSM8K without TRL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from tinylora_rl.profiling import UnifiedMemoryProfiler
from tinylora_rl.trainer import TinyLoRAGRPOTrainer, TrainConfig


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe representation of config metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    text = str(value)
    return text.removeprefix("torch.")


def summarize_quantization_config(config: object | None) -> dict[str, Any] | None:
    """Compact a possibly enormous HF quantization config without losing identity."""
    if config is None:
        return None
    normalized = _json_safe(config)
    if not isinstance(normalized, dict):
        normalized = {"value": normalized}
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result: dict[str, Any] = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "serialized_bytes": len(encoded),
    }

    # Small configs (for example BitsAndBytes) are useful verbatim.  ModelOpt
    # configs can contain one entry for every expert tensor and exceed 200 KB,
    # so retain their reproducibility hash and the semantically relevant
    # settings/counts instead of copying thousands of paths into every run.
    if len(encoded) <= 16_384:
        result["configuration"] = normalized
        return result

    for key in (
        "quant_method",
        "quant_algo",
        "producer",
        "kv_cache_scheme",
        "quantization_status",
    ):
        if key in normalized:
            result[key] = normalized[key]

    config_groups = normalized.get("config_groups")
    if isinstance(config_groups, Mapping):
        groups: dict[str, Any] = {}
        for name, group in config_groups.items():
            if not isinstance(group, Mapping):
                groups[str(name)] = _json_safe(group)
                continue
            compact_group = {
                str(key): item
                for key, item in group.items()
                if key != "targets"
            }
            targets = group.get("targets")
            if isinstance(targets, list):
                compact_group["target_count"] = len(targets)
            groups[str(name)] = compact_group
        result["config_groups"] = groups

    quantized_layers = normalized.get("quantized_layers")
    if isinstance(quantized_layers, Mapping):
        algorithms = Counter(
            str(layer.get("quant_algo"))
            for layer in quantized_layers.values()
            if isinstance(layer, Mapping) and layer.get("quant_algo") is not None
        )
        result["quantized_layer_count"] = len(quantized_layers)
        result["quantized_layer_algorithms"] = dict(sorted(algorithms.items()))
    ignored = normalized.get("ignore")
    if isinstance(ignored, list):
        result["ignored_target_count"] = len(ignored)
    result["configuration_omitted"] = True
    return result


def summarize_model_config(config: object | None) -> dict[str, Any]:
    """Select stable architecture fields and quantization details from an HF config."""
    if config is None:
        return {}
    raw = _json_safe(config.to_dict()) if hasattr(config, "to_dict") else {}
    if not isinstance(raw, dict):
        raw = {}
    summary = {}
    for key in (
        "architectures",
        "model_type",
        "dtype",
        "torch_dtype",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "n_routed_experts",
        "num_experts_per_tok",
        "max_position_embeddings",
        "vocab_size",
        "transformers_version",
    ):
        value = raw.get(key, getattr(config, key, None))
        if value is not None:
            summary[key] = _json_safe(value)
    quantization = raw.get(
        "quantization_config",
        getattr(config, "quantization_config", None),
    )
    compact_quantization = summarize_quantization_config(quantization)
    if compact_quantization is not None:
        summary["quantization_config"] = compact_quantization
    return summary


def _resolved_revision(*configs: object | None) -> str | None:
    for config in configs:
        if config is None:
            continue
        for name in ("_commit_hash", "commit_hash", "revision"):
            value = getattr(config, name, None)
            if isinstance(value, str) and value:
                return value
    return None


def _library_versions() -> dict[str, str]:
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "vllm": "vllm",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "numpy": "numpy",
    }
    versions = {}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def verify_factor_cache(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Hash one immutable factor cache and optionally enforce its identity."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"frozen-factor cache does not exist: {cache_path}")
    digest = hashlib.sha256()
    with cache_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(
            "factor-cache hash mismatch: "
            f"expected {expected_sha256}, got {observed}"
        )
    return {
        "path": str(cache_path),
        "sha256": observed,
        "bytes": cache_path.stat().st_size,
    }


def build_run_provenance(
    args: argparse.Namespace,
    learner_model: object,
    rollout: object,
) -> dict[str, Any]:
    """Describe the two-policy execution setup that produced an adapter run."""
    learner_config = getattr(learner_model, "config", None)
    llm = getattr(rollout, "llm", None)
    vllm_model_config = getattr(llm, "model_config", None)
    rollout_config = getattr(vllm_model_config, "hf_config", None)
    rollout_model_id = args.rollout_model or args.model
    requested_model_revision = getattr(args, "model_revision", None)
    requested_rollout_revision = getattr(args, "rollout_revision", None)
    if requested_rollout_revision is None and rollout_model_id == args.model:
        requested_rollout_revision = requested_model_revision
    learner_dtype = getattr(learner_model, "dtype", torch.bfloat16)

    return {
        "learner": {
            "role": "differentiable_policy",
            "model_id": args.model,
            "requested_revision": requested_model_revision,
            "resolved_revision": _resolved_revision(learner_config),
            "load": {
                "dtype": _json_safe(learner_dtype),
                "attention_implementation": "sdpa",
                "low_cpu_mem_usage": True,
                "device_map": {"": "cuda"},
                "trust_remote_code": args.trust_remote_code,
                "gradient_checkpointing": args.gradient_checkpointing,
            },
            "model_config": summarize_model_config(learner_config),
        },
        "rollout": {
            "role": "sampling_actor",
            "backend": "vllm",
            "model_id": rollout_model_id,
            "requested_revision": requested_rollout_revision,
            "resolved_revision": _resolved_revision(
                rollout_config,
                vllm_model_config,
            ),
            "sync_mode": args.rollout_sync,
            "model_config": summarize_model_config(rollout_config),
            "resolved_engine_model": {
                "dtype": _json_safe(getattr(vllm_model_config, "dtype", None)),
                "quantization": _json_safe(
                    getattr(vllm_model_config, "quantization", None)
                ),
                "max_model_length": _json_safe(
                    getattr(vllm_model_config, "max_model_len", None)
                ),
            },
            "engine": {
                "tensor_parallel_size": 1,
                "distributed_executor_backend": "external_launcher",
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_model_length": args.max_model_length,
                "max_num_batched_tokens": args.max_model_length,
                "enforce_eager": True,
                "logprobs_mode": "processed_logprobs",
                "trust_remote_code": args.trust_remote_code,
                "cache": {
                    "kv_cache_dtype": args.vllm_kv_cache_dtype,
                    "kv_cache_memory_bytes": args.vllm_kv_cache_memory_bytes,
                },
                "backends": {
                    "moe": args.vllm_moe_backend,
                    "mamba": args.vllm_mamba_backend,
                    "mamba_cache_mode": args.vllm_mamba_cache_mode,
                },
                "lora": {
                    "enabled": args.rollout_sync == "lora",
                    "max_loras": 1 if args.rollout_sync == "lora" else 0,
                    "max_cpu_loras": 1 if args.rollout_sync == "lora" else 0,
                    "max_rank": args.max_lora_rank if args.rollout_sync == "lora" else None,
                    "dtype": "bfloat16" if args.rollout_sync == "lora" else None,
                    "target_modules": (
                        [
                            item.strip()
                            for item in args.target_modules.split(",")
                            if item.strip()
                        ]
                        if args.rollout_sync == "lora"
                        else []
                    ),
                },
            },
        },
        "environment": {
            "python": platform.python_version(),
            "libraries": _library_versions(),
            "cuda": {
                "torch_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision")
    parser.add_argument("--rollout-model", help="May be a quantized version of --model")
    parser.add_argument("--rollout-revision")
    parser.add_argument("--rollout-sync", choices=("merged", "lora"), default="merged")
    parser.add_argument("--output-dir", default="/workspace/outputs/tinylora-from-scratch")
    parser.add_argument("--factor-cache")
    parser.add_argument(
        "--factor-cache-sha256",
        help="Expected SHA-256 for an existing immutable frozen-factor cache",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision")
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
    parser.add_argument(
        "--target-layer-indices",
        help="Optional comma-separated transformer layer indices to adapt",
    )
    parser.add_argument("--parameter-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--grouping", choices=("tiled", "structured"), default="tiled")
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--projection-std", type=float)
    parser.add_argument("--adapter-scaling", type=float, default=1.0)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument(
        "--vllm-kv-cache-memory-bytes",
        type=int,
        help="Explicit vLLM KV-cache budget; avoids colocated memory-profiling double counts",
    )
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
    parser.add_argument("--prompt-style", choices=("concise", "verl"), default="concise")
    parser.add_argument("--reward-mode", choices=("flexible", "strict"), default="flexible")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-sample-interval", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RL training entrypoint")
    output_dir = Path(args.output_dir)
    profiler = UnifiedMemoryProfiler(
        output_dir / "memory_profile.json",
        enabled=args.profile_memory,
        sample_interval=args.profile_sample_interval,
    )
    profiler.start()
    status = "failed"
    failure: str | None = None
    try:
        set_seed(args.seed)
        with profiler.phase("process_group_init"):
            ensure_single_gpu_process_group()
        with profiler.phase("tokenizer_load"):
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                revision=args.model_revision,
                trust_remote_code=args.trust_remote_code,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        with profiler.phase("learner_load"):
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.model_revision,
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
        profiler.record_module("learner_before_adapter", model)
        adapter_config = TinyLoRAConfig(
            rank=args.rank,
            projection_dim=args.projection_dim,
            modules_per_group=args.modules_per_group,
            num_groups=args.num_groups,
            target_modules=tuple(
                item.strip() for item in args.target_modules.split(",") if item.strip()
            ),
            target_layer_indices=(
                tuple(
                    int(item.strip())
                    for item in args.target_layer_indices.split(",")
                    if item.strip()
                )
                if args.target_layer_indices is not None
                else None
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
        factor_cache_before = None
        if Path(factor_cache).is_file():
            factor_cache_before = verify_factor_cache(
                factor_cache,
                expected_sha256=args.factor_cache_sha256,
            )
        elif args.factor_cache_sha256 is not None:
            raise FileNotFoundError(
                "an expected factor-cache SHA-256 requires an existing cache: "
                f"{factor_cache}"
            )
        with profiler.phase("tinylora_attach"):
            apply_tinylora(model, adapter_config, factor_cache=factor_cache)
        factor_cache_provenance = verify_factor_cache(
            factor_cache,
            expected_sha256=args.factor_cache_sha256,
        )
        if (
            factor_cache_before is not None
            and factor_cache_before["sha256"] != factor_cache_provenance["sha256"]
        ):
            raise RuntimeError("apply_tinylora unexpectedly mutated the factor cache")
        profiler.record_module("learner_with_tinylora", model)
        print(
            json.dumps(
                {
                    "target_layers": sum(1 for module in model.modules() if module.__class__.__name__ == "TinyLoRALinear"),
                    "trainable_parameters": trainable_parameter_count(model),
                    "adapter_config": adapter_config.__dict__,
                    "factor_cache": factor_cache,
                    "factor_cache_sha256": factor_cache_provenance["sha256"],
                },
                default=list,
                indent=2,
            ),
            flush=True,
        )
        with profiler.phase("dataset_load"):
            dataset = load_dataset(
                "openai/gsm8k",
                "main",
                split=args.dataset_split,
                revision=args.dataset_revision,
            )
            dataset = dataset.shuffle(seed=args.seed).select(range(min(args.samples, len(dataset))))
        rollout_model = args.rollout_model or args.model
        engine_kwargs = {"kv_cache_dtype": args.vllm_kv_cache_dtype}
        rollout_revision = args.rollout_revision
        if rollout_revision is None and rollout_model == args.model:
            rollout_revision = args.model_revision
        if rollout_revision is not None:
            engine_kwargs["revision"] = rollout_revision
        if args.vllm_kv_cache_memory_bytes is not None:
            engine_kwargs["kv_cache_memory_bytes"] = args.vllm_kv_cache_memory_bytes
        for key, value in (
            ("moe_backend", args.vllm_moe_backend),
            ("mamba_backend", args.vllm_mamba_backend),
            ("mamba_cache_mode", args.vllm_mamba_cache_mode),
        ):
            if value is not None:
                engine_kwargs[key] = value
        with profiler.phase("rollout_engine_init"):
            if args.rollout_sync == "lora":
                rollout = VLLMLoRARolloutBackend(
                    rollout_model,
                    adapter_root=output_dir / "vllm_adapters",
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    max_model_len=args.max_model_length,
                    max_lora_rank=args.max_lora_rank,
                    lora_target_modules=adapter_config.target_modules,
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
        profiler.record_module("rollout_model", rollout.model_for_profiling())
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
            prompt_style=args.prompt_style,
            reward_mode=args.reward_mode,
        )
        run_provenance = build_run_provenance(args, model, rollout)
        run_provenance["factor_cache"] = factor_cache_provenance
        trainer = TinyLoRAGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            rollout=rollout,
            config=train_config,
            output_dir=output_dir,
            memory_profiler=profiler,
            run_provenance=run_provenance,
            dataset_split=args.dataset_split,
            dataset_revision=args.dataset_revision,
        )
        trainer.train(dataset)
        status = "complete"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        profiler.mark("run_failed", error=failure)
        raise
    finally:
        try:
            with profiler.phase("process_group_destroy"):
                destroy_process_group()
        finally:
            profiler.finish(status=status, error=failure)


if __name__ == "__main__":
    main()
