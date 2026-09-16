#!/usr/bin/env python3
"""Paired learning-rate/scaling sweep for Nemotron 3.5 TinyLoRA.

The BF16 learner, fixed TinyLoRA factors, dataset sample, and NVFP4 rollout
engine are loaded once.  Each candidate starts from the identical zero adapter
and uses the same prompt order and rollout seeds, which makes this much less
expensive and less noisy than launching an independent process per candidate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tinylora_rl.adapters import (
    TinyLoRAConfig,
    apply_tinylora,
    export_vllm_lora,
    iter_tinylora_layers,
    trainable_parameter_count,
)
from tinylora_rl.rollout import (
    VLLMLoRARolloutBackend,
    destroy_process_group,
    ensure_single_gpu_process_group,
)
from tinylora_rl.trainer import TinyLoRAGRPOTrainer, TrainConfig


_VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
_RUNTIME_SOURCE_PATHS = (
    "train_nemotron35_tinylora_sweep.py",
    "tinylora_rl/__init__.py",
    "tinylora_rl/adapters.py",
    "tinylora_rl/objectives.py",
    "tinylora_rl/profiling.py",
    "tinylora_rl/prompts.py",
    "tinylora_rl/rewards.py",
    "tinylora_rl/rollout.py",
    "tinylora_rl/trainer.py",
)


class FirstStepReplayRollout:
    """Replay one identical first-step rollout for every sweep candidate.

    A common seed is insufficient for paired comparisons because vLLM engine
    history and quantized kernels can still change sampled tokens. The first
    candidate therefore generates the reference rollout once; later candidates
    receive deep copies of the exact token IDs, text, and rollout logprobs.
    Subsequent steps continue to sample from each candidate's updated policy.
    """

    sync_before_first_rollout = True

    def __init__(self, backend: VLLMLoRARolloutBackend) -> None:
        self.backend = backend
        self._cached_first_result: object | None = None
        self._cached_first_request: dict[str, object] | None = None
        self._candidate_label: str | None = None
        self._generate_calls = 0
        self.first_step_replayed = False

    def __getattr__(self, name: str) -> object:
        return getattr(self.backend, name)

    def begin_candidate(self, label: str) -> None:
        if not label:
            raise ValueError("candidate label must be non-empty")
        self._candidate_label = label
        self._generate_calls = 0
        self.first_step_replayed = False

    @property
    def first_step_request_sha256(self) -> str | None:
        if self._cached_first_request is None:
            return None
        serialized = json.dumps(
            self._cached_first_request,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _request_signature(
        prompt_ids: Sequence[Sequence[int]],
        *,
        num_generations: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> dict[str, object]:
        return {
            "prompt_ids": [list(prompt) for prompt in prompt_ids],
            "num_generations": int(num_generations),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": seed,
        }

    def generate(
        self,
        prompt_ids: Sequence[list[int]],
        *,
        num_generations: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> object:
        if self._candidate_label is None:
            raise RuntimeError("begin_candidate() must be called before generate()")
        self._generate_calls += 1
        if self._generate_calls != 1:
            return self.backend.generate(
                prompt_ids,
                num_generations=num_generations,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )

        signature = self._request_signature(
            prompt_ids,
            num_generations=num_generations,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        if self._cached_first_result is None:
            result = self.backend.generate(
                prompt_ids,
                num_generations=num_generations,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            self._cached_first_request = copy.deepcopy(signature)
            self._cached_first_result = copy.deepcopy(result)
            return copy.deepcopy(result)

        if signature != self._cached_first_request:
            raise RuntimeError(
                "paired sweep invalid: first-step rollout request differs from "
                f"the cached reference for candidate {self._candidate_label}"
            )
        self.first_step_replayed = True
        return copy.deepcopy(self._cached_first_result)


def supported_max_lora_rank(required: int) -> int:
    """Round an adapter rank up to a max-rank bucket accepted by vLLM."""

    if required < 1:
        raise ValueError("adapter rank must be positive")
    for candidate in _VLLM_MAX_LORA_RANKS:
        if required <= candidate:
            return candidate
    raise ValueError(f"adapter rank {required} exceeds vLLM's supported maximum")


def parse_candidate(value: str) -> tuple[str, float, float]:
    fields = value.split(":")
    if len(fields) not in (2, 3) or not fields[0]:
        raise argparse.ArgumentTypeError("candidate must be LABEL:LEARNING_RATE[:SCALING]")
    learning_rate = float(fields[1])
    scaling = float(fields[2]) if len(fields) == 3 else 1.0
    if learning_rate <= 0 or scaling <= 0:
        raise argparse.ArgumentTypeError("candidate learning rate and scaling must be positive")
    return fields[0], learning_rate, scaling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    )
    parser.add_argument(
        "--rollout-model",
        default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    )
    parser.add_argument(
        "--model-revision",
        default="a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    )
    parser.add_argument(
        "--rollout-revision",
        default="bee7596271d1495f6992ae224aefde4410e816b8",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        metavar="LABEL:LR[:SCALING]",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--factor-cache",
        default="/workspace/cache/nemotron35-lightning-attention-r2-svd.safetensors",
    )
    parser.add_argument(
        "--expected-factor-cache-sha256",
        default="da2cccb61f2745aec631ca94c5ec4a0e764386e9d0c21c789f1d83c1ced95594",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--train-split",
        default="train[:-512]",
        help="Dataset split reserved for fitting; the final 512 train rows are held out.",
    )
    parser.add_argument(
        "--dataset-revision",
        default="740312add88f781978c0658806c59bc2815b9866",
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-model-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-style", choices=("concise", "verl"), default="concise")
    parser.add_argument("--reward-mode", choices=("flexible", "strict"), default="strict")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--tis-mode", default="token_clip")
    parser.add_argument("--tis-minimum", type=float, default=0.1)
    parser.add_argument("--tis-maximum", type=float, default=10.0)
    parser.add_argument("--loss-reduction", default="sample_mean")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=1)
    parser.add_argument("--num-groups", type=int, default=13)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--parameter-dtype", default="float32")
    parser.add_argument("--grouping", default="tiled")
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.23)
    parser.add_argument("--vllm-moe-backend", default="marlin")
    parser.add_argument("--vllm-mamba-backend", default="flashinfer")
    parser.add_argument("--vllm-mamba-cache-mode", default="align")
    parser.add_argument("--vllm-kv-cache-dtype", default="fp8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def last_json_line(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_sha256(source_root: Path) -> dict[str, str]:
    """Hash every local source file that can affect the sweep."""

    hashes: dict[str, str] = {}
    for relative_path in _RUNTIME_SOURCE_PATHS:
        path = source_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing runtime source file: {path}")
        hashes[relative_path] = file_sha256(path)
    return hashes


def assert_runtime_sources_unchanged(
    source_root: Path, expected: dict[str, str]
) -> None:
    current = runtime_source_sha256(source_root)
    if current == expected:
        return
    changed = sorted(
        path
        for path in expected.keys() | current.keys()
        if expected.get(path) != current.get(path)
    )
    raise RuntimeError(
        "runtime source changed during the sweep; refusing mixed-code results: "
        + ", ".join(changed)
    )


def trajectory_step_sha256(path: Path, step: int = 1) -> str:
    """Hash a canonicalized rollout step for paired-candidate validation."""

    digest = hashlib.sha256()
    count = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["step"]) != step:
                continue
            digest.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            )
            digest.update(b"\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"no trajectories for step {step} in {path}")
    return digest.hexdigest()


def trajectory_design_sha256(path: Path, step: int = 1) -> str:
    """Hash only the common-random-number design, not stochastic outputs.

    vLLM honors the same explicit seeds, but tiny nondeterminism in quantized
    kernels can make sampled completions diverge even for a reloaded zero LoRA.
    The scientifically required invariant is identical prompts, ordering,
    group IDs, and sampling configuration; exact output equality is recorded
    as a diagnostic rather than treated as a fatal invariant.
    """

    digest = hashlib.sha256()
    count = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["step"]) != step:
                continue
            design = {
                "step": int(row["step"]),
                "group_id": int(row["group_id"]),
                "question": row["question"],
                "gold_answer": row["gold_answer"],
            }
            digest.update(json.dumps(design, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"no trajectories for step {step} in {path}")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be at least 1")
    if args.prompts_per_step < 1 or args.generations < 1:
        raise ValueError("prompts-per-step and generations must be positive")
    source_root = Path(__file__).resolve().parent
    source_hashes = runtime_source_sha256(source_root)
    factor_cache_path = Path(args.factor_cache)
    if not factor_cache_path.is_file():
        raise FileNotFoundError(
            f"the frozen-factor cache must already exist: {factor_cache_path}"
        )
    factor_cache_sha256 = file_sha256(factor_cache_path)
    if (
        args.expected_factor_cache_sha256
        and factor_cache_sha256 != args.expected_factor_cache_sha256
    ):
        raise RuntimeError(
            "factor-cache hash mismatch: "
            f"expected {args.expected_factor_cache_sha256}, got {factor_cache_sha256}"
        )
    max_lora_rank = supported_max_lora_rank(args.rank)
    labels = [candidate[0] for candidate in args.candidate]
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        candidate_dir = output_root / label
        if candidate_dir.exists() and any(candidate_dir.iterdir()):
            raise FileExistsError(f"refusing to append to non-empty {candidate_dir}")

    set_seed(args.seed)
    ensure_single_gpu_process_group()
    started = time.perf_counter()
    status = "failed"
    results: dict[str, object] = {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
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
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_adapter_config = TinyLoRAConfig(
            rank=args.rank,
            projection_dim=args.projection_dim,
            num_groups=args.num_groups,
            target_modules=tuple(
                item.strip() for item in args.target_modules.split(",") if item.strip()
            ),
            grouping=args.grouping,
            projection_seed=args.projection_seed,
            scaling=1.0,
            parameter_dtype=args.parameter_dtype,
            svd_niter=args.svd_niter,
        )
        apply_tinylora(model, base_adapter_config, factor_cache=args.factor_cache)
        if file_sha256(factor_cache_path) != factor_cache_sha256:
            raise RuntimeError("apply_tinylora unexpectedly mutated the frozen-factor cache")
        actual_targets = {layer.original_name for _, layer in iter_tinylora_layers(model)}
        expected_targets = {
            f"model.layers.{layer}.mixer.{module}"
            for layer in (5, 12, 19, 26, 33, 42)
            for module in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        if actual_targets != expected_targets:
            raise RuntimeError(
                "unexpected Nemotron TinyLoRA target universe: "
                f"missing={sorted(expected_targets - actual_targets)}, "
                f"extra={sorted(actual_targets - expected_targets)}"
            )
        if trainable_parameter_count(model) != args.num_groups * args.projection_dim:
            raise RuntimeError("unexpected TinyLoRA parameter count")

        assert_runtime_sources_unchanged(source_root, source_hashes)
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split=args.train_split,
            revision=args.dataset_revision,
        )
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.samples, len(dataset))))
        rollout_backend = VLLMLoRARolloutBackend(
            args.rollout_model,
            adapter_root=output_root / "rollout_adapters",
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.max_model_length,
            max_lora_rank=max_lora_rank,
            lora_target_modules=base_adapter_config.target_modules,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            engine_kwargs={
                "kv_cache_dtype": args.vllm_kv_cache_dtype,
                "moe_backend": args.vllm_moe_backend,
                "mamba_backend": args.vllm_mamba_backend,
                "mamba_cache_mode": args.vllm_mamba_cache_mode,
                "revision": args.rollout_revision,
            },
        )
        rollout = FirstStepReplayRollout(rollout_backend)

        sweep_manifest = {
            "model": args.model,
            "model_revision": args.model_revision,
            "rollout_model": args.rollout_model,
            "rollout_revision": args.rollout_revision,
            "adapter_config": asdict(base_adapter_config),
            "factor_cache": args.factor_cache,
            "factor_cache_sha256": factor_cache_sha256,
            "runtime_source_sha256": source_hashes,
            "vllm_max_lora_rank": max_lora_rank,
            "pairing": {
                "protocol": "exact_first_step_token_text_logprob_replay_v1",
                "later_steps_replayed": False,
                "timings_comparable_across_candidates": False,
            },
            "candidates": [
                {"label": label, "learning_rate": learning_rate, "scaling": scaling}
                for label, learning_rate, scaling in args.candidate
            ],
            "common": {
                key: getattr(args, key)
                for key in (
                    "steps",
                    "train_split",
                    "dataset_revision",
                    "samples",
                    "prompts_per_step",
                    "generations",
                    "max_completion_length",
                    "max_model_length",
                    "micro_batch_size",
                    "temperature",
                    "top_p",
                    "prompt_style",
                    "reward_mode",
                    "ppo_epochs",
                    "clip_epsilon",
                    "tis_mode",
                    "tis_minimum",
                    "tis_maximum",
                    "loss_reduction",
                    "seed",
                )
            },
        }
        (output_root / "sweep_manifest.json").write_text(
            json.dumps(sweep_manifest, indent=2) + "\n"
        )

        first_step_reference_hash: str | None = None
        first_step_reference_design_hash: str | None = None
        for label, learning_rate, scaling in args.candidate:
            assert_runtime_sources_unchanged(source_root, source_hashes)
            rollout.begin_candidate(label)
            set_seed(args.seed)
            with torch.no_grad():
                model.tinylora_bank.v.zero_()
                model.tinylora_bank.v.grad = None
                for _, layer in iter_tinylora_layers(model):
                    layer.scaling = scaling
            reset_bank = model.tinylora_bank.v.detach()
            if not torch.isfinite(reset_bank).all() or torch.count_nonzero(reset_bank):
                raise RuntimeError(f"failed to reset TinyLoRA bank for candidate {label}")
            model.tinylora_config = replace(base_adapter_config, scaling=scaling)
            candidate_dir = output_root / label
            config = TrainConfig(
                steps=args.steps,
                prompts_per_step=args.prompts_per_step,
                generations_per_prompt=args.generations,
                max_completion_length=args.max_completion_length,
                temperature=args.temperature,
                top_p=args.top_p,
                learning_rate=learning_rate,
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
                save_every=0,
                prompt_style=args.prompt_style,
                reward_mode=args.reward_mode,
            )
            candidate_started = time.perf_counter()
            trainer = TinyLoRAGRPOTrainer(
                model=model,
                tokenizer=tokenizer,
                rollout=rollout,
                config=config,
                output_dir=candidate_dir,
                dataset_split=args.train_split,
                dataset_revision=args.dataset_revision,
                run_provenance={
                    "sweep": {
                        "label": label,
                        "runtime_source_sha256": source_hashes,
                        "first_step_rollout_protocol": "exact_token_logprob_replay_v1",
                    },
                    "learner": {
                        "model_id": args.model,
                        "requested_revision": args.model_revision,
                        "load_dtype": "bfloat16",
                    },
                    "rollout": {
                        "model_id": args.rollout_model,
                        "requested_revision": args.rollout_revision,
                        "sync_mode": "lora",
                        "kv_cache_dtype": args.vllm_kv_cache_dtype,
                    },
                },
            )
            trainer.train(dataset)
            first_step_hash = trajectory_step_sha256(
                candidate_dir / "trajectories.jsonl"
            )
            first_step_design_hash = trajectory_design_sha256(
                candidate_dir / "trajectories.jsonl"
            )
            if first_step_reference_hash is None:
                first_step_reference_hash = first_step_hash
            elif first_step_hash != first_step_reference_hash:
                raise RuntimeError(
                    "exact first-step replay invariant failed: candidate trajectory "
                    f"hash differs ({first_step_reference_hash} != {first_step_hash} "
                    f"for {label})"
                )
            if first_step_reference_design_hash is None:
                first_step_reference_design_hash = first_step_design_hash
            elif first_step_design_hash != first_step_reference_design_hash:
                raise RuntimeError(
                    "common-random-number sweep invalid: candidates used different "
                    f"first-step prompt designs ({first_step_reference_design_hash} != "
                    f"{first_step_design_hash} for {label})"
                )
            export_vllm_lora(
                model,
                candidate_dir / "peft_adapter",
                base_model_name=args.rollout_model,
            )
            results[label] = {
                "learning_rate": learning_rate,
                "scaling": scaling,
                "elapsed_seconds": time.perf_counter() - candidate_started,
                "last_training_metrics": last_json_line(candidate_dir / "metrics.jsonl"),
                "adapter_norm": float(model.tinylora_bank.v.detach().float().norm().item()),
                "first_step_trajectory_sha256": first_step_hash,
                "first_step_design_sha256": first_step_design_hash,
                "first_step_trajectory_matches_reference": (
                    first_step_hash == first_step_reference_hash
                ),
                "first_step_rollout_replayed": rollout.first_step_replayed,
                "first_step_request_sha256": rollout.first_step_request_sha256,
            }
            del trainer
            torch.cuda.empty_cache()
            assert_runtime_sources_unchanged(source_root, source_hashes)
            (output_root / "sweep_results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
        status = "complete"
    finally:
        destroy_process_group()
        final = {
            "status": status,
            "elapsed_seconds": time.perf_counter() - started,
            "results": results,
        }
        (output_root / "sweep_status.json").write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
