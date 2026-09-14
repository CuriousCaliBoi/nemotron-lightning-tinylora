"""A small vLLM rollout backend with direct merged-weight synchronization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .adapters import export_vllm_lora, iter_tinylora_layers


@dataclass
class Trajectory:
    prompt_ids: list[int]
    completion_ids: list[int]
    rollout_logprobs: list[float]
    completion: str
    reward: float
    group_id: int


class VLLMRolloutBackend:
    """Colocated single-process vLLM used only for sampling.

    TinyLoRA is not natively supported by vLLM. After an optimizer update we
    materialize each adapted linear weight and load that subset into vLLM,
    matching the approach described in the TinyLoRA paper.
    """

    def __init__(
        self,
        model_name: str,
        *,
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int,
        trust_remote_code: bool = False,
        enable_lora: bool = False,
        max_lora_rank: int = 8,
        engine_kwargs: dict[str, object] | None = None,
    ) -> None:
        # Import lazily so objective and adapter unit tests do not initialize CUDA.
        from vllm import LLM

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        lora_args = {}
        if enable_lora:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_cpu_loras": 1,
                "max_lora_rank": max_lora_rank,
                "lora_dtype": "bfloat16",
            }
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_model_len,
            enforce_eager=True,
            logprobs_mode="processed_logprobs",
            seed=seed,
            trust_remote_code=trust_remote_code,
            **lora_args,
            **(engine_kwargs or {}),
        )

    def _active_lora_request(self) -> object | None:
        return None

    @staticmethod
    def _chosen_logprob(token_id: int, candidates: dict[int, object]) -> float:
        item = candidates.get(token_id)
        if item is None:
            raise RuntimeError(f"vLLM did not return the chosen token {token_id} in logprobs")
        return float(item.logprob)

    def generate(
        self,
        prompt_ids: Sequence[list[int]],
        *,
        num_generations: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> list[tuple[int, list[int], list[float], str]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=num_generations,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=1,
            seed=seed,
        )
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
        request_outputs = self.llm.generate(
            prompts,
            params,
            use_tqdm=False,
            lora_request=self._active_lora_request(),
        )
        trajectories = []
        for group_id, request in enumerate(request_outputs):
            for output in request.outputs:
                token_ids = list(output.token_ids)
                if output.logprobs is None or len(output.logprobs) != len(token_ids):
                    raise RuntimeError("vLLM returned incomplete rollout logprobs")
                logprobs = [
                    self._chosen_logprob(token_id, candidates)
                    for token_id, candidates in zip(token_ids, output.logprobs, strict=True)
                ]
                trajectories.append((group_id, token_ids, logprobs, output.text))
        return trajectories

    @torch.no_grad()
    def sync_tinylora(self, model: nn.Module) -> int:
        """Load materialized adapted weights into the colocated vLLM model."""
        runner_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        layers = list(iter_tinylora_layers(model))

        # Pass the original Hugging Face names. vLLM's model loader maps Q/K/V
        # and gate/up shards into its packed qkv_proj and gate_up_proj tensors.
        # Loading the complete stream in one call also lets the loader account
        # for every packed destination before returning its validation set.
        def materialized_weights():
            for _, layer in layers:
                weight_name = f"{layer.original_name}.weight"
                effective = layer.effective_weight().detach().contiguous()
                yield weight_name, effective

        loaded = runner_model.load_weights(materialized_weights())
        if loaded is not None:
            expected = {
                self._vllm_destination_name(f"{layer.original_name}.weight")
                for _, layer in layers
            }
            missing = expected.difference(loaded)
            if missing:
                preview = ", ".join(sorted(missing)[:5])
                raise RuntimeError(
                    f"vLLM did not report loading {len(missing)} destination weights: {preview}"
                )
        self.llm.reset_prefix_cache()
        torch.cuda.empty_cache()
        return len(layers)

    @staticmethod
    def _vllm_destination_name(weight_name: str) -> str:
        """Map HF projection shard names to vLLM's packed Qwen names."""
        for source in ("q_proj", "k_proj", "v_proj"):
            weight_name = weight_name.replace(f".{source}.", ".qkv_proj.")
        for source in ("gate_proj", "up_proj"):
            weight_name = weight_name.replace(f".{source}.", ".gate_up_proj.")
        return weight_name


class VLLMLoRARolloutBackend(VLLMRolloutBackend):
    """vLLM sampler that hot-reloads TinyLoRA as a native LoRA overlay.

    The frozen rollout model may be quantized (including ModelOpt NVFP4).  No
    quantized base tensor is rewritten: after every learner update, the tiny
    residual is exported as rank-r A/B tensors and vLLM installs that adapter
    over its existing quantized kernels.
    """

    sync_before_first_rollout = True

    def __init__(
        self,
        model_name: str,
        *,
        adapter_root: str | Path,
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int,
        max_lora_rank: int = 8,
        trust_remote_code: bool = False,
        engine_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            trust_remote_code=trust_remote_code,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            engine_kwargs=engine_kwargs,
        )
        self.model_name = model_name
        self.adapter_root = Path(adapter_root)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        self._sync_index = 0
        self._lora_request = None

    def _active_lora_request(self) -> object | None:
        return self._lora_request

    @torch.no_grad()
    def sync_tinylora(self, model: nn.Module) -> int:
        from vllm.lora.request import LoRARequest

        self._sync_index += 1
        adapter_id = self._sync_index
        adapter_dir = self.adapter_root / f"step-{adapter_id:06d}"
        export_vllm_lora(
            model,
            adapter_dir,
            base_model_name=self.model_name,
        )
        request = LoRARequest(
            lora_name=f"tinylora-step-{adapter_id}",
            lora_int_id=adapter_id,
            lora_path=str(adapter_dir),
            base_model_name=self.model_name,
        )

        previous = self._lora_request
        if previous is not None:
            removed = self.llm.llm_engine.remove_lora(previous.lora_int_id)
            if not removed:
                raise RuntimeError(f"vLLM failed to remove LoRA id {previous.lora_int_id}")
        loaded = self.llm.llm_engine.add_lora(request)
        if not loaded:
            raise RuntimeError(f"vLLM failed to load LoRA id {adapter_id} from {adapter_dir}")
        self._lora_request = request
        self.llm.reset_prefix_cache()
        torch.cuda.empty_cache()
        return sum(1 for _ in iter_tinylora_layers(model))


def ensure_single_gpu_process_group() -> None:
    """Initialize the one-rank group required by vLLM's external launcher."""
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)


def destroy_process_group() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
