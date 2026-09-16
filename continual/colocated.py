"""A rollout backend over the colocated in-process vLLM engine from RLtests.

This is the topology the research runs use: the loop process holds the BF16
learner and its own NVFP4 vLLM actor, and TinyLoRA updates are hot-loaded
as rank-2 LoRA overlays. It exposes the same publish/complete interface as
the HTTP backend so the loop does not care which one it drives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from tinylora_rl.rollout import VLLMLoRARolloutBackend, ensure_single_gpu_process_group


class ColocatedRolloutBackend:
    sync_before_first_rollout = False

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None,
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int,
        max_lora_rank: int = 8,
        lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        engine_kwargs: dict[str, Any] | None = None,
        adapter_root: str | Path = "/tmp/cl-colocated-adapters",
    ) -> None:
        ensure_single_gpu_process_group()
        kwargs = dict(engine_kwargs or {})
        if revision is not None:
            kwargs["revision"] = revision
        self.base_model = model_name
        self._engine = VLLMLoRARolloutBackend(
            model_name,
            adapter_root=adapter_root,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            max_lora_rank=max_lora_rank,
            lora_target_modules=lora_target_modules,
            engine_kwargs=kwargs,
        )
        self._requests: dict[str, Any] = {}
        self._next_id = 0
        self.served_model = model_name

    # ------------------------------------------------------------ lifecycle
    def wait_healthy(self, timeout_s: float = 0.0) -> None:
        return None

    def health(self) -> bool:
        return True

    def loaded_adapters(self, prefix: str) -> list[str]:
        return [name for name in self._requests if name.startswith(prefix)]

    # ------------------------------------------------------------- adapters
    def _add(self, name: str, path: str | Path) -> None:
        from vllm.lora.request import LoRARequest

        self._next_id += 1
        request = LoRARequest(
            lora_name=name,
            lora_int_id=self._next_id,
            lora_path=str(path),
            base_model_name=self.base_model,
        )
        if not self._engine.llm.llm_engine.add_lora(request):
            raise RuntimeError(f"vLLM failed to load LoRA {name} from {path}")
        self._requests[name] = request

    def _remove(self, name: str) -> None:
        request = self._requests.pop(name, None)
        if request is None:
            return
        self._engine.llm.llm_engine.remove_lora(request.lora_int_id)

    def publish(self, name: str, path: str | Path) -> None:
        """Serve ``name``; the engine holds one adapter slot, so the previous one leaves first."""
        previous = [other for other in list(self._requests) if other != name]
        for other in previous:
            self._remove(other)
        if name not in self._requests:
            self._add(name, path)
        self.served_model = name
        self._engine.llm.reset_prefix_cache()

    def ensure_served(self, name: str, path: str | Path) -> str:
        self.publish(name, path)
        return name

    # ------------------------------------------------------------- sampling
    def complete(
        self,
        prompt_ids: Sequence[Sequence[int]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: int | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        target = model or self.served_model
        request = self._requests.get(target)
        if target != self.base_model and request is None:
            raise RuntimeError(f"adapter {target!r} is not loaded in the colocated engine")
        params = SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p, logprobs=1, seed=seed)
        outputs = self._engine.llm.generate(
            [{"prompt_token_ids": list(ids)} for ids in prompt_ids],
            params,
            use_tqdm=False,
            lora_request=request,
        )
        samples = []
        for prompt_index, request_output in enumerate(outputs):
            for output in request_output.outputs:
                token_ids = list(output.token_ids)
                if output.logprobs is None or len(output.logprobs) != len(token_ids):
                    raise RuntimeError("vLLM returned incomplete rollout logprobs")
                logprobs = [self._engine._chosen_logprob(token, candidates) for token, candidates in zip(token_ids, output.logprobs, strict=True)]
                samples.append(
                    {
                        "prompt_index": prompt_index,
                        "token_ids": token_ids,
                        "logprobs": logprobs,
                        "text": output.text,
                        "finish_reason": output.finish_reason,
                    }
                )
        return samples

    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        num_generations: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[tuple[int, list[int], list[float], str]]:
        samples = self.complete(prompts, n=num_generations, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
        return [(s["prompt_index"], s["token_ids"], s["logprobs"], s["text"]) for s in samples]

    def sync_tinylora(self, model: Any) -> int:
        raise RuntimeError("the continual loop publishes releases explicitly; sync_tinylora is not used")
