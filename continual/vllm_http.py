"""A rollout backend that samples from a running vLLM server over HTTP.

It presents the interface ``tinylora_rl.trainer.TinyLoRAGRPOTrainer`` expects
from a rollout backend (``generate`` and ``sync_tinylora``) and adds the
runtime LoRA verbs the continual loop uses to publish adapters.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from tinylora_rl.adapters import export_vllm_lora, iter_tinylora_layers


class VLLMHTTPError(RuntimeError):
    pass


class VLLMHTTPRolloutBackend:
    """Sample completions from vLLM's OpenAI-compatible server with logprobs."""

    sync_before_first_rollout = False

    def __init__(
        self,
        base_url: str,
        *,
        base_model: str,
        served_model: str | None = None,
        timeout_s: float = 1800.0,
        retry_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.base_model = base_model
        self.served_model = served_model or base_model
        self.timeout_s = timeout_s
        self.retry_s = retry_s
        self._sync_index = 0

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        deadline = time.monotonic() + self.retry_s
        while True:
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout_s) as response:
                    payload = response.read()
                    return json.loads(payload) if payload else {}
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:2000]
                raise VLLMHTTPError(f"{method} {path} -> HTTP {error.code}: {detail}") from None
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                if time.monotonic() > deadline:
                    raise VLLMHTTPError(f"{method} {path} unreachable: {error}") from None
                time.sleep(5.0)

    def health(self) -> bool:
        try:
            self._request("GET", "/health", timeout=10.0)
            return True
        except VLLMHTTPError:
            return False

    def wait_healthy(self, timeout_s: float = 1800.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(self.base_url + "/health", timeout=5.0)
                return
            except Exception:
                time.sleep(5.0)
        raise VLLMHTTPError(f"vLLM at {self.base_url} did not become healthy within {timeout_s:.0f}s")

    def models(self) -> list[str]:
        payload = self._request("GET", "/v1/models", timeout=60.0)
        return [item["id"] for item in payload.get("data", [])]

    # ------------------------------------------------------------- adapters
    def load_adapter(self, name: str, path: str | Path) -> None:
        self._request("POST", "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)}, timeout=600.0)
        if name not in self.models():
            raise VLLMHTTPError(f"vLLM did not list adapter {name!r} after loading it from {path}")

    def unload_adapter(self, name: str) -> None:
        try:
            self._request("POST", "/v1/unload_lora_adapter", {"lora_name": name}, timeout=120.0)
        except VLLMHTTPError as error:
            if "404" in str(error) or "not found" in str(error).lower():
                return
            raise

    def loaded_adapters(self, prefix: str) -> list[str]:
        return [name for name in self.models() if name.startswith(prefix)]

    def publish(self, name: str, path: str | Path, *, prefix: str = "cl-") -> None:
        """Serve ``name``: load it first (two slots), then drop every other adapter of ours."""
        if name not in self.models():
            self.load_adapter(name, path)
        self.served_model = name
        for other in self.loaded_adapters(prefix):
            if other != name:
                self.unload_adapter(other)

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
        """Return one entry per sample: prompt_index, token_ids, logprobs, text, finish_reason."""
        body: dict[str, Any] = {
            "model": model or self.served_model,
            "prompt": [list(ids) for ids in prompt_ids],
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
        }
        if seed is not None:
            body["seed"] = seed
        payload = self._request("POST", "/v1/completions", body)
        samples = []
        for choice in sorted(payload["choices"], key=lambda item: item["index"]):
            logprobs = choice.get("logprobs") or {}
            tokens = logprobs.get("tokens") or []
            values = logprobs.get("token_logprobs") or []
            if len(tokens) != len(values):
                raise VLLMHTTPError("vLLM returned mismatched token and logprob lengths")
            token_ids = []
            for token in tokens:
                if not isinstance(token, str) or not token.startswith("token_id:"):
                    raise VLLMHTTPError(f"unexpected token encoding {token!r}; is return_tokens_as_token_ids honored?")
                token_ids.append(int(token[len("token_id:"):]))
            if any(value is None for value in values):
                raise VLLMHTTPError("vLLM returned a null logprob for a sampled token")
            samples.append(
                {
                    "prompt_index": choice["index"] // n,
                    "token_ids": token_ids,
                    "logprobs": [float(value) for value in values],
                    "text": choice.get("text", ""),
                    "finish_reason": choice.get("finish_reason"),
                }
            )
        expected = len(prompt_ids) * n
        if len(samples) != expected:
            raise VLLMHTTPError(f"vLLM returned {len(samples)} samples, expected {expected}")
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
        """The trainer-facing contract: (group_id, completion_ids, rollout_logprobs, text)."""
        samples = self.complete(
            prompts,
            n=num_generations,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return [
            (sample["prompt_index"], sample["token_ids"], sample["logprobs"], sample["text"])
            for sample in samples
        ]

    # --------------------------------------------------------------- export
    def sync_tinylora(self, model: Any, *, adapter_dir: str | Path | None = None, name: str | None = None) -> int:
        """Export the model's TinyLoRA update as a PEFT LoRA and serve it.

        The continual loop calls this with an explicit release directory and
        adapter name; the trainer's own ``train`` loop would call it bare.
        """
        self._sync_index += 1
        adapter_dir = Path(adapter_dir) if adapter_dir is not None else Path("/tmp/cl-sync") / f"step-{self._sync_index:06d}"
        name = name or f"cl-sync-{self._sync_index}"
        export_vllm_lora(model, adapter_dir, base_model_name=self.base_model)
        previous = self.served_model
        self.load_adapter(name, adapter_dir)
        self.served_model = name
        if previous != self.base_model and previous != name:
            self.unload_adapter(previous)
        return sum(1 for _ in iter_tinylora_layers(model))
