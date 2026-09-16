"""The continual learning loop.

One process owns the resident BF16 learner and drives the cycle:
sample from the served adapter, score, update, publish, record. Every
step resumes from the incumbent release (native adapter plus optimizer
state), so a restart continues exactly where the chain left off.

Two rollout topologies share the loop. ``ROLLOUT_BACKEND=colocated`` runs
the NVFP4 actor inside this process, as the research runs do.
``ROLLOUT_BACKEND=http`` samples from a separately running vLLM server that
allows runtime LoRA loading. Configuration comes from environment variables.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from continual.chain import ReleaseChain, atomic_write_json, utc_now
from tinylora_rl.adapters import export_vllm_lora, load_tinylora, save_tinylora, trainable_parameter_count
from tinylora_rl.rewards import gsm8k_reward
from tinylora_rl.trainer import TinyLoRAGRPOTrainer, TrainConfig


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


CONFIG: dict[str, Any] = {
    "state_dir": env("STATE_DIR", "/home/nimitz/cl-state/nemotron35-gsm8k"),
    "rollout_backend": env("ROLLOUT_BACKEND", "colocated"),
    "learner_model": env("LEARNER_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"),
    "learner_revision": env("LEARNER_REVISION", "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"),
    "rollout_model": env("ROLLOUT_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"),
    "rollout_revision": env("ROLLOUT_REVISION", "bee7596271d1495f6992ae224aefde4410e816b8"),
    "vllm_url": env("VLLM_URL", "http://127.0.0.1:30000"),
    "vllm_gpu_memory_utilization": float(env("VLLM_GPU_MEMORY_UTILIZATION", "0.23")),
    "vllm_max_model_len": int(env("VLLM_MAX_MODEL_LEN", "1024")),
    "vllm_kv_cache_memory_bytes": env("VLLM_KV_CACHE_MEMORY_BYTES", ""),
    "start_adapter": env(
        "START_ADAPTER",
        "/workspace/outputs/nemotron35-tinylora-lr-scale-replay-canary-20260916/lr1e-5_s1/final_adapter",
    ),
    "dataset_revision": env("DATASET_REVISION", "740312add88f781978c0658806c59bc2815b9866"),
    "train_split": env("TRAIN_SPLIT", "train[:-512]"),
    "eval_split": env("EVAL_SPLIT", "train[-512:]"),
    "eval_rows": int(env("EVAL_ROWS", "128")),
    "eval_every": int(env("EVAL_EVERY", "5")),
    "eval_max_tokens": int(env("EVAL_MAX_TOKENS", "512")),
    "rollback_drop": float(env("ROLLBACK_DROP", "0.10")),
    "prompts_per_step": int(env("PROMPTS_PER_STEP", "4")),
    "generations": int(env("GENERATIONS", "8")),
    "max_completion_length": int(env("MAX_COMPLETION_LENGTH", "256")),
    "temperature": float(env("TEMPERATURE", "1.0")),
    "top_p": float(env("TOP_P", "1.0")),
    "learning_rate": float(env("LEARNING_RATE", "1e-4")),
    "weight_decay": float(env("WEIGHT_DECAY", "0.0")),
    "max_grad_norm": float(env("MAX_GRAD_NORM", "1.0")),
    "ppo_epochs": int(env("PPO_EPOCHS", "1")),
    "clip_epsilon": float(env("CLIP_EPSILON", "0.2")),
    "micro_batch_size": int(env("MICRO_BATCH", "1")),
    "tis_mode": env("TIS_MODE", "token_clip"),
    "tis_minimum": float(env("TIS_MINIMUM", "0.1")),
    "tis_maximum": float(env("TIS_MAXIMUM", "10.0")),
    "loss_reduction": env("LOSS_REDUCTION", "sample_mean"),
    "prompt_style": env("PROMPT_STYLE", "concise"),
    "reward_mode": env("REWARD_MODE", "strict"),
    "seed": int(env("SEED", "42")),
    "max_steps": int(env("MAX_STEPS", "0")),
    "gradient_checkpointing": env("GRADIENT_CHECKPOINTING", "1") == "1",
}

ADAPTER_PREFIX = "cl-"


def log(message: str, **fields: Any) -> None:
    print(json.dumps({"time": utc_now(), "message": message, **fields}, default=str), flush=True)


class Status:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state: dict[str, Any] = {"started_at": utc_now(), "pid": os.getpid()}

    def update(self, **fields: Any) -> None:
        self.state.update(fields)
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.path, self.state)


class StopRequest:
    def __init__(self, stop_file: Path) -> None:
        self.stop_file = stop_file
        self.requested = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, *_: Any) -> None:
        self.requested = True
        log("stop requested by signal; finishing the current step")

    def should_stop(self) -> bool:
        return self.requested or self.stop_file.exists()


def adapter_name(release_id: str) -> str:
    return f"{ADAPTER_PREFIX}{release_id}"


def load_learner(config: dict[str, Any]) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config["learner_model"], revision=config["learner_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        config["learner_model"],
        revision=config["learner_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    )
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log("learner loaded", seconds=round(time.time() - started, 1), allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2))
    return model, tokenizer


def build_backend(config: dict[str, Any]) -> Any:
    if config["rollout_backend"] == "http":
        from continual.vllm_http import VLLMHTTPRolloutBackend

        backend = VLLMHTTPRolloutBackend(config["vllm_url"], base_model=config["rollout_model"])
        backend.wait_healthy()
        return backend
    if config["rollout_backend"] == "colocated":
        from continual.colocated import ColocatedRolloutBackend

        engine_kwargs: dict[str, Any] = {
            "kv_cache_dtype": env("VLLM_KV_CACHE_DTYPE", "fp8"),
            "moe_backend": env("VLLM_MOE_BACKEND", "marlin"),
            "mamba_backend": env("VLLM_MAMBA_BACKEND", "flashinfer"),
            "mamba_cache_mode": env("VLLM_MAMBA_CACHE_MODE", "align"),
        }
        if config["vllm_kv_cache_memory_bytes"]:
            engine_kwargs["kv_cache_memory_bytes"] = int(config["vllm_kv_cache_memory_bytes"])
        started = time.time()
        backend = ColocatedRolloutBackend(
            config["rollout_model"],
            revision=config["rollout_revision"] or None,
            gpu_memory_utilization=config["vllm_gpu_memory_utilization"],
            max_model_len=config["vllm_max_model_len"],
            seed=config["seed"],
            engine_kwargs=engine_kwargs,
            adapter_root=Path(config["state_dir"]) / "work" / "vllm_adapters",
        )
        log("colocated engine ready", seconds=round(time.time() - started, 1), allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2))
        return backend
    raise SystemExit(f"unknown ROLLOUT_BACKEND={config['rollout_backend']!r}")


def seed_release(chain: ReleaseChain, model: Any, config: dict[str, Any]) -> str:
    """Create release r000000 from the start adapter."""
    start = Path(config["start_adapter"])
    release_id, directory = chain.begin()
    native = directory / "final_adapter"
    native.mkdir()
    for name in ("adapter.safetensors", "adapter_config.json"):
        (native / name).write_bytes((start / name).read_bytes())
    load_tinylora(model, native)
    export_vllm_lora(model, directory / "peft_adapter", base_model_name=config["rollout_model"])
    chain.commit(
        release_id,
        {
            "step": 0,
            "parent_release_id": None,
            "operation": "seed",
            "source": str(start),
            "lora_name": adapter_name(release_id),
            "trainable_parameters": trainable_parameter_count(model),
            "config": config,
        },
    )
    log("seeded chain", release_id=release_id, source=str(start))
    return release_id


def evaluate(trainer: TinyLoRAGRPOTrainer, backend: Any, rows: Any, config: dict[str, Any], model_name: str) -> dict[str, Any]:
    questions, answers = rows["question"], rows["answer"]
    prompts = trainer._prompt_ids(questions)
    started = time.time()
    samples = backend.complete(prompts, n=1, max_tokens=config["eval_max_tokens"], temperature=0.0, model=model_name)
    results = []
    for sample in samples:
        index = sample["prompt_index"]
        results.append(
            {
                "index": index,
                "reward": gsm8k_reward(sample["text"], answers[index], mode=config["reward_mode"]),
                "completion_tokens": len(sample["token_ids"]),
                "finish_reason": sample["finish_reason"],
                "completion": sample["text"],
            }
        )
    correct = sum(item["reward"] for item in results)
    return {
        "model": model_name,
        "rows": len(results),
        "correct": int(correct),
        "accuracy": correct / max(len(results), 1),
        "clipped_fraction": sum(1 for item in results if item["finish_reason"] == "length") / max(len(results), 1),
        "seconds": round(time.time() - started, 1),
        "results": results,
    }


@torch.no_grad()
def load_bank(model: Any, native_dir: Path) -> list[float]:
    """Copy a saved release's 13 values into the resident learner; the frozen factors are shared."""
    from safetensors.torch import load_file

    tensors = load_file(str(native_dir / "adapter.safetensors"))
    values = tensors["bank.v"].to(model.tinylora_bank.v)
    model.tinylora_bank.v.copy_(values)
    return [round(float(value), 6) for value in values.flatten()]


def restore_release(chain: ReleaseChain, model: Any, trainer: TinyLoRAGRPOTrainer, release_id: str) -> None:
    directory = chain.release_dir(release_id)
    load_bank(model, directory / "final_adapter")
    optimizer_file = directory / "optimizer.pt"
    if optimizer_file.exists():
        trainer.optimizer.load_state_dict(torch.load(optimizer_file, map_location="cuda"))


def main() -> int:
    config = dict(CONFIG)
    state_dir = Path(config["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "records").mkdir(exist_ok=True)
    (state_dir / "evals").mkdir(exist_ok=True)
    status = Status(state_dir / "status.json")
    stop = StopRequest(state_dir / "STOP")
    atomic_write_json(state_dir / "config.last.json", config)
    log("loop starting", config=config)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    status.update(phase="loading_learner")
    model, tokenizer = load_learner(config)
    status.update(phase="starting_rollout_backend")
    backend = build_backend(config)
    chain = ReleaseChain(state_dir)

    head = chain.head()
    if head is None:
        status.update(phase="seeding")
        head = seed_release(chain, model, config)
    else:
        load_tinylora(model, chain.release_dir(head) / "final_adapter")
        log("resumed incumbent", release_id=head, step=chain.manifest(head).get("step"))
    served = backend.ensure_served(adapter_name(head), chain.release_dir(head) / "peft_adapter")

    train_config = TrainConfig(
        steps=1,
        prompts_per_step=config["prompts_per_step"],
        generations_per_prompt=config["generations"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        ppo_epochs=config["ppo_epochs"],
        clip_epsilon=config["clip_epsilon"],
        micro_batch_size=config["micro_batch_size"],
        tis_mode=config["tis_mode"],
        tis_minimum=config["tis_minimum"],
        tis_maximum=config["tis_maximum"],
        loss_reduction=config["loss_reduction"],
        seed=config["seed"],
        prompt_style=config["prompt_style"],
        reward_mode=config["reward_mode"],
    )
    trainer = TinyLoRAGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        rollout=backend,
        config=train_config,
        output_dir=state_dir / "work",
        dataset_split=config["train_split"],
        dataset_revision=config["dataset_revision"],
    )
    optimizer_path = chain.release_dir(head) / "optimizer.pt"
    if optimizer_path.exists():
        trainer.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cuda"))
        log("optimizer state restored", release_id=head)

    status.update(phase="loading_dataset")
    train_rows = load_dataset("openai/gsm8k", "main", split=config["train_split"], revision=config["dataset_revision"])
    eval_rows = load_dataset("openai/gsm8k", "main", split=config["eval_split"], revision=config["dataset_revision"]).select(
        range(config["eval_rows"])
    )
    log("datasets ready", train_rows=len(train_rows), eval_rows=len(eval_rows))

    step = int(chain.manifest(head).get("step", 0))
    evals_path = state_dir / "evals.jsonl"

    def best_accuracy(release_id: str | None) -> float:
        if release_id is None:
            return float("-inf")
        path = state_dir / "evals" / f"{release_id}.json"
        if not path.exists():
            return float("-inf")
        return float(json.loads(path.read_text())["accuracy"])

    def record_eval(release_id: str, at_step: int, reason: str) -> dict[str, Any]:
        result = evaluate(trainer, backend, eval_rows, config, adapter_name(release_id))
        (state_dir / "evals" / f"{release_id}.json").write_text(json.dumps(result, indent=1) + "\n")
        summary = {key: value for key, value in result.items() if key != "results"}
        summary.update({"release_id": release_id, "step": at_step, "reason": reason, "time": utc_now()})
        with evals_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        log("evaluation", **summary)
        best = chain.best()
        previous = best_accuracy(best)
        if best is None or result["accuracy"] > previous:
            chain.set_best(release_id)
            log("new best", release_id=release_id, accuracy=result["accuracy"], previous=previous)
        return summary

    if not (state_dir / "evals" / f"{head}.json").exists():
        status.update(phase="baseline_eval", head=head)
        record_eval(head, step, "baseline")

    status.update(phase="ready", head=head, served=served, step=step)
    consecutive_failures = 0
    while not stop.should_stop():
        if config["max_steps"] and step >= config["max_steps"]:
            log("max_steps reached", step=step)
            break
        step += 1
        picker = random.Random(config["seed"] * 1000003 + step)
        indices = picker.sample(range(len(train_rows)), config["prompts_per_step"])
        rows = train_rows.select(indices)
        started = time.time()
        release_id, directory = chain.begin()
        try:
            status.update(phase="rollout", step=step, head=head, served=served, candidate=release_id)
            trajectories = trainer._sample_trajectories(rows["question"], rows["answer"], step)
            rollout_seconds = time.time() - started
            with (state_dir / "records" / f"step-{step:06d}.jsonl").open("w") as handle:
                for trajectory in trajectories:
                    handle.write(
                        json.dumps(
                            {
                                "receipt": uuid.uuid4().hex,
                                "step": step,
                                "served_release": head,
                                "group_id": trajectory.group_id,
                                "question": rows["question"][trajectory.group_id],
                                "gold_answer": rows["answer"][trajectory.group_id],
                                "completion": trajectory.completion,
                                "completion_ids": trajectory.completion_ids,
                                "rollout_logprobs": trajectory.rollout_logprobs,
                                "reward": trajectory.reward,
                            }
                        )
                        + "\n"
                    )
            reward_mean = sum(item.reward for item in trajectories) / len(trajectories)
            status.update(phase="update", step=step, reward_mean=reward_mean)
            update_started = time.time()
            metrics = trainer._update(trajectories, step)
            update_seconds = time.time() - update_started

            status.update(phase="publish", step=step, candidate=release_id)
            save_tinylora(model, directory / "final_adapter")
            torch.save(trainer.optimizer.state_dict(), directory / "optimizer.pt")
            export_vllm_lora(model, directory / "peft_adapter", base_model_name=config["rollout_model"])
            name = adapter_name(release_id)
            backend.publish(name, directory / "peft_adapter")
            record = chain.commit(
                release_id,
                {
                    "step": step,
                    "parent_release_id": head,
                    "operation": "training",
                    "lora_name": name,
                    "metrics": {**metrics, "rollout_seconds": round(rollout_seconds, 1), "update_seconds": round(update_seconds, 1)},
                    "prompt_indices": indices,
                    "learning_rate": config["learning_rate"],
                },
            )
            served, head = name, release_id
            consecutive_failures = 0
            log(
                "step committed",
                step=step,
                release_id=release_id,
                content_id=record["content_id"],
                reward_mean=metrics.get("reward_mean"),
                loss=metrics.get("loss"),
                grad_norm=metrics.get("grad_norm"),
                adapter_norm=metrics.get("adapter_norm"),
                tis_truncated_fraction=metrics.get("tis_truncated_fraction"),
                rollout_seconds=round(rollout_seconds, 1),
                update_seconds=round(update_seconds, 1),
                step_seconds=round(time.time() - started, 1),
            )
            status.update(
                phase="committed", step=step, head=head, served=served, last_metrics=metrics, step_seconds=round(time.time() - started, 1)
            )

            if config["eval_every"] and step % config["eval_every"] == 0:
                status.update(phase="eval", step=step, head=head)
                summary = record_eval(head, step, "periodic")
                best = chain.best()
                drop = best_accuracy(best) - summary["accuracy"]
                if config["rollback_drop"] > 0 and best is not None and best != head and drop > config["rollback_drop"]:
                    status.update(phase="rollback", step=step, from_release=head, to_release=best)
                    rollback_id, rollback_dir = chain.begin()
                    chain.copy_content(best, rollback_dir)
                    restore_release(chain, model, trainer, best)
                    rollback_name = adapter_name(rollback_id)
                    backend.publish(rollback_name, rollback_dir / "peft_adapter")
                    chain.commit(
                        rollback_id,
                        {
                            "step": step,
                            "parent_release_id": head,
                            "operation": "rollback",
                            "restored_from": best,
                            "lora_name": rollback_name,
                            "reason": f"accuracy dropped {drop:.3f} below best",
                        },
                    )
                    served, head = rollback_name, rollback_id
                    log("rolled back", release_id=rollback_id, restored_from=best)
        except Exception as error:  # keep the loop alive across a failed step
            consecutive_failures += 1
            log("step failed", step=step, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-3000:])
            status.update(phase="error", step=step, error=f"{type(error).__name__}: {error}", consecutive_failures=consecutive_failures)
            if not chain.has_manifest(release_id):
                chain.discard(release_id)
            # An update may have been applied to the learner without being published; restore the incumbent.
            restore_release(chain, model, trainer, head)
            try:
                backend.publish(served, chain.release_dir(head) / "peft_adapter")
            except Exception as publish_error:
                log("could not re-serve incumbent", error=str(publish_error))
            step -= 1
            if consecutive_failures >= 5:
                log("too many consecutive failures; exiting")
                return 1
            time.sleep(30)
    status.update(phase="stopped", step=step, head=head, served=served)
    log("loop stopped", step=step, head=head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
