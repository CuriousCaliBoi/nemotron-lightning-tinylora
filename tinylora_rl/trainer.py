"""A compact, explicit PyTorch GRPO trainer for language models."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .adapters import iter_tinylora_layers, save_tinylora, trainable_parameter_count
from .objectives import GRPOBatch, compute_group_advantages, grpo_policy_loss, selected_completion_logprobs
from .rewards import gsm8k_reward
from .rollout import Trajectory, VLLMRolloutBackend


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 4
    prompts_per_step: int = 4
    generations_per_prompt: int = 4
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ppo_epochs: int = 1
    clip_epsilon: float = 0.2
    micro_batch_size: int = 2
    tis_mode: str = "token_clip"
    tis_minimum: float = 0.1
    tis_maximum: float = 10.0
    loss_reduction: str = "sample_mean"
    seed: int = 42
    save_every: int = 0


@dataclass
class TensorBatch:
    input_ids: Tensor
    attention_mask: Tensor
    response_mask: Tensor
    rollout_logprobs: Tensor

    def select(self, indices: Tensor) -> "TensorBatch":
        return TensorBatch(
            input_ids=self.input_ids[indices],
            attention_mask=self.attention_mask[indices],
            response_mask=self.response_mask[indices],
            rollout_logprobs=self.rollout_logprobs[indices],
        )


def collate_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    pad_token_id: int,
    device: torch.device,
) -> TensorBatch:
    max_length = max(len(item.prompt_ids) + len(item.completion_ids) for item in trajectories)
    batch_size = len(trajectories)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    response_mask = torch.zeros((batch_size, max_length), dtype=torch.float32, device=device)
    rollout_logprobs = torch.zeros((batch_size, max_length), dtype=torch.float32, device=device)
    for row, trajectory in enumerate(trajectories):
        tokens = trajectory.prompt_ids + trajectory.completion_ids
        prompt_length = len(trajectory.prompt_ids)
        end = len(tokens)
        input_ids[row, :end] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention_mask[row, :end] = 1
        response_mask[row, prompt_length:end] = 1
        rollout_logprobs[row, prompt_length:end] = torch.tensor(
            trajectory.rollout_logprobs, dtype=torch.float32, device=device
        )
    return TensorBatch(input_ids, attention_mask, response_mask, rollout_logprobs)


def policy_logprobs(model: nn.Module, batch: TensorBatch) -> Tensor:
    outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    return selected_completion_logprobs(outputs.logits, batch.input_ids, batch.response_mask)


@torch.no_grad()
def batched_policy_logprobs(model: nn.Module, batch: TensorBatch, micro_batch_size: int) -> Tensor:
    rows = []
    for start in range(0, batch.input_ids.shape[0], micro_batch_size):
        indices = torch.arange(start, min(start + micro_batch_size, batch.input_ids.shape[0]), device=batch.input_ids.device)
        rows.append(policy_logprobs(model, batch.select(indices)))
    return torch.cat(rows, dim=0)


class TinyLoRAGRPOTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: object,
        rollout: VLLMRolloutBackend,
        config: TrainConfig,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rollout = rollout
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.trajectories_path = self.output_dir / "trajectories.jsonl"

    def _prompt_ids(self, questions: Sequence[str]) -> list[list[int]]:
        instruction = (
            "Solve with a concise calculation. End with a final line exactly in the "
            "form: #### <number>."
        )
        encoded_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": question},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            for question in questions
        ]
        prompts = []
        for encoded in encoded_prompts:
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            if isinstance(encoded, Tensor):
                encoded = encoded.squeeze().tolist()
            prompts.append(list(encoded))
        return prompts

    def _sample_trajectories(self, questions: Sequence[str], answers: Sequence[str], step: int) -> list[Trajectory]:
        prompts = self._prompt_ids(questions)
        raw = self.rollout.generate(
            prompts,
            num_generations=self.config.generations_per_prompt,
            max_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed + step,
        )
        trajectories = []
        for group_id, completion_ids, rollout_logprobs, completion in raw:
            trajectories.append(
                Trajectory(
                    prompt_ids=prompts[group_id],
                    completion_ids=completion_ids,
                    rollout_logprobs=rollout_logprobs,
                    completion=completion,
                    reward=gsm8k_reward(completion, answers[group_id]),
                    group_id=group_id,
                )
            )
        return trajectories

    def _update(self, trajectories: Sequence[Trajectory]) -> dict[str, float]:
        device = next(self.model.parameters()).device
        tensor_batch = collate_trajectories(
            trajectories,
            pad_token_id=self.tokenizer.pad_token_id,
            device=device,
        )
        rewards = torch.tensor([item.reward for item in trajectories], dtype=torch.float32, device=device)
        group_ids = torch.tensor([item.group_id for item in trajectories], dtype=torch.long, device=device)
        advantages = compute_group_advantages(rewards, group_ids)
        self.model.eval()
        old_logprobs = batched_policy_logprobs(self.model, tensor_batch, self.config.micro_batch_size)
        # Hugging Face activates gradient checkpointing only in training mode.
        # The frozen base has no dropout in the target models, while TinyLoRA's
        # bank still receives gradients through non-reentrant checkpoints.
        self.model.train()
        grpo_batch = GRPOBatch(
            rewards=rewards,
            group_ids=group_ids,
            old_logprobs=old_logprobs.detach(),
            rollout_logprobs=tensor_batch.rollout_logprobs,
            response_mask=tensor_batch.response_mask,
        )

        last_metrics: dict[str, Tensor] = {}
        last_loss = torch.tensor(0.0, device=device)
        grad_norm = torch.tensor(0.0, device=device)
        batch_size = len(trajectories)
        for _ in range(self.config.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            permutation = torch.randperm(batch_size, generator=self.generator).to(device)
            accumulated_loss = torch.tensor(0.0, device=device)
            accumulated_metrics: dict[str, Tensor] = {}
            for start in range(0, batch_size, self.config.micro_batch_size):
                indices = permutation[start : start + self.config.micro_batch_size]
                micro = tensor_batch.select(indices)
                current = policy_logprobs(self.model, micro)
                selected_batch = GRPOBatch(
                    rewards=rewards[indices],
                    group_ids=group_ids[indices],
                    old_logprobs=grpo_batch.old_logprobs[indices],
                    rollout_logprobs=grpo_batch.rollout_logprobs[indices],
                    response_mask=grpo_batch.response_mask[indices],
                )
                loss, last_metrics = grpo_policy_loss(
                    current,
                    advantages[indices],
                    selected_batch,
                    clip_epsilon=self.config.clip_epsilon,
                    tis_mode=self.config.tis_mode,
                    tis_minimum=self.config.tis_minimum,
                    tis_maximum=self.config.tis_maximum,
                    reduction=self.config.loss_reduction,
                )
                if self.config.loss_reduction == "sample_mean":
                    micro_weight = len(indices) / batch_size
                else:
                    micro_weight = float(micro.response_mask.sum().item()) / max(
                        float(tensor_batch.response_mask.sum().item()), 1.0
                    )
                (loss * micro_weight).backward()
                accumulated_loss += loss.detach() * micro_weight
                for name, value in last_metrics.items():
                    accumulated_metrics[name] = accumulated_metrics.get(
                        name, torch.tensor(0.0, device=device)
                    ) + value * micro_weight
            last_loss = accumulated_loss
            last_metrics = accumulated_metrics
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        reward_groups = []
        for group_id in torch.unique(group_ids):
            reward_groups.append(float(rewards[group_ids == group_id].std(unbiased=False).item() == 0.0))
        completion_lengths = torch.tensor(
            [len(item.completion_ids) for item in trajectories], dtype=torch.float32
        )
        return {
            "loss": float(last_loss.item()),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "zero_variance_group_fraction": sum(reward_groups) / len(reward_groups),
            "completion_length_mean": float(completion_lengths.mean().item()),
            "completion_clipped_fraction": float(
                (completion_lengths == self.config.max_completion_length).float().mean().item()
            ),
            "grad_norm": float(grad_norm.item()),
            "adapter_norm": float(self.model.tinylora_bank.v.detach().float().norm().item()),
            **{name: float(value.item()) for name, value in last_metrics.items()},
        }

    def train(self, dataset: object) -> None:
        manifest = {
            "trainer": "from-scratch PyTorch GRPO",
            "base_model": getattr(getattr(self.model, "config", None), "_name_or_path", None),
            "adapter_config": asdict(self.model.tinylora_config),
            "train_config": asdict(self.config),
            "target_layers": sum(1 for _ in iter_tinylora_layers(self.model)),
            "trainable_parameters": trainable_parameter_count(self.model),
        }
        (self.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        order = torch.randperm(len(dataset), generator=self.generator).tolist()
        cursor = 0
        if getattr(self.rollout, "sync_before_first_rollout", False):
            synced = self.rollout.sync_tinylora(self.model)
            print(json.dumps({"initial_adapter_sync": synced}), flush=True)
        for step in range(1, self.config.steps + 1):
            if cursor + self.config.prompts_per_step > len(order):
                order = torch.randperm(len(dataset), generator=self.generator).tolist()
                cursor = 0
            indices = order[cursor : cursor + self.config.prompts_per_step]
            cursor += self.config.prompts_per_step
            rows = dataset.select(indices)
            started = time.perf_counter()
            trajectories = self._sample_trajectories(rows["question"], rows["answer"], step)
            with self.trajectories_path.open("a") as handle:
                for trajectory in trajectories:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "group_id": trajectory.group_id,
                                "question": rows["question"][trajectory.group_id],
                                "gold_answer": rows["answer"][trajectory.group_id],
                                "completion": trajectory.completion,
                                "completion_tokens": len(trajectory.completion_ids),
                                "reward": trajectory.reward,
                            }
                        )
                        + "\n"
                    )
            metrics = self._update(trajectories)
            synced = self.rollout.sync_tinylora(self.model)
            metrics.update(
                {
                    "step": step,
                    "step_seconds": time.perf_counter() - started,
                    "synced_weights": synced,
                }
            )
            with self.metrics_path.open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if self.config.save_every and step % self.config.save_every == 0:
                save_tinylora(self.model, self.output_dir / f"checkpoint-{step}")
        save_tinylora(self.model, self.output_dir / "final_adapter")
