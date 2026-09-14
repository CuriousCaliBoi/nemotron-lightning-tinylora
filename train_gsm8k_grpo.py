#!/usr/bin/env python3
"""Small, real GRPO+LoRA post-training run on GSM8K using colocated vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?[$]?[\d,]+(?:\.\d+)?")


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().replace("$", "").replace(",", "")
    try:
        number = float(normalized)
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() else f"{number:.8g}"


def reference_answer(answer: str) -> str | None:
    match = FINAL_ANSWER_RE.search(answer)
    return normalize_number(match.group(1) if match else None)


def completion_text(completion: Any) -> str:
    # TRL supplies strings for plain prompts and message lists for conversational
    # prompts. Supporting both keeps this script resilient across TRL releases.
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def predicted_answer(completion: Any) -> str | None:
    text = completion_text(completion)
    tagged = FINAL_ANSWER_RE.search(text)
    if tagged:
        return normalize_number(tagged.group(1))
    matches = NUMBER_RE.findall(text)
    return normalize_number(matches[-1] if matches else None)


def correctness_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    return [
        1.0 if predicted_answer(candidate) == reference_answer(gold) else 0.0
        for candidate, gold in zip(completions, answer, strict=True)
    ]


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    return [0.1 if "####" in completion_text(candidate) else 0.0 for candidate in completions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="/workspace/outputs/gsm8k-grpo-lora")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.samples, len(dataset))))

    instruction = (
        "Solve with a concise calculation. End with a final line exactly in the "
        "form: #### <number>."
    )
    dataset = dataset.map(
        lambda row: {
            "prompt": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": row["question"]},
            ],
            "answer": row["answer"],
        }
    )

    config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        seed=args.seed,
        bf16=True,
        learning_rate=1.0e-5,
        warmup_steps=0,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_completion_length=256,
        temperature=0.9,
        top_p=0.95,
        beta=0.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.20,
        vllm_enable_sleep_mode=True,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=4,
        report_to="none",
        save_strategy="no",
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    run_manifest = {
        "model": args.model,
        "dataset": "openai/gsm8k:main",
        "algorithm": "GRPO",
        "rollout_backend": "vLLM colocate",
        "max_steps": args.max_steps,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(json.dumps(run_manifest, indent=2), flush=True)

    result = trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["trainable_parameters"] = trainable
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    # Avoid NCCL choosing a host interface unsuitable for a single-GPU run.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
