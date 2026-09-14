#!/usr/bin/env python3
"""Compare base and trained LoRA adapter with the same vLLM engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from train_gsm8k_grpo import predicted_answer, reference_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--adapter", default="/workspace/outputs/gsm8k-grpo-lora-4step/final_adapter"
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", default="/workspace/outputs/gsm8k-adapter-eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_dataset("openai/gsm8k", "main", split=f"test[:{args.samples}]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    instruction = (
        "Solve with a concise calculation. End with a final line exactly in the "
        "form: #### <number>."
    )
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=256)
    engine = LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=1024,
        gpu_memory_utilization=0.20,
    )

    base_outputs = engine.generate(prompts, sampling, use_tqdm=True)
    adapter_outputs = engine.generate(
        prompts,
        sampling,
        lora_request=LoRARequest("gsm8k-grpo", 1, args.adapter),
        use_tqdm=True,
    )
    details = []
    for row, base, adapted in zip(rows, base_outputs, adapter_outputs, strict=True):
        gold = reference_answer(row["answer"])
        base_text = base.outputs[0].text
        adapter_text = adapted.outputs[0].text
        base_prediction = predicted_answer(base_text)
        adapter_prediction = predicted_answer(adapter_text)
        details.append(
            {
                "question": row["question"],
                "gold": gold,
                "base_prediction": base_prediction,
                "adapter_prediction": adapter_prediction,
                "base_correct": base_prediction == gold,
                "adapter_correct": adapter_prediction == gold,
                "output_changed": base_text != adapter_text,
                "base_output": base_text,
                "adapter_output": adapter_text,
            }
        )

    result = {
        "samples": len(details),
        "base_accuracy": sum(x["base_correct"] for x in details) / len(details),
        "adapter_accuracy": sum(x["adapter_correct"] for x in details) / len(details),
        "outputs_changed": sum(x["output_changed"] for x in details),
        "details": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
