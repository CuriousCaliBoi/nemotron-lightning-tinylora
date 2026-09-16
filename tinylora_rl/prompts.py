"""Canonical prompt builders shared by training and evaluation."""

from __future__ import annotations

from typing import Literal


PromptStyle = Literal["concise", "verl"]


def gsm8k_messages(question: str, style: PromptStyle = "concise") -> list[dict[str, str]]:
    """Return chat messages for one GSM8K problem.

    ``verl`` mirrors the prompt used by the canonical VERL GSM8K recipe.  A
    Qwen tokenizer supplies its default system message when only a user turn
    is present. ``concise`` preserves this repository's original prompt.
    """

    if style == "concise":
        return [
            {
                "role": "system",
                "content": (
                    "Solve with a concise calculation. End with a final line exactly in the "
                    "form: #### <number>."
                ),
            },
            {"role": "user", "content": question},
        ]
    if style == "verl":
        return [
            {
                "role": "user",
                "content": (
                    f'{question} Let\'s think step by step and output the final answer after "####".'
                ),
            }
        ]
    raise ValueError(f"unknown GSM8K prompt style: {style}")
