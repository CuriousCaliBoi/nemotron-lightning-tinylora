"""Deterministic verifiers used by the research training loop."""

from __future__ import annotations

import re


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)")
BOXED_RE = re.compile(r"\\boxed\{\s*([-+]?[$]?[\d,]+(?:\.\d+)?)\s*\}")
NUMBER_RE = re.compile(r"[-+]?[$]?[\d,]+(?:\.\d+)?")


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        number = float(value.strip().replace("$", "").replace(",", ""))
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() else f"{number:.8g}"


def reference_answer(answer: str) -> str | None:
    match = FINAL_ANSWER_RE.search(answer)
    return normalize_number(match.group(1) if match else None)


def predicted_answer(completion: str) -> str | None:
    for pattern in (FINAL_ANSWER_RE, BOXED_RE):
        match = pattern.search(completion)
        if match:
            return normalize_number(match.group(1))
    matches = NUMBER_RE.findall(completion)
    return normalize_number(matches[-1] if matches else None)


def gsm8k_reward(completion: str, answer: str) -> float:
    return float(predicted_answer(completion) == reference_answer(answer))
