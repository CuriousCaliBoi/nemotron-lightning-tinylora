"""Deterministic verifiers used by the research training loop."""

from __future__ import annotations

import re


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)")
VERL_FINAL_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
BOXED_RE = re.compile(r"\\boxed\{\s*([-+]?[$]?[\d,]+(?:\.\d+)?)\s*\}")
NUMBER_RE = re.compile(r"[-+]?[$]?[\d,]+(?:\.\d+)?")
VERL_SOLUTION_CLIP_CHARS = 300


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


def strict_reference_answer(answer: str) -> str | None:
    """Extract the GSM8K ground truth exactly as VERL preprocessing does."""

    match = VERL_FINAL_ANSWER_RE.search(answer)
    return match.group(1).replace(",", "") if match else None


def predicted_answer(completion: str) -> str | None:
    for pattern in (FINAL_ANSWER_RE, BOXED_RE):
        matches = pattern.findall(completion)
        if matches:
            return normalize_number(matches[-1])
    matches = NUMBER_RE.findall(completion)
    return normalize_number(matches[-1] if matches else None)


def strict_predicted_answer(completion: str) -> str | None:
    """Mirror VERL's canonical strict GSM8K response extraction."""

    clipped = completion[-VERL_SOLUTION_CLIP_CHARS:]
    matches = VERL_FINAL_ANSWER_RE.findall(clipped)
    return matches[-1].replace(",", "").replace("$", "") if matches else None


def gsm8k_reward(completion: str, answer: str, *, mode: str = "flexible") -> float:
    if mode == "strict":
        prediction = strict_predicted_answer(completion)
        gold = strict_reference_answer(answer)
    elif mode == "flexible":
        prediction = predicted_answer(completion)
        gold = reference_answer(answer)
    else:
        raise ValueError(f"unknown GSM8K reward mode: {mode}")
    return float(prediction is not None and gold is not None and prediction == gold)
