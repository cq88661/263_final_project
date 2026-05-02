"""
Builds prompts for all experimental conditions.

Zero-shot: plain question, no examples.
Few-shot:  demonstration examples prepended; mode='fixed' or 'matched'.
"""
from __future__ import annotations
import random
from typing import List


def format_question(item: dict) -> str:
    """Append lettered choices for multiple-choice items."""
    q = item["question"]
    if item.get("choices"):
        lines = "\n".join(f"{k}. {v}" for k, v in item["choices"].items())
        return f"{q}\n\nChoices:\n{lines}"
    return q


def build_zero_shot(item: dict) -> str:
    return f"Solve the following problem step by step.\n\nProblem: {format_question(item)}"


def build_few_shot(item: dict, examples: List[dict]) -> str:
    """
    Build a few-shot prompt.

    Examples that have an 'explanation' field use the DeepSeek-R1
    <think> block format.  BBH-style examples (no explanation) show
    only question + answer as a minimal demonstration.
    """
    parts = []
    for ex in examples:
        q           = format_question(ex)
        explanation = (ex.get("explanation") or "").strip()
        answer      = str(ex.get("answer") or "")
        if explanation:
            parts.append(
                f"Problem: {q}\n<think>\n{explanation}\n</think>\nAnswer: {answer}"
            )
        else:
            parts.append(f"Problem: {q}\nAnswer: {answer}")

    target_q   = format_question(item)
    shots_block = "\n\n".join(parts)
    return f"{shots_block}\n\nProblem: {target_q}"


def select_examples(
    item: dict,
    pool: List[dict],
    n_shot: int,
    mode: str = "fixed",
    seed: int = 42,
) -> List[dict]:
    """
    Select n_shot examples from pool.

    mode='fixed'   — draw from entire pool (domain-agnostic)
    mode='matched' — draw from same source as the target item
    """
    rng = random.Random(seed)

    if mode == "matched":
        candidates = [
            e for e in pool
            if e["source"] == item["source"] and e["id"] != item["id"]
        ]
    else:
        candidates = [e for e in pool if e["id"] != item["id"]]

    if not candidates:
        return []

    return rng.sample(candidates, min(n_shot, len(candidates)))
