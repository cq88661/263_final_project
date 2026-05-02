"""
Ranks SFT-eligible training items by explanation quality using DeepSeek-R1
as a local LLM judge.  Outputs data/ranked_train.json.

Usage:
    conda activate llm-testing
    python -m pipeline.src.sample_ranker
    # or from project root:
    python pipeline/src/sample_ranker.py
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.src.config import TRAIN_PATH, RANKED_TRAIN_PATH, MODEL_ID

_RUBRIC_PROMPT = """\
You are an expert AI researcher evaluating chain-of-thought explanation quality.

Rate the explanation below on a scale from 1 to 5:
  5 — Excellent: detailed step-by-step reasoning, no gaps, clearly leads to the answer
  4 — Good: mostly clear steps, minor gaps or redundancy
  3 — Acceptable: some steps present but logic is incomplete or partially unclear
  2 — Weak: large reasoning gaps or mostly restates the question
  1 — Poor: no meaningful reasoning or factually wrong throughout

QUESTION: {question}

EXPLANATION: {explanation}

STATED ANSWER: {answer}

Respond with a single digit (1-5) only. Do not explain."""


def score_explanation(item: dict, model, tokenizer) -> float:
    from mlx_lm import generate

    prompt = _RUBRIC_PROMPT.format(
        question=item["question"][:600],
        explanation=(item.get("explanation") or "")[:1200],
        answer=str(item.get("answer") or ""),
    )
    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = generate(model, tokenizer, prompt=formatted, max_tokens=64, temp=0.0, verbose=False)
    return _parse_score(raw)


def _parse_score(raw: str) -> float:
    # Strip think tags and look for a single digit 1-5
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\b([1-5])\b", clean) or re.search(r"\b([1-5])\b", raw)
    return float(m.group(1)) if m else 3.0


def _heuristic_score(item: dict) -> float:
    """Fallback when LLM scoring is unavailable: proxy by explanation length."""
    n = len(item.get("explanation") or "")
    if n > 800:  return 5.0
    if n > 400:  return 4.0
    if n > 200:  return 3.0
    if n > 50:   return 2.0
    return 1.0


def group_by_category(items: list[dict]) -> dict[str, list]:
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(_group_key(item), []).append(item)
    return groups


def _group_key(item: dict) -> str:
    source = item["source"]
    cat    = item.get("category") or ""
    if isinstance(cat, list):
        cat = cat[0] if cat else ""
    if source == "hendrycks_math":
        return f"MATH/{cat}"
    return source


def rank_dataset(
    sft_items: list[dict],
    model=None,
    tokenizer=None,
) -> list[dict]:
    """
    Score every item and assign quality_rank within each category group.
    Rank 1 = best quality within the group.
    Returns a new list of dicts (originals are not mutated).
    """
    scored: list[dict] = []
    groups = group_by_category(sft_items)

    for group_name, items in groups.items():
        print(f"  [{group_name}] scoring {len(items)} items...", flush=True)
        for item in items:
            score = (
                score_explanation(item, model, tokenizer)
                if model is not None
                else _heuristic_score(item)
            )
            scored.append({**item, "quality_score": score, "group_key": group_name})

    # Assign rank within each group (1 = highest score)
    for group_name in groups:
        group_items = [x for x in scored if x["group_key"] == group_name]
        sorted_group = sorted(group_items, key=lambda x: x["quality_score"], reverse=True)
        rank_map = {item["id"]: i + 1 for i, item in enumerate(sorted_group)}
        for item in scored:
            if item["id"] in rank_map:
                item["quality_rank"] = rank_map[item["id"]]

    # Clean up temporary key
    for item in scored:
        item.pop("group_key", None)

    return scored


def _print_summary(ranked: list[dict]):
    print("\n── Quality Ranking Summary ──")
    for item in sorted(ranked, key=lambda x: (x.get("quality_rank", 99), x["source"])):
        gk = _group_key(item)
        print(
            f"  Rank {item.get('quality_rank','-'):1} | "
            f"Score {item.get('quality_score', 0):.1f} | "
            f"{item['id']:15s} | {gk}"
        )


def main():
    print("Loading training data...")
    with open(TRAIN_PATH) as f:
        train_data = json.load(f)

    sft_eligible = [d for d in train_data if d.get("explanation")]
    print(f"SFT-eligible items: {len(sft_eligible)}")

    print(f"\nLoading model {MODEL_ID} for LLM-judge scoring...")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)

    print("\nScoring explanations...")
    ranked = rank_dataset(sft_eligible, model=model, tokenizer=tokenizer)

    with open(RANKED_TRAIN_PATH, "w") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {RANKED_TRAIN_PATH}")
    _print_summary(ranked)


if __name__ == "__main__":
    main()
