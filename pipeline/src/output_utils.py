"""
Shared helper for building the unified output record written by every step.
"""
from __future__ import annotations
from .code_evaluator import evaluate_item


def build_record(item: dict, inference_result: dict, condition: str) -> dict:
    """
    Build the standard output record for teammate evaluation.

    model_reasoning  — primary evaluation target (content of <think> block)
    extracted_answer — secondary accuracy check (text after </think>)
    code_eval        — populated only for Codeforces items, null otherwise
    """
    code_eval = None
    if item["source"] == "Codeforces":
        code_eval = evaluate_item(item, inference_result.get("raw_output", ""))

    return {
        "id":                   item["id"],
        "source":               item["source"],
        "difficulty":           item.get("difficulty"),
        "question":             item["question"],
        "ground_truth_answer":  item.get("answer"),
        "model_reasoning":      inference_result["model_reasoning"],
        "extracted_answer":     inference_result["extracted_answer"],
        "code_eval":            code_eval,
        "condition":            condition,
    }
