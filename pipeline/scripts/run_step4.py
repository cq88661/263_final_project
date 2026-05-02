"""
Step 4 — Few-Shot Scaling (Base Model Only)

Runs the base model at 1/3/5-shot to measure how shot count scales.
Fine-tuned models already run zero-shot in Step 3 — no few-shot needed there.

Usage:
    python pipeline/scripts/run_step4.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.src.config import TEST_PATH, TRAIN_PATH, RESULTS_DIR
from pipeline.src.inference import run_inference
from pipeline.src.output_utils import build_record
from pipeline.src.prompt_builder import build_few_shot, select_examples

SHOT_COUNTS = [1, 3, 5]


def run_scaling_condition(
    test_data: list[dict],
    train_pool: list[dict],
    n_shot: int,
) -> list[dict]:
    condition = f"step4_base_{n_shot}shot"
    results   = []

    for item in tqdm(test_data, desc=condition):
        examples = select_examples(item, train_pool, n_shot=n_shot, mode="fixed")
        prompt   = build_few_shot(item, examples)
        result   = run_inference(prompt)
        results.append(build_record(item, result, condition))

    out_path = RESULTS_DIR / f"{condition}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(results)} records → {out_path}")
    return results


def main():
    with open(TEST_PATH) as f:
        test_data = json.load(f)
    with open(TRAIN_PATH) as f:
        train_pool = json.load(f)

    print("\n── Base model few-shot scaling (1/3/5-shot) ──")
    for n in SHOT_COUNTS:
        run_scaling_condition(test_data, train_pool, n_shot=n)

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
