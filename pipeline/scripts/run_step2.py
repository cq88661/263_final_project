"""
Step 2 — Few-Shot Chain-of-Thought Prompting

Runs four conditions on the full test set:
  2a fixed  3-shot / 5-shot  — same examples for every question
  2b matched 3-shot / 5-shot — examples from the same domain

The example pool is the training set (all 24 items including BBH).
BBH items lack an explanation so they contribute question+answer only.
"""
from __future__ import annotations
import json
import sys
from itertools import product
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.src.config import TEST_PATH, TRAIN_PATH, RESULTS_DIR
from pipeline.src.inference import run_inference
from pipeline.src.prompt_builder import build_few_shot, select_examples
from pipeline.src.output_utils import build_record


def run_condition(
    test_data: list[dict],
    train_pool: list[dict],
    mode: str,
    n_shot: int,
):
    condition = f"step2{'a' if mode == 'fixed' else 'b'}_{mode}_{n_shot}shot"
    results   = []

    for item in tqdm(test_data, desc=condition):
        examples = select_examples(item, train_pool, n_shot=n_shot, mode=mode)
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

    for mode, n_shot in product(["fixed", "matched"], [3, 5]):
        run_condition(test_data, train_pool, mode=mode, n_shot=n_shot)

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
