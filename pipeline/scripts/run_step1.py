"""
Step 1 — Zero-Shot Baseline
Establishes raw model capability with no guidance (lower bound).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.src.config import TEST_PATH, RESULTS_DIR
from pipeline.src.inference import run_inference
from pipeline.src.prompt_builder import build_zero_shot
from pipeline.src.output_utils import build_record

CONDITION = "step1_zeroshot"


def main():
    with open(TEST_PATH) as f:
        test_data = json.load(f)

    results = []
    for item in tqdm(test_data, desc="Zero-shot"):
        prompt = build_zero_shot(item)
        result = run_inference(prompt)
        results.append(build_record(item, result, CONDITION))

    out_path = RESULTS_DIR / f"{CONDITION}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} records → {out_path}")


if __name__ == "__main__":
    main()
