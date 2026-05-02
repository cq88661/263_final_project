"""
Step 3 — LoRA Fine-Tuning with Optuna HPO + Data Ablation

Full automated pipeline:
  1. Optuna searches for best hyperparameters (N trials, unattended)
  2. Best config applied to 3 ablation conditions (k=1, k=2, k=3 quality-ranked examples)
  3. Inference run on full test set for each fine-tuned checkpoint
  4. Results saved to results/step3_*.json

Usage:
    python pipeline/scripts/run_step3.py --n-trials 10
    python pipeline/scripts/run_step3.py --skip-hpo   # use defaults, skip search
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.src.config import (
    RANKED_TRAIN_PATH, RESULTS_DIR, TEST_PATH,
)
from pipeline.src.inference import run_inference
from pipeline.src.output_utils import build_record
from pipeline.src.prompt_builder import build_zero_shot
from pipeline.src.train import (
    run_all_ablations,
    run_optuna_search,
    select_ablation_items,
)


def run_inference_for_adapter(
    test_data: list[dict],
    adapter_path: Path,
    condition: str,
) -> list[dict]:
    results = []
    for item in tqdm(test_data, desc=condition):
        prompt = build_zero_shot(item)
        result = run_inference(prompt, adapter_path=str(adapter_path))
        results.append(build_record(item, result, condition))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Step 3: LoRA fine-tuning ablation")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--skip-hpo", action="store_true")
    args = parser.parse_args()

    if not RANKED_TRAIN_PATH.exists():
        print(f"[ERROR] {RANKED_TRAIN_PATH} not found. Run sample_ranker.py first.")
        sys.exit(1)

    with open(RANKED_TRAIN_PATH) as f:
        ranked_items = json.load(f)
    with open(TEST_PATH) as f:
        test_data = json.load(f)

    # ── Hyperparameter search ──────────────────────────────────────────────
    if args.skip_hpo:
        best_params = {
            "lr": 1e-4, "lora_rank": 8,
            "lora_layers": 16, "num_iters": 100, "batch_size": 4,
        }
        print("Skipping HPO. Using defaults:", best_params)
    else:
        full_set    = select_ablation_items(ranked_items, k=3, ordering="quality")
        study       = run_optuna_search(full_set, n_trials=args.n_trials)
        best_params = study.best_params

    # ── Train 3 ablation conditions (k=1,2,3 quality) ────────────────────
    adapters = run_all_ablations(ranked_items, best_params)

    # Save adapter map so run_step4.py can auto-detect the best checkpoint
    adapter_map = {name: str(path) for name, path in adapters.items()}
    adapter_map_path = RESULTS_DIR / "adapter_map.json"
    with open(adapter_map_path, "w") as f:
        json.dump(adapter_map, f, indent=2)
    print(f"\nSaved adapter map → {adapter_map_path}")

    # ── Inference for each checkpoint ─────────────────────────────────────
    for run_name, adapter_path in adapters.items():
        condition = f"step3_{run_name}"
        print(f"\nInference: {condition}")
        results   = run_inference_for_adapter(test_data, adapter_path, condition)
        out_path  = RESULTS_DIR / f"{condition}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(results)} records → {out_path}")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
