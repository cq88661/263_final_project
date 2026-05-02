"""
Full pipeline runner — executes all 4 steps unattended.

Usage:
    conda run -n llm-testing python run_pipeline.py
    conda run -n llm-testing python run_pipeline.py --n-trials 10
    conda run -n llm-testing python run_pipeline.py --skip-hpo

Steps:
    Pre  — rank training samples by quality (writes data/ranked_train.json)
    1    — zero-shot baseline on full test set
    2    — few-shot ablation (base model, fixed vs. matched, 3- and 5-shot)
    3    — LoRA fine-tuning (Optuna HPO + k=1,2,3 quality ablations)
    4    — few-shot scaling (base vs. best fine-tuned, 1/3/5-shot)
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = sys.executable


def run(script: Path, extra_args: list[str] | None = None) -> None:
    cmd = [PYTHON, str(script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("="*60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {script.name} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full CS263 pipeline unattended")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Optuna HPO trials for Step 3 (default: 10)")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="Skip HPO in Step 3 and use default hyperparameters")
    parser.add_argument(
        "--start-from",
        choices=["pre", "1", "2", "3", "4"],
        default="pre",
        help="Resume from a specific step (default: pre)",
    )
    args = parser.parse_args()

    steps = {
        "pre": ROOT / "pipeline" / "src"     / "sample_ranker.py",
        "1":   ROOT / "pipeline" / "scripts" / "run_step1.py",
        "2":   ROOT / "pipeline" / "scripts" / "run_step2.py",
        "3":   ROOT / "pipeline" / "scripts" / "run_step3.py",
        "4":   ROOT / "pipeline" / "scripts" / "run_step4.py",
    }

    order    = ["pre", "1", "2", "3", "4"]
    start_idx = order.index(args.start_from)

    step3_args = []
    if args.skip_hpo:
        step3_args.append("--skip-hpo")
    else:
        step3_args += ["--n-trials", str(args.n_trials)]

    for key in order[start_idx:]:
        extra = step3_args if key == "3" else []
        run(steps[key], extra)

    print("\n" + "="*60)
    print("Pipeline complete. Results in pipeline/results/")
    print("="*60)


if __name__ == "__main__":
    main()
