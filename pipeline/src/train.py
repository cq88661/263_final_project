"""
LoRA fine-tuning with Optuna hyperparameter search + wandb logging.

Fully automated: run `python pipeline/src/train.py --n-trials 10`
and walk away.  The study persists in SQLite so it survives crashes.

Training backend: mlx-lm (Apple Silicon, no CUDA required).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.src.config import (
    CHECKPOINTS_DIR, MODEL_ID, OPTUNA_DB_PATH,
    RANKED_TRAIN_PATH, RESULTS_DIR, WANDB_PROJECT,
)

# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "<|im_start|>user\n"
    "Solve step by step: {question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

_COMPLETION_TEMPLATE = (
    "<think>\n{explanation}\n</think>\n"
    "{answer_line}"
    "<|im_end|>"
)


def _format_prompt(item: dict) -> str:
    return _PROMPT_TEMPLATE.format(question=item["question"])


def _format_completion(item: dict) -> str:
    answer      = item.get("answer")
    answer_line = f"Answer: {answer}\n" if answer else ""
    return _COMPLETION_TEMPLATE.format(
        explanation = (item.get("explanation") or "").strip(),
        answer_line = answer_line,
    )


def _group_key(item: dict) -> str:
    source = item["source"]
    cat    = item.get("category") or ""
    if isinstance(cat, list):
        cat = cat[0] if cat else ""
    if source == "hendrycks_math":
        return f"MATH/{cat}"
    return source


def _group_items(items: list[dict]) -> dict[str, list]:
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(_group_key(item), []).append(item)
    return groups


def prepare_training_data(items: list[dict], output_dir: Path) -> Path:
    """
    Write train.jsonl and valid.jsonl to output_dir.

    Validation set: 1 item per group (lowest quality rank = held-out).
    This ensures every group is represented in training even at small k.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _group_items(items)

    train_items: list[dict] = []
    valid_items: list[dict] = []

    for group_items in groups.values():
        by_rank = sorted(group_items, key=lambda x: x.get("quality_rank", 99))
        if len(by_rank) > 1:
            valid_items.append(by_rank[-1])
            train_items.extend(by_rank[:-1])
        else:
            train_items.extend(by_rank)

    _write_jsonl(train_items, output_dir / "train.jsonl")
    # mlx-lm requires a valid.jsonl; when k=1 every group has one item so
    # none can be held out — reuse the last training item as a proxy.
    if not valid_items:
        valid_items = [train_items[-1]]
    # mlx-lm enforces: validation set size >= batch_size (max batch_size=4).
    # Replicate items until we have at least 4 to satisfy the constraint.
    while len(valid_items) < 4:
        valid_items += valid_items[:4 - len(valid_items)]
    _write_jsonl(valid_items, output_dir / "valid.jsonl")

    return output_dir


def _write_jsonl(items: list[dict], path: Path):
    with open(path, "w") as f:
        for item in items:
            record = {
                "prompt":     _format_prompt(item),
                "completion": _format_completion(item),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# mlx-lm LoRA training subprocess
# ---------------------------------------------------------------------------

def _write_lora_config(
    config_path: Path,
    data_dir: Path,
    adapter_path: Path,
    learning_rate: float,
    lora_rank: int,
    lora_layers: int,
    num_iters: int,
    batch_size: int,
) -> Path:
    """Write a YAML config file for mlx-lm lora training."""
    cfg = {
        "model":           MODEL_ID,
        "train":           True,
        "data":            str(data_dir),
        "num_layers":      lora_layers,
        "batch_size":      batch_size,
        "iters":           num_iters,
        "learning_rate":   learning_rate,
        "adapter_path":    str(adapter_path),
        "mask_prompt":     True,
        "val_batches":     -1,
        "steps_per_eval":  max(10, num_iters // 5),
        "steps_per_report": 10,
        "lora_parameters": {"rank": lora_rank, "dropout": 0.0, "scale": float(lora_rank)},
    }
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return config_path


def run_lora_training(
    data_dir: Path,
    adapter_path: Path,
    learning_rate: float,
    lora_rank: int,
    lora_layers: int,
    num_iters: int,
    batch_size: int,
) -> float:
    """
    Launch mlx-lm LoRA training as a subprocess.
    Returns the final validation loss (inf on failure).
    """
    adapter_path.mkdir(parents=True, exist_ok=True)
    config_path = adapter_path / "train_config.yaml"

    _write_lora_config(
        config_path, data_dir, adapter_path,
        learning_rate, lora_rank, lora_layers, num_iters, batch_size,
    )

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "-c", str(config_path),
    ]

    print(f"    Training: lr={learning_rate:.1e}, rank={lora_rank}, "
          f"layers={lora_layers}, iters={num_iters}, bs={batch_size}")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if proc.returncode != 0:
        print(f"    [WARN] Training failed (rc={proc.returncode})\n{proc.stderr[-500:]}")
        return float("inf")

    return _parse_val_loss(proc.stderr + proc.stdout)


def _parse_val_loss(log: str) -> float:
    """Extract the last reported validation loss from mlx-lm training log."""
    matches = re.findall(r"Val loss\s+([\d.]+)", log)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"Loss\s+([\d.]+)", log)
    if matches:
        return float(matches[-1])
    return float("inf")


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _init_wandb(params: dict, trial_number: int) -> object:
    """Initialise a wandb run, falling back to offline mode gracefully."""
    import wandb
    mode = os.environ.get("WANDB_MODE", "online")
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"trial_{trial_number}",
            config=params,
            reinit=True,
            mode=mode,
        )
    except Exception:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"trial_{trial_number}",
            config=params,
            reinit=True,
            mode="offline",
        )
    return run


def make_objective(data_items: list[dict], runs_dir: Path):
    """Return an Optuna objective closure bound to the given training items."""

    def objective(trial) -> float:
        import wandb

        params = {
            "lr":          trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "lora_rank":   trial.suggest_categorical("lora_rank", [4, 8, 16]),
            "lora_layers": trial.suggest_categorical("lora_layers", [8, 16]),
            "num_iters":   trial.suggest_categorical("num_iters", [50, 100, 200]),
            "batch_size":  trial.suggest_categorical("batch_size", [2, 4]),
        }

        adapter_path = runs_dir / f"trial_{trial.number:03d}"

        run = _init_wandb(params, trial.number)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = prepare_training_data(data_items, Path(tmpdir))
            val_loss  = run_lora_training(
                data_dir      = data_path,
                adapter_path  = adapter_path,
                learning_rate = params["lr"],
                lora_rank     = params["lora_rank"],
                lora_layers   = params["lora_layers"],
                num_iters     = params["num_iters"],
                batch_size    = params["batch_size"],
            )

        wandb.log({"val_loss": val_loss, "trial": trial.number})
        wandb.finish()
        return val_loss

    return objective


# ---------------------------------------------------------------------------
# Optuna search
# ---------------------------------------------------------------------------

def run_optuna_search(
    data_items: list[dict],
    n_trials: int = 20,
) -> "optuna.Study":
    import optuna

    runs_dir = CHECKPOINTS_DIR / "optuna_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction   = "minimize",
        pruner      = optuna.pruners.MedianPruner(),
        storage     = f"sqlite:///{OPTUNA_DB_PATH}",
        study_name  = "cs263-sft",
        load_if_exists = True,
    )

    print(f"\nStarting Optuna search — {n_trials} trials, study: cs263-sft")
    study.optimize(make_objective(data_items, runs_dir), n_trials=n_trials)

    best = study.best_trial
    print(f"\n── Best trial: #{best.number} ──")
    print(f"  val_loss = {study.best_value:.4f}")
    print(f"  params   = {best.params}")

    # Save human-readable summary
    summary = [
        {"trial": t.number, "val_loss": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
    ]
    with open(RESULTS_DIR / "optuna_trials.json", "w") as f:
        json.dump(summary, f, indent=2)

    _save_optuna_plot(study)
    return study


def _save_optuna_plot(study):
    """Save a parallel-coordinates visualisation of the Optuna study."""
    try:
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as ov

        fig = ov.plot_parallel_coordinate(study)
        plt.tight_layout()
        out_path = RESULTS_DIR / "optuna_parallel_coordinates.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved Optuna plot → {out_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate Optuna plot: {e}")


# ---------------------------------------------------------------------------
# Ablation runs
# ---------------------------------------------------------------------------

def select_ablation_items(
    ranked_items: list[dict],
    k: int,
    ordering: str,
    seed: int = 42,
) -> list[dict]:
    """Pick k items per category group in quality or random order."""
    groups   = _group_items(ranked_items)
    selected = []
    for group_items in groups.values():
        if ordering == "quality":
            ordered = sorted(group_items, key=lambda x: x.get("quality_rank", 99))
        else:
            rng     = random.Random(seed)
            ordered = rng.sample(group_items, len(group_items))
        selected.extend(ordered[:k])
    return selected


def train_ablation_run(
    run_name: str,
    items: list[dict],
    best_params: dict,
) -> Path:
    """Fine-tune one ablation condition. Returns adapter path."""
    adapter_path = CHECKPOINTS_DIR / run_name
    print(f"\n── {run_name} ({len(items)} items) ──")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = prepare_training_data(items, Path(tmpdir))
        val_loss  = run_lora_training(
            data_dir      = data_path,
            adapter_path  = adapter_path,
            learning_rate = best_params["lr"],
            lora_rank     = best_params["lora_rank"],
            lora_layers   = best_params["lora_layers"],
            num_iters     = best_params["num_iters"],
            batch_size    = best_params["batch_size"],
        )

    print(f"  val_loss = {val_loss:.4f}")
    return adapter_path


_ABLATION_CONDITIONS = [
    ("sft_k1_quality", 1, "quality"),
    ("sft_k2_quality", 2, "quality"),
    ("sft_k3_quality", 3, "quality"),
]


def run_all_ablations(
    ranked_items: list[dict],
    best_params: dict,
) -> dict[str, Path]:
    """Train 3 ablation conditions (k=1,2,3 best-quality examples). Returns {run_name: adapter_path}."""
    adapters: dict[str, Path] = {}
    for run_name, k, ordering in _ABLATION_CONDITIONS:
        items              = select_ablation_items(ranked_items, k=k, ordering=ordering)
        adapters[run_name] = train_ablation_run(run_name, items, best_params)
    return adapters


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Optuna HPO + LoRA ablations")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of Optuna trials for HPO")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="Skip HPO and use default hyperparameters")
    args = parser.parse_args()

    if not RANKED_TRAIN_PATH.exists():
        print(f"[ERROR] {RANKED_TRAIN_PATH} not found. Run sample_ranker.py first.")
        sys.exit(1)

    with open(RANKED_TRAIN_PATH) as f:
        ranked_items = json.load(f)

    print(f"Loaded {len(ranked_items)} ranked training items.")

    if args.skip_hpo:
        best_params = {"lr": 1e-4, "lora_rank": 8, "lora_layers": 16, "num_iters": 100, "batch_size": 4}
        print("Skipping HPO — using default params:", best_params)
    else:
        # Run HPO on k=3 quality-ordered items (all 15)
        full_set = select_ablation_items(ranked_items, k=3, ordering="quality")
        study    = run_optuna_search(full_set, n_trials=args.n_trials)
        best_params = study.best_params

    adapters = run_all_ablations(ranked_items, best_params)

    # Save adapter map
    adapter_map = {name: str(path) for name, path in adapters.items()}
    with open(RESULTS_DIR / "adapter_map.json", "w") as f:
        json.dump(adapter_map, f, indent=2)

    print("\n── All ablations complete ──")
    for name, path in adapter_map.items():
        print(f"  {name:20s} → {path}")


if __name__ == "__main__":
    main()
