"""Central configuration — all paths and constants live here."""
from pathlib import Path

PIPELINE_DIR  = Path(__file__).parent.parent
PROJECT_ROOT  = PIPELINE_DIR.parent

DATA_DIR          = PROJECT_ROOT / "data"
TRAIN_PATH        = DATA_DIR / "train" / "final_dataset.json"
TEST_PATH         = DATA_DIR / "test" / "test_dataset.json"
RANKED_TRAIN_PATH = DATA_DIR / "ranked_train.json"

RESULTS_DIR     = PIPELINE_DIR / "results"
CHECKPOINTS_DIR = PIPELINE_DIR / "checkpoints"
OPTUNA_DB_PATH  = RESULTS_DIR / "optuna.db"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

MODEL_ID      = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"
TEMPERATURE   = 0.7
MAX_TOKENS    = 2048
WANDB_PROJECT = "cs263-pipeline"
