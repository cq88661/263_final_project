# CS263 Final Project — Few-Shot Prompting vs. LoRA Fine-Tuning on DeepSeek-R1

Investigates how few-shot prompting and LoRA fine-tuning affect chain-of-thought (CoT) reasoning quality of **DeepSeek-R1-Distill-Qwen-7B** across multi-domain scientific problems (math, physics, code, logical reasoning).

**Model:** `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`  
**Hardware:** Apple Silicon (mlx-lm backend, no CUDA required)

---

## Research Questions

1. Does few-shot prompting improve CoT reasoning, and do domain-matched examples help more than random ones?
2. Does LoRA fine-tuning on high-quality explanations improve CoT reasoning?
3. How does the number of SFT training examples (k=1, 2, 3 per category) scale with performance?

---

## Dataset

| Split | Items | Sources |
|-------|-------|---------|
| Train | 24    | MATH (algebra, counting, geometry) × 3; GPQA Physics × 3; Codeforces × 3; BBH × 9 |
| Test  | 36    | MATH × 15; GPQA (Physics, Biology, Chemistry) × 9; Codeforces × 3; BBH × 9 |

- **SFT-eligible** (have explanation): 15 train items — MATH, GPQA, Codeforces
- **Few-shot only** (no explanation): 9 BBH items

---

## Pipeline Overview

```
Pre  →  Step 1  →  Step 2  →  Step 3  →  Step 4
rank    0-shot    few-shot   fine-tune   shot scaling
train   base      base       3 models    base only
items   model     model      + infer
```

### Pre-step: Quality Ranking
Uses DeepSeek-R1 as a local LLM judge to score each of the 15 SFT-eligible training items (1–5 rubric) and assign a `quality_rank` within each category group.  
→ `data/ranked_train.json`

### Step 1: Zero-Shot Baseline
Base model with no examples. Establishes lower bound.  
→ `step1_zeroshot.json`

### Step 2: Few-Shot Prompting (Base Model)
Four conditions — selection strategy × shot count:
- **Fixed**: examples drawn from the full 24-item pool regardless of domain
- **Matched**: examples drawn only from the same source domain as the test item

→ `step2a_fixed_3shot`, `step2a_fixed_5shot`, `step2b_matched_3shot`, `step2b_matched_5shot`

### Step 3: LoRA Fine-Tuning + Inference
1. **Optuna HPO** (10 trials, TPE sampler) searches `lr`, `lora_rank`, `lora_layers`, `num_iters`, `batch_size`. Study persists in SQLite and survives crashes.
2. **3 ablation conditions** using the best hyperparameters:
   - `sft_k1_quality` — 1 best-ranked example per category group (5 items)
   - `sft_k2_quality` — 2 best-ranked examples per group (10 items)
   - `sft_k3_quality` — 3 best-ranked examples per group (15 items)
3. Zero-shot inference on the full test set for each checkpoint.

→ 3 checkpoints in `pipeline/checkpoints/`; 3 inference JSONs in `pipeline/results/`

### Step 4: Few-Shot Scaling (Base Model)
Base model at 1/3/5-shot (fixed selection) to measure how shot count scales independently of fine-tuning.  
→ `step4_base_1shot`, `step4_base_3shot`, `step4_base_5shot`

---

## Output Files

```
pipeline/results/
  step1_zeroshot.json             # base, 0-shot
  step2a_fixed_3shot.json         # base, fixed 3-shot
  step2a_fixed_5shot.json         # base, fixed 5-shot
  step2b_matched_3shot.json       # base, matched 3-shot
  step2b_matched_5shot.json       # base, matched 5-shot
  step3_sft_k1_quality.json       # fine-tuned k=1, 0-shot
  step3_sft_k2_quality.json       # fine-tuned k=2, 0-shot
  step3_sft_k3_quality.json       # fine-tuned k=3, 0-shot
  step4_base_1shot.json           # base, 1-shot
  step4_base_3shot.json           # base, 3-shot
  step4_base_5shot.json           # base, 5-shot
  adapter_map.json                # checkpoint name → path mapping
  optuna_trials.json              # HPO trial log

pipeline/checkpoints/
  sft_k1_quality/                 # LoRA adapter (~5–11 MB)
  sft_k2_quality/
  sft_k3_quality/
```

Each JSON contains 36 records:

```json
{
  "id": "MATH_TEST_1",
  "source": "hendrycks_math",
  "difficulty": "Level 5",
  "question": "...",
  "ground_truth_answer": "...",
  "model_reasoning": "<think> block — primary evaluation target",
  "extracted_answer": "text after </think>",
  "code_eval": null,
  "condition": "step1_zeroshot"
}
```

`code_eval` is populated only for Codeforces items: `{ code_extracted, language, compiles, tests_passed, tests_total }`.

---

## Setup

```bash
conda create -n llm-testing python=3.10 -y
conda activate llm-testing
pip install mlx mlx-lm optuna wandb pyyaml tqdm matplotlib datasets pytest

# Download model (~4 GB, first run only)
python -c "from mlx_lm import load; load('mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit')"

# Add wandb API key
echo "WANDB_API_KEY=your_key_here" > .env
```

---

## Running the Pipeline

```bash
# Full pipeline unattended (~4 hours on M4)
conda run -n llm-testing python run_pipeline.py

# Resume from a specific step if interrupted
conda run -n llm-testing python run_pipeline.py --start-from 3

# Skip Optuna HPO and use default hyperparameters
conda run -n llm-testing python run_pipeline.py --skip-hpo
```

Individual steps:

```bash
conda run -n llm-testing python pipeline/src/sample_ranker.py
conda run -n llm-testing python pipeline/scripts/run_step1.py
conda run -n llm-testing python pipeline/scripts/run_step2.py
conda run -n llm-testing python pipeline/scripts/run_step3.py --n-trials 10
conda run -n llm-testing python pipeline/scripts/run_step4.py
```

---

## Project Structure

```
263_final_project/
├── run_pipeline.py           # single entry point for full pipeline
├── data/
│   ├── train/final_dataset.json
│   ├── test/test_dataset.json
│   └── ranked_train.json     # generated by Pre-step
└── pipeline/
    ├── src/
    │   ├── config.py          # all paths and constants
    │   ├── inference.py       # mlx-lm wrapper with singleton model cache
    │   ├── prompt_builder.py  # zero-shot and few-shot prompt construction
    │   ├── sample_ranker.py   # LLM-judge quality scoring
    │   ├── train.py           # LoRA training and Optuna HPO
    │   ├── code_evaluator.py  # Codeforces sample I/O subprocess execution
    │   └── output_utils.py    # unified output record builder
    ├── scripts/
    │   ├── run_step1.py
    │   ├── run_step2.py
    │   ├── run_step3.py
    │   └── run_step4.py
    ├── tests/                 # 58 unit tests
    ├── results/               # inference JSONs (generated)
    └── checkpoints/           # LoRA adapters (generated)
```

---

## Tests

```bash
conda run -n llm-testing python -m pytest pipeline/tests/ -v
# 58 passed
```

Covers inference parsing, prompt construction, code evaluation, and sample ranking.
