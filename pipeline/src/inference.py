"""
Inference wrapper for DeepSeek-R1-Distill-Qwen-7B via mlx-lm.

DeepSeek-R1 natively produces <think>...</think> blocks before its final answer.
We extract these directly — no heuristic CoT parsing needed.
"""
from __future__ import annotations
import re
from typing import Optional

from .config import MODEL_ID, TEMPERATURE, MAX_TOKENS

_model = None
_tokenizer = None
_current_adapter: Optional[str] = None


def load_model(
    model_id: str = MODEL_ID,
    adapter_path: Optional[str] = None,
) -> tuple:
    global _model, _tokenizer, _current_adapter
    from mlx_lm import load
    _model, _tokenizer = load(model_id, adapter_path=adapter_path)
    _current_adapter = adapter_path
    return _model, _tokenizer


def get_model(adapter_path: Optional[str] = None) -> tuple:
    """Return cached model, reloading only when adapter changes."""
    global _model, _tokenizer, _current_adapter
    if _model is None or _current_adapter != adapter_path:
        load_model(MODEL_ID, adapter_path=adapter_path)
    return _model, _tokenizer


def run_inference(
    prompt: str,
    adapter_path: Optional[str] = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict:
    """
    Run inference on a single prompt.

    Returns dict with keys:
      model_reasoning  — content inside <think>...</think>
      extracted_answer — text after </think>
      raw_output       — full model response
    """
    from mlx_lm import generate

    model, tokenizer = get_model(adapter_path=adapter_path)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    raw = generate(
        model, tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )

    result = extract_reasoning_and_answer(raw)
    result["raw_output"] = raw
    return result


def extract_reasoning_and_answer(raw_output: str) -> dict:
    """
    Split DeepSeek-R1 output into the think block and the final answer.

    The model always wraps its reasoning in <think>...</think>.
    If tags are absent (e.g. very short response), the full output is
    treated as reasoning with an empty extracted_answer.
    """
    match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        answer    = raw_output[match.end():].strip()
    else:
        reasoning = raw_output.strip()
        answer    = ""
    return {"model_reasoning": reasoning, "extracted_answer": answer}
