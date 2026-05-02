"""
Codeforces item evaluation pipeline:
  1. Extract code block from model output
  2. Parse sample I/O from the embedded problem description
  3. Execute code in a subprocess against sample test cases
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile
from typing import Optional

# Fenced code block (``` lang ... ```)
_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z+]*)\n(?P<code>.*?)```", re.DOTALL)

# Sample I/O scanner: finds consecutive Input / Output blocks after an "Example" header
_INPUT_OUTPUT_RE = re.compile(
    r"\bInput\b\s*\n+(?P<inp>.*?)\n\s*\bOutput\b\s*\n+(?P<out>.*?)(?=\n\s*\bInput\b|\Z)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_block(text: str) -> tuple[str, str]:
    """
    Return (code, language) from model output.
    Language is 'python', 'cpp', or 'unknown'.
    Falls back to heuristic language detection when no fence is present.
    """
    match = _FENCE_RE.search(text)
    if match:
        lang = _normalize_lang(match.group("lang").lower().strip())
        code = match.group("code").strip()
        return code, lang

    code = text.strip()
    return code, _detect_language(code)


def _normalize_lang(raw: str) -> str:
    if raw in {"cpp", "c++", "cxx"}:
        return "cpp"
    if raw in {"python", "py", "python3"}:
        return "python"
    return "unknown"


def _detect_language(code: str) -> str:
    if any(kw in code for kw in ["#include", "int main(", "using namespace"]):
        return "cpp"
    if any(kw in code for kw in ["def ", "import ", "print(", "input("]):
        return "python"
    return "unknown"


# ---------------------------------------------------------------------------
# Sample I/O parsing
# ---------------------------------------------------------------------------

def parse_sample_io(problem_description: str) -> list[tuple[str, str]]:
    """
    Extract (input_str, expected_output_str) pairs from a Codeforces
    problem description.  The description embeds sample cases in the form:

        Examples
        Input
        <test input>
        Output
        <expected output>

    Returns a (possibly empty) list of pairs.
    """
    example_match = re.search(r"\bExamples?\b", problem_description, re.IGNORECASE)
    if not example_match:
        return []

    section = problem_description[example_match.start():]
    pairs = []
    for m in _INPUT_OUTPUT_RE.finditer(section):
        inp = m.group("inp").strip()
        out = m.group("out").strip()
        if inp and out:
            pairs.append((inp, out))
    return pairs


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def run_code(
    code: str,
    language: str,
    test_input: str,
    timeout: int = 5,
) -> tuple[Optional[str], int]:
    """
    Execute code against test_input in a temp directory.

    Returns (stdout_stripped, returncode).
    Special return codes:
      -1  compilation error (C++)
      -2  unknown language
      -3  timeout
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if language == "python":
            src = os.path.join(tmpdir, "sol.py")
            with open(src, "w") as f:
                f.write(code)
            cmd = ["python3", src]

        elif language == "cpp":
            src     = os.path.join(tmpdir, "sol.cpp")
            binary  = os.path.join(tmpdir, "sol")
            with open(src, "w") as f:
                f.write(code)
            compile_proc = subprocess.run(
                ["g++", "-O2", "-o", binary, src],
                capture_output=True,
                timeout=30,
            )
            if compile_proc.returncode != 0:
                return None, -1
            cmd = [binary]

        else:
            return None, -2

        try:
            proc = subprocess.run(
                cmd,
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.stdout.strip(), proc.returncode
        except subprocess.TimeoutExpired:
            return None, -3


# ---------------------------------------------------------------------------
# Full item evaluation
# ---------------------------------------------------------------------------

def evaluate_item(item: dict, model_output: str) -> dict:
    """
    End-to-end evaluation for one Codeforces item.

    Parses the problem description for sample I/O, extracts a code block
    from the model output, runs it, and returns a structured result dict.
    """
    code, lang   = extract_code_block(model_output)
    test_pairs   = parse_sample_io(item["question"])

    if not test_pairs:
        return {
            "code_extracted": bool(code),
            "language":       lang,
            "compiles":       None,
            "tests_passed":   0,
            "tests_total":    0,
        }

    passed   = 0
    compiles = None
    for test_in, expected_out in test_pairs:
        stdout, rc = run_code(code, lang, test_in)
        if rc == -1:          # compile error — stop immediately
            compiles = False
            break
        if rc == -2:          # unknown language — can't run
            break
        compiles = True
        if stdout is not None and stdout.strip() == expected_out.strip():
            passed += 1

    return {
        "code_extracted": bool(code),
        "language":       lang,
        "compiles":       compiles,
        "tests_passed":   passed,
        "tests_total":    len(test_pairs),
    }
