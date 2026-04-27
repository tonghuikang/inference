"""Run each saved line-jumping prompt through the server and grade the output.

For each N in {10, 100, 1000, 10000}:
  1. POST the saved prompts/{N}.txt prompt at temperature=0.
  2. Parse the assistant message for a comma-separated list of integers.
  3. Compare against the ground-truth path in prompts/{N}.path.txt.
  4. Report path-prefix correctness and the first divergence.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = os.environ.get("BENCH_URL", "http://localhost:1433/v1/chat/completions")
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-0.6B")
SIZES = [10, 100, 1000]
MAX_TOKENS = {10: 500, 100: 1500, 1000: 4000}


def post(prompt: str, max_tokens: int) -> tuple[str, float]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"], time.perf_counter() - t0


def parse_path(text: str, rules: dict[int, int]) -> list[int]:
    """Extract the model's traversed path by greedy-matching against the rules
    graph: walk through every integer in the post-<think> output in order, and
    accept it iff it's a valid next hop from the current cursor.

    This handles all the formats the model might use:
      - bare comma list:        "1, 6, 10, 9, 4"
      - arrow chain:            "1 → 6 → 10 → 9 → 4"
      - step prose:             "Step 1: at 1, rule says 6, next is 6. Step 2: at 6, ..."
      - reasoning + final list: "<think>...</think>\\n\\n1, 6, 10, 9, 4"

    We must START at line 1 (the prompt's mandated start). The first occurrence
    of "1" in the output anchors the walk; from there, each next integer that
    matches `rules[cursor]` advances; everything else is ignored as filler.
    """
    after = text.split("</think>", 1)[-1]
    nums = [int(m) for m in re.findall(r"\b\d+\b", after)]
    if not nums:
        return []

    # Find first occurrence of the start (line 1) — the model's own answer
    # boundary, not a stray integer in restated rules earlier in the output.
    try:
        start_idx = nums.index(1)
    except ValueError:
        return []

    path = [1]
    cursor = 1
    for n in nums[start_idx + 1 :]:
        target = rules.get(cursor)
        if n == target:
            path.append(n)
            cursor = n
    return path


def parse_rules(prompt_text: str) -> dict[int, int]:
    """Pull the `i: target` rules out of the prompt header."""
    rules: dict[int, int] = {}
    for line in prompt_text.splitlines():
        m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", line)
        if m:
            rules[int(m.group(1))] = int(m.group(2))
    return rules


def grade(N: int) -> dict:
    prompt = (ROOT / "prompts" / f"{N}.txt").read_text()
    expected = [int(x) for x in (ROOT / "prompts" / f"{N}.path.txt").read_text().strip().split(",")]
    rules = parse_rules(prompt)
    text, dt = post(prompt, MAX_TOKENS[N])
    parsed = parse_path(text, rules)

    correct_prefix = 0
    for got, exp in zip(parsed, expected, strict=False):
        if got == exp:
            correct_prefix += 1
        else:
            break

    return {
        "N": N,
        "expected_len": len(expected),
        "parsed_len": len(parsed),
        "correct_prefix": correct_prefix,
        "wall_s": dt,
        "first_divergence": (
            None if correct_prefix == min(len(parsed), len(expected))
            else (parsed[correct_prefix], expected[correct_prefix])
        ),
        "first_8_expected": expected[:8],
        "first_8_parsed": parsed[:8],
        "tail_text": text[-300:],
    }


def main() -> int:
    for N in SIZES:
        print(f"--- N={N} ---", flush=True)
        try:
            r = grade(N)
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
            print(f"  failed: {e}")
            continue
        ok = r["correct_prefix"] == r["expected_len"]
        verdict = "PASS" if ok else f"partial ({r['correct_prefix']}/{r['expected_len']})"
        print(f"  {verdict}  wall={r['wall_s']:.1f}s  parsed={r['parsed_len']} expected={r['expected_len']}", flush=True)
        print(f"  expected first 8: {r['first_8_expected']}", flush=True)
        print(f"  parsed   first 8: {r['first_8_parsed']}", flush=True)
        if r["first_divergence"]:
            print(f"  first divergence (got, exp): {r['first_divergence']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
