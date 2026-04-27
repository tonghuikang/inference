#!/usr/bin/env python3
"""Concurrency × prefix-length throughput sweep + cold prefill column.

Modelled directly on ~/Desktop/setup/spark/bench_vllm.py — same shape, same
shared-prefix workload, same output format. The only differences:

- Default URL is ours (port 1433).
- Modest grid sizes (our pure-Python paged attention is much slower than
  vLLM's fused kernels — the table is meant to be comparable shape, not
  comparable scale).
- Cold-prefill column is computed inline instead of a separate script.

Run:
    BENCH_MODEL=Qwen/Qwen3-0.6B python3 scripts/bench.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.request

threading.stack_size(512 * 1024)

URL = os.environ.get("BENCH_URL", "http://localhost:1433/v1/completions")
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-0.6B")
API_KEY = os.environ.get("VLLM_API_KEY", "")

# Sweep parameters — match ~/Desktop/setup/spark/bench_vllm.py one-for-one so
# the table shape is directly comparable. On pure-Python paged attention these
# are punishing (98 304-token prefix takes minutes per cell); cells beyond
# what the box can do in a reasonable time will simply timeout the urlopen
# and bubble up. Override via env vars if you want a smaller subset.
PREFIX_LENGTHS = [
    int(x) for x in os.environ.get("BENCH_PREFIX", "1,4096,32768,98304").split(",")
]
CONCURRENCIES = [
    int(x) for x in os.environ.get("BENCH_CONC", "1,4,16,64,256,1024").split(",")
]
GEN_BUDGET_PER_CELL = int(os.environ.get("BENCH_GEN", str(128 * 1024)))  # 131 072
OUTPUT_TOKENS_MIN = int(os.environ.get("BENCH_OUT_MIN", "64"))
OUTPUT_TOKENS_MAX = int(os.environ.get("BENCH_OUT_MAX", "1024"))

# Random IDs in a safe vocab range (Qwen3 vocab is 151,936; specials are at
# the top end, so we stay in the middle).
TOK_LO, TOK_HI = 1000, 100_000


def random_token_ids(n_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(TOK_LO, TOK_HI) for _ in range(n_tokens)]


def one(prompt_ids: list[int], output_tokens: int) -> tuple[float, int, int]:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stop": [],
    }).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "bench/1.0"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    dt = time.perf_counter() - t0
    u = data.get("usage") or {}
    return dt, u.get("prompt_tokens", len(prompt_ids)), u.get("completion_tokens", output_tokens)


def plan_requests(N: int, prefix_tokens: int, seed_base: int) -> list[tuple[list[int], int]]:
    shared_prefix = random_token_ids(prefix_tokens, seed=seed_base)
    out_tok = max(OUTPUT_TOKENS_MIN, min(OUTPUT_TOKENS_MAX, GEN_BUDGET_PER_CELL // N))
    return [(shared_prefix, out_tok) for _ in range(N)]


def sweep_cell(N: int, prefix_tokens: int, seed_base: int) -> tuple[float, int, int, int, float]:
    plan = plan_requests(N, prefix_tokens, seed_base)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(one, p, k) for (p, k) in plan]
        results = [f.result() for f in cf.as_completed(futs)]
    wall = time.perf_counter() - t0
    p_tok = sum(r[1] for r in results)
    c_tok = sum(r[2] for r in results)
    return c_tok / wall, len(plan), p_tok, c_tok, wall


def cold_prefill(prefix_tokens: int, seed: int) -> float:
    """One request, fresh prefix, max_tokens=1. Wall time ≈ prefill time."""
    prompt = random_token_ids(prefix_tokens, seed=seed)
    dt, _, _ = one(prompt, 1)
    return dt


def warmup() -> None:
    print("# warmup…", file=sys.stderr)
    one(random_token_ids(32, seed=99_000), 8)


def main() -> int:
    print(
        f"# URL={URL} MODEL={MODEL}",
        file=sys.stderr,
    )
    warmup()

    header = "prefix \\ N            | " + " | ".join(f"{N:>5d}" for N in CONCURRENCIES) + " | prefill (s)"
    print(header)
    print("-" * len(header))

    seed_base = 1
    for L in PREFIX_LENGTHS:
        # Cold prefill (separate cell so vLLM's prefix cache isn't warmed yet).
        prefill_s = cold_prefill(L, seed=seed_base + 50_000)

        row = [f"{L:>22d}"]
        for N in CONCURRENCIES:
            tps, n_req, p_tok, c_tok, wall = sweep_cell(N, L, seed_base)
            seed_base += n_req + 7
            row.append(f"{tps:>5.0f}")
            print(
                f"#   L={L:>5} N={N:>3}: {n_req:>3} reqs, p={p_tok:>7d}, "
                f"c={c_tok:>6d}, wall={wall:>6.1f}s, out_tps={tps:>6.1f}",
                file=sys.stderr,
            )
        print("  | ".join(row) + f" | {prefill_s:>6.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
