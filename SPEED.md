# SPEED

Throughput numbers for the stripped-down server, measured 2026-04-26 on
**Spark** (NVIDIA GB10, sm_121, 128 GB unified memory, aarch64). Same
hardware that runs the docker `vllm.service`. Same harness as
`~/Desktop/setup/spark/bench_vllm.py` — `scripts/bench.py` here is a port.

## Methodology

- `/v1/completions` with `prompt` as a raw token-ID array (no tokenizer in the
  loop; prompts/completions are gibberish — only throughput matters).
- Each `(N, prefix_tokens)` cell sends N concurrent requests that **share one
  random prefix** of length `prefix_tokens`. The engine's prefix-hash dedup
  absorbs prefill cost after the first request; the cell measures
  decode throughput once prefill is amortised.
- Per-request output length is pinned with `min_tokens = max_tokens` and
  `ignore_eos = true` (vLLM extensions; we wired both into our completions
  endpoint so the comparison is apples-to-apples).
- `output_tokens = clamp(GEN_BUDGET / N, OUT_MIN, OUT_MAX)`. For this run
  `GEN_BUDGET = 4096`, `OUT_MIN = 64`, `OUT_MAX = 256`. Smaller than the
  reference `131_072 / [64, 1024]` because our pure-Python attention is
  ~50× slower; an apples-to-apples gen budget would burn an hour per cell.
- Cell value is **total decode tokens / second** =
  `sum(completion_tokens) / wall_time` across all N requests in the cell.
- Cold-prefill column: one request, fresh prefix, `max_tokens=1`. Wall time
  ≈ prefill cost (one decode step is tens of ms, negligible).

## Results

### Ours — `inference.server` with Qwen3-0.6B (pure-Python paged attention)

Applied:
- `enable_gqa=True` in SDPA (drops `repeat_interleave` of K/V).
- Cached `block_table_tensor` on each `Sequence` (rebuild only when block
  count changes, ~once per `block_size` decode steps).
- Conditional batched-decode SDPA when `B >= 8 AND B * max_seq_len <= 4096`.
- Per-step attention metadata (padded block tables, attn mask) computed once
  in `model_runner` and passed via a `contextvar` so each of the 28 layers
  doesn't recompute.

| prefix tokens |   N=1 |   N=4 |  N=16 |  N=64 |  N=256 |  N=1024 | prefill (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1 |  70 | 234 | 558 | 1116 |   —  |   —  | 0.02 |
|  4096 |  47 |  92 | 120 |  105 |   —  |   —  | 0.27 |
| 32768 |   — |   — |   — |   —  |   —  |   —  |   —  |
| 98304 |   — |   — |   — |   —  |   —  |   —  |   —  |

Cells past N=64 and L>=32768 not yet measured — pure-Python prefill at
those scales takes minutes-to-hours per cell with the current attention.

Pre-optimisation v1 numbers (for reference): 67 / 206 / 403 / **572** at
L=1 and 34 / 54 / 62 / **26** at L=4096. L=1 N=64 went up 1.95× (572 →
1116); L=4096 N=64 went up 4× (26 → 105).

### vLLM (Qwen3-0.6B) — same model, fused kernels

| prefix tokens |   N=1 |   N=4 |  N=16 |  N=64 |  N=256 |  N=1024 | prefill (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1 | 128 | 611 | 2100 | 6830 |   —  |   —  | 0.01 |
|  4096 |  98 | 408 | 1232 | 3836 |   —  |   —  | 0.12 |
| 32768 |   — |   — |   — |   — |   —  |   —  |   —  |
| 98304 |   — |   — |   — |   — |   —  |   —  |   —  |

### vLLM (gpt-oss-20b) — bigger model, different architecture

| prefix tokens |   N=1 |   N=4 |  N=16 |  N=64 |  N=256 |  N=1024 | prefill (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|     1 |  47 | 175 |  530 | 1706 |   —  |   —  | 0.03 |
|  4096 |  42 | 156 |  518 | 1386 |   —  |   —  | 0.55 |
| 32768 |   — |   — |   — |   — |   —  |   —  |   —  |
| 98304 |   — |   — |   — |   — |   —  |   —  |   —  |

## What the gap looks like (after the optimisations above)

vLLM-Qwen3 / Ours-Qwen3, same model, same hardware:

| prefix tokens |  N=1 |  N=4 | N=16 | N=64 | prefill |
| ---: | ---: | ---: | ---: | ---: | ---: |
|     1 | 1.8× | 2.6× |  3.8× |  6.1× | ~2× |
|  4096 | 2.1× | 4.4× | 10.3× | 36.5× | ~2.3× |

Gap closed from 147× → 37× on the worst cell (L=4096 N=64) and from 12× →
6.1× on L=1 N=64. We're now ~2× slower than vLLM on the easy cells, ~10×
on long-prefix concurrent decode. The remaining gap is the lack of a
fused PagedAttention kernel — gather + SDPA is fundamentally bandwidth-
bound on K/V reads, while vLLM's kernel overlaps gather with attention
math at register speed.

## Why we're slow

The whole engine is unfused Python, by design. Specifically:

1. **Attention is gather + SDPA per sequence in a Python `for` loop.**
   `src/inference/layers/attention.py:112-144` gathers each sequence's
   K/V from the block table, runs `scaled_dot_product_attention` on it,
   and scatters back. With N=64 and L=4096 the per-step Python overhead
   dominates — each step gathers `64 × 4097 = ~262k` KV positions through
   advanced indexing one sequence at a time.

2. **No flash-attn varlen.** The proper kernel would do all N sequences
   in one launch using a `block_table` argument. We don't ship one because
   flash-attn isn't trivially buildable on aarch64 + CUDA 13 (sm_121); the
   Triton fallback we'd have to write is non-trivial to keep fast.

3. **No CUDA graph capture.** Each decode step pays full Python launch
   overhead for every Linear, RMSNorm, RoPE call. vLLM captures the
   decode step into a CUDA graph and replays it.

4. **GQA is materialised, not fused.** `attention.py:124-126` does
   `repeat_interleave` on K and V to expand kv_heads to q_heads before
   SDPA, instead of letting the kernel handle GQA natively.

## Where the gap comes from in numbers

The dominant term at high N + long prefix is the per-step gather. Our
per-decode-step cost scales roughly as `O(N · L)` Python operations,
whereas vLLM's is `O(N · L)` *fused* — same complexity but ~50× lower
constant. That's the 147× cell at L=4096 N=64.

At L=1 the per-step gather is trivial; the gap there (12× at N=64) comes
from CUDA-graph + fused linear layers.

## How to close the gap (deferred work, impact-vs-effort ordered)

1. **Batched paged-attention decode.** `attention.py:107-146` today launches
   one SDPA per sequence; replace the decode branch (`is_prefill=False`,
   all `q_len==1`) with a single SDPA call:
   - Stack `block_tables` into a padded `(B, max_blocks)` int tensor
     (build once in `build_inputs`, not per layer).
   - `K_pad = k_cache[block_tables_padded]`, view as
     `(B, max_blocks*block_size, kv_heads, head_dim)`, mask via `seq_lens`.
   - Single SDPA over `(B, heads, 1, head_dim)` × `(B, heads, L_max, head_dim)`.
   - Use `enable_gqa=True` instead of `repeat_interleave`.
   Expected: 5-20× on the L=4096 N=64 cell. Effort: ~1 day. Keep the
   current per-seq path for prefill (variable q_len, causal).

2. **`enable_gqa=True` in SDPA** (`attention.py:124-126`). Torch 2.11
   accepts mismatched K/V head counts; a one-line change drops the
   `repeat_interleave` and saves a memory pass per layer per seq.
   Expected: 10-25% on decode. Effort: 30 min plus tolerance bump in
   `test_paged_attention_matches_unpaged_reference`.

3. **Move `build_inputs` off the Python hot path** (`model_runner.py:41-89`).
   Cache per-seq `block_table` as a CPU tensor on `Sequence`, grow in-place
   in `scheduler.post_step`, copy once per step with pinned-memory H2D.
   Replace `block_tables: list[Tensor]` with one `(B, max_blocks)` tensor
   (depends on (1)). Expected: 20-40% on N=1 decode (closes the 67-vs-128
   gap). Effort: half-day.

4. **`torch.compile(mode="reduce-overhead")` on `Qwen3DecoderLayer`**. Uses
   CUDA graphs internally; pad decode batches to fixed buckets (powers of 2
   up to `max_num_seqs`) so recompiles are bounded. Skip for prefill.
   Expected: 1.3-2× after (1-3). Effort: ~1 day, mostly debugging shape-
   graph breaks around the paged-attention call.

5. **Triton paged-attention kernel** — only if (1-4) aren't enough. Closes
   the remaining gap to vLLM. ~80 lines, sm_121 has Triton support in torch
   2.11 out of the box. Effort: 2-3 days.

flash-attn from source on aarch64+sm_121 is fragile (the build's arch
dispatcher needs patching, head_dim instantiation is sensitive); Triton is
the saner bet if you decide to write a fused kernel.

The KV cache infrastructure, scheduler, and observer are not the bottleneck
— the engine spends almost all of its time inside attention.

## Tests to add before optimising

`tests/test_paged_attention_matches_unpaged_reference` covers the static
case. Before any of (1-5) lands:

- `test_paged_attention_decode_matches_prefill` — prefill-then-decode vs
  prefill-of-N+1, last-token equality. Catches off-by-one in `seq_lens`.
- `test_batched_decode_matches_per_seq_decode` — varied `seq_lens`
  (e.g., [17, 64, 4096, 31]) through both paths, bf16 tolerance 2e-3.
  THE regression test for (1).
- `test_gqa_repeat_interleave_equiv_enable_gqa` — small fp32 shapes,
  tight tolerance.
- `test_sliding_window_attention` — `window` arg currently uncovered;
  gpt-oss-20b will need it.
- `test_block_table_growth_under_decode` — drive enough decode steps to
  cross a block boundary; assert KV lands at slot 0 of the new block.
- `test_preemption_then_recompute_matches_uninterrupted` — small KV pool,
  large N, force preemption, compare greedy outputs with/without it.
- Throughput regression harness — wire `bench.py` cells into pytest with
  a `--bench` flag, ±15% bands against a stored baseline.

## Block size sweep

Empirical sweep across `block_size ∈ {16, 32, 64, 128}` on the same
hardware, same engine code, same bench harness (`scripts/bench.py` with
`BENCH_PREFIX=1,1024 BENCH_CONC=1,16,64`). Cell value is decode tok/s.

| block_size |   L=1 N=1 |  L=1 N=16 |  L=1 N=64 | L=1024 N=1 | L=1024 N=16 | L=1024 N=64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16  | 71 | 663 | 1251 | 67 | 318 | 402 |
| **32**  | **73** | **694** | **1315** | **69** | **330** | **413** |
| 64  | 71 | 556 |  907 | 65 | 309 | 398 |
| 128 | 73 | 449 |  598 | 64 | 306 | 392 |

**Winner: `block_size=32`.** Beats 64 by ~45% on L=1 N=64 (1315 vs 907)
and stays within noise of 16 on the cells where it doesn't lead.

The original first-principles analysis (subagent) predicted 64+ should
win because "every decode step pays a Python kernel launch per block".
The empirical answer disagrees: at small `block_size`, more sequences
hit the batched-decode fast path (gated on `B * max_seq_len <= 4096`),
which more than compensates for any per-block overhead. The crossover
is sharp — 32 → 64 drops L=1 N=64 by 30%.

Switched the default to `block_size=32` (`config.py`, `serve.sh`,
`server.py`).

Knobs that change the picture:
- The batched-decode threshold (`_BATCH_KV_THRESHOLD=4096` in
  `attention.py`). Raising it lets bs=64 stay batched on more cells.
- Long-prefix workloads (L>=4096) flatten the curve — at L=4096 N=64 the
  batched path is off regardless of bs, so all four bs values converge.
- Prefix-dedup hit rate: smaller bs catches finer prefixes.

## YaRN extended context

`RotaryEmbedding` is built with `yarn_factor=4.0`,
`yarn_orig_max_pos=32768`, giving cos/sin tables of size `(131072, 128)`.
Verified unit-level: `apply_rotary` runs cleanly at positions 0, 1000,
32768, 40960, 100000, 131071 with stable output norms (~35-38 in bf16).
No RoPE out-of-bounds errors past the 40960 trained ceiling.

The 10000-line prompt (107k tokens) won't actually run end-to-end at the
default `num_kv_blocks=1024 × block_size=64 = 65536` token cache — the
prompt is bigger than the cache. Bumping `num_kv_blocks` past 1700 would
fit it, but pure-Python prefill on 107k tokens would take ~hours per
request, well past a sane interactive budget. The 1000-line case
(~9k tokens) fits comfortably and works.

## Functional grading: can the model actually solve the line-jumping prompts?

We graded Qwen3-0.6B at temperature=0 on the saved prompts under
`prompts/`. Each ground-truth path is in `prompts/{N}.path.txt`.

| N    | model output    | wall  | first divergence      |
| ---: | ---:            | ---:  | ---:                  |
| 10   | partial 1/5     | 24 s  | got `1`, expected `6` |
| 100  | partial 1/50    | 98 s  | got `100`, expected `93` |
| 1000 | (running)       | —     | —                     |

The model identifies the first hop (line 1's target shows up in its
thinking trace), but it doesn't follow the path; instead it lists the
rule numbers verbatim from the prompt. This is a model-capability limit,
not an engine bug — Qwen3-0.6B is too small for multi-step in-context
graph traversal. The 10-line case is on the edge: in earlier
puppeteer-driven runs the model produced "1→6→10→9→4" inside `<think>`
but didn't reformat into the requested comma-separated answer. The grade
script's parser is also naive (greedy regex over digits) and misclassifies
verbose reasoning. A capable model (Qwen3-1.7B+ or instruction-tuned
~3B+) would be expected to solve N=10 and N=100 cleanly.

Bottom line: the prompts are correctly constructed and the engine serves
them correctly. The 0.6B model isn't smart enough to be the right
demo backend for the *correctness* of the line-jumping task; it's still
fine for demonstrating *KV-cache behaviour* (prefill, prefix sharing,
eviction).

## Reproducing

```sh
./serve.sh                       # ours, port 1433
scripts/bench.py                 # defaults match the table above
```

For the vLLM rows:

```sh
docker run --rm -d --name vllm --gpus all --ipc=host --shm-size=16g \
  -p 127.0.0.1:8000:8000 \
  -v /srv/vllm/hf:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.6 --max-model-len 32768

BENCH_URL=http://localhost:8000/v1/completions \
BENCH_MODEL=Qwen/Qwen3-0.6B \
BENCH_PREFIX=1,4096 BENCH_CONC=1,4,16,64 \
BENCH_GEN=4096 BENCH_OUT_MIN=64 BENCH_OUT_MAX=256 \
scripts/bench.py
```

For gpt-oss-20b, swap `Qwen/Qwen3-0.6B` for `openai/gpt-oss-20b` in both
the docker invocation and the bench env. Bump
`--gpu-memory-utilization 0.75` and drop `--max-model-len`. Our impl
doesn't currently support gpt-oss-20b — see `src/inference/models/gpt_oss.py`
which raises `NotImplementedError` (deferred to v2).
