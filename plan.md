# Plan: close the gap to vLLM (≤ 1.1×) + reproduce nano-vllm + finish gpt-oss-20b

Status: in flight. This file says what's done and what isn't. Numbers are
from `SPEED.md`. Anything below labelled **TODO** is still open.

## Where we are

| Area                                            | State |
| ---                                             | ---   |
| OpenAI HTTP surface (`/v1/{models,completions,chat/completions}`) | done; `server.py` |
| KV cache: paged blocks, refcount, prefix-hash dedup, deferred LRU eviction | done; `engine/block_manager.py` |
| Continuous-batching scheduler with preemption    | done; `engine/scheduler.py` |
| Triton paged-attention decode kernel             | done; `layers/paged_attn_triton.py` (numerical match vs unpaged SDPA at atol=5e-3) |
| HTML observer (`/observer/`) — live block grid, click-to-decode, demo buttons, streaming, concurrent | done; `web/static/*` |
| YaRN extended context (factor=4 → 131k positions) | done; `layers/rotary.py` |
| Qwen3-0.6B model file                            | done; `models/qwen3.py` — produces coherent output |
| Bench harness mirroring `setup/spark/bench_vllm.py` | done; `scripts/bench.py` |
| Functional grading on the line-jumping prompts  | done; `scripts/grade_prompts.py` (N=10 PASS, N=100 partial 12/50 — model-capability limit, not engine bug) |
| **Match vLLM throughput within 1.1×**           | **TODO** — currently 2-20× behind (see SPEED.md) |
| **Reproduce nano-vllm bench number on this box** | **TODO** — flash-attn build needed |
| **gpt-oss-20b end-to-end with real weights**    | **TODO** — `models/gpt_oss.py` exists with MoE + sliding-window + MXFP4 dequant, but loader has not been validated against real weights, and attention sinks (gpt-oss-specific softmax bias) are not yet implemented |

Gap to vLLM right now (Qwen3-0.6B):

| prefix tokens |  N=1 |  N=4 | N=16 | N=64 |
| ---:          | ---: | ---: | ---: | ---: |
|     1         | 2.0× | 2.1× |  2.3× |  3.6× |
|  4096         | 2.6× | 3.1× |  5.0× | 20×  |

Goal: every cell ≤ 1.1×. Nano-vllm already matches vLLM in ~1,200 LOC, so
nothing about this hardware makes the target unreachable.

## Plan to close the gap to vLLM

Nano-vllm's ingredients we don't yet have wired up:
1. `flash_attn_with_kvcache` — fused paged-decode kernel.
2. `enforce_eager=False` — CUDA graph capture for the decode step.

Both are achievable on this hardware. Tracks A and B run in parallel.

### Track A — flash-attn

flash-attn is the cheapest way to match nano-vllm. The 2.8.3 source build
on aarch64+sm_121 is 1118 .cu files because nvcc emits sm_80/90/100/120
quadruple builds and includes the backward kernels.

Steps:

1. Patch `setup.py` in the cached source tree (path is in
   `flash_attn_build.log`) to:
   - Drop the backward sources (`flash_bwd_*.cu`) — we don't train.
   - Drop `flash_fwd_split_hdim*` if not used by `flash_attn_with_kvcache`
     for our shapes.
   - Emit only `compute_120/sm_120` (sm_121 runs sm_120 PTX). Set
     `cc_flag` to just the sm_120 entry; remove the others.
   - Keep only `head_dim=128 bf16` instantiations (Qwen3 head_dim).
   This typically cuts the .cu count from 1118 to <40.
2. Resume the build:
   `nohup MAX_JOBS=8 uv add flash-attn --no-build-isolation
   >>./flash_attn_build.log 2>&1 &`.
3. Once `flash_attn` imports, replace the Triton call in
   `layers/attention.py` decode path with
   `flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=seq_lens,
   block_table=padded_block_tables, softmax_scale=scale, causal=True)`.
4. Re-bench, append to SPEED.md.

### Track B — CUDA graph capture

The L=1 N=64 cell is launch-overhead bound. nano-vllm captures the decode
forward into a CUDA graph once per `(batch_size_bucket,
max_blocks_bucket)` and replays it.

Steps:

1. Pad block_tables in `engine/model_runner.build_inputs` to
   `next_pow2(ceil(max_seq_len / block_size))`.
2. Bucket batch size into {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024}; pad
   short batches with masked-out slots.
3. Compile each `Qwen3DecoderLayer` with `mode="reduce-overhead",
   dynamic=False` once per bucket. Cache the compiled artefacts.
4. Mark the Triton/flash-attn call as a custom op with
   `@torch._dynamo.disable` so Inductor doesn't trace through it.

### Track C — sharper kernel (only if flash-attn unbuildable)

If flash-attn can't be built and we stay on the Triton kernel:

- Tile the seq axis with multiple warps (BLOCK_N=64, cooperative
  reduction).
- `triton.autotune` over `num_warps`, `num_stages`, `BLOCK_N` per
  (head_dim, block_size).
- Keep `for blk_idx in range(0, n_blocks)` runtime, NOT
  `tl.static_range` — the static unroll blew up PTX size at L=4096.

### Track D — Python overhead cleanup

After A+B should be tight, profile with `torch.profiler` to confirm
CUDA-bound. Anything left:

- Cache `padded_block_tables` per Sequence and grow in place.
- Pin slot_mapping/positions buffers, copy_(non_blocking=True).

## Plan to ship gpt-oss-20b

`models/gpt_oss.py` is structurally complete but unverified.

1. Load real weights from
   `/srv/vllm/hf/hub/models--openai--gpt-oss-20b/snapshots/<sha>` and
   confirm no shape/key mismatches.
2. Pin down the MXFP4 layout — both `gate_up_proj_blocks` axis order
   and `(gate, up)` chunk order against the reference implementation.
3. Implement attention sinks (gpt-oss-specific learnable softmax bias).
   The Triton kernel needs an extra `sink` parameter folded into the
   denominator before division.
4. Numerical equivalence test against
   `transformers.AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")`
   on a fixed prompt at temperature=0. Tolerance ~1e-2 bf16. Track
   first-divergence layer when it fails.
5. Bench gpt-oss-20b on our server and append the row to SPEED.md
   alongside the existing vLLM-gpt-oss-20b row.

## Reproducing nano-vllm on this hardware

Once flash-attn is installed:

1. `uv add 'nano-vllm @ git+https://github.com/GeeeekExplorer/nano-vllm.git'
   --no-build-isolation`.
2. Adapt `/tmp/nanovllm-clone/nano-vllm/bench.py`: point `path` at
   `/srv/vllm/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha>`. Their
   bench needs `enforce_eager=False`.
3. Run `uv run python bench.py`.
4. Append the result row to SPEED.md§Results between the vLLM and ours
   rows.

## Acceptance criteria

- SPEED.md§Results has a nano-vllm row with **measured** numbers from
  this box (not their published 4070 numbers).
- "Ours vs vLLM" gap table in SPEED.md§"What the gap looks like" shows
  every cell ≤ 1.1× (ours ≥ 0.91 × vLLM tok/s).
- 16 tests still pass; ruff clean.
- gpt-oss-20b loaded end-to-end and producing logits within numerical
  agreement of `transformers.GptOssForCausalLM` on a fixed prompt.

## Sanity checks (every change)

1. `uv run --frozen pytest tests/test_triton_kernel.py` — numerical
   match vs unpaged SDPA at atol=5e-3.
2. `uv run --frozen pytest tests/` — full 16 tests; should be green.
3. `uv run --frozen ruff check src/ tests/ scripts/` — lint clean.
4. End-to-end coherence via `curl /v1/chat/completions` — eyeball that
   the model still emits reasonable text.

## Constraints

- `uv` only — never `pip` or `uv pip` (per `CLAUDE.md`).
- aarch64 + CUDA 13 + sm_121 — flash-attn does not ship a wheel for this
  combo; building from source needs the trim described in Track A.
- HF cache lives at `/srv/vllm/hf` (shared with the pre-existing docker
  vllm.service).
- `.gitignore` ignores `*.txt` except `prompts/*.txt`. Keep bench output
  in `/tmp`.
