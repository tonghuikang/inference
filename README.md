# inference

A hackable, OpenAI-compatible LLM server modelled on
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). Currently serves
**Qwen3-0.6B** (dense, with YaRN-extended context to 131k tokens). The KV
cache allocator, block manager, scheduler, and paged attention are pure
readable Python.

The whole point: a live HTML observer at `/observer/` that renders every
physical KV block, color-coded by status, with hover/click to see the
decoded tokens — so you can *see* prefill, prefix sharing, deferred
eviction, and continuous batching happening in real time.

## Quick start

```sh
./serve.sh                              # http://localhost:1433
open http://localhost:1433/observer/    # KV grid + demo buttons
```

The observer page has demo buttons (`10 lines / 100 lines / 1000 lines /
shared-prefix pair`) that fire the saved line-jumping prompts under
`prompts/`. Multiple buttons can be in flight simultaneously — the engine
batches them in a single forward pass via continuous batching.

For the OpenAI surface: `POST /v1/chat/completions` and `/v1/completions`
work without auth by default. Bearer-token auth turns on if you set
`VLLM_API_KEY` in the environment.

## Repo layout

```
src/inference/
  config.py              # ModelConfig, EngineConfig, SamplingParams
  server.py              # FastAPI: /v1/{models,chat/completions,completions} + SSE
                         #          /observer/ static + /observer/ws + /observer/snapshot
                         #          /observer/decode (token-id → text)
                         #          /prompts/ static mount of saved demo prompts
  engine/
    llm_engine.py        # top loop: pull requests → schedule → run → emit tokens
    scheduler.py         # continuous batching: admission, prefill/decode mix, preempt
    block_manager.py     # KV block pool, refcount, xxhash prefix dedup, deferred LRU eviction
    sequence.py          # Sequence with cached block_table tensor
    model_runner.py      # batched input prep, forward, KV write-back
  layers/
    attention.py         # PagedAttention: per-seq path for small B / prefill,
                         #                 padded batched SDPA for decode B>=8;
                         #                 sliding-window variant available
    rotary.py            # RoPE + YaRN extension (factor=4 → 131k positions)
    rmsnorm.py
    sampler.py           # greedy / temperature / top-p / top-k
    moe.py, quant_mxfp4.py   # placeholders for gpt-oss-20b
  models/
    qwen3.py             # Qwen3ForCausalLM — dense, GQA, QK-norm
    gpt_oss.py           # STUB — raises NotImplementedError (deferred to v2)
  utils/
    loader.py            # safetensors load with per-arch name remap
    kv_observer.py       # pub/sub bus: alloc / hit / append / release / evict
  web/static/
    index.html, app.js, style.css   # KV block grid + demo buttons
prompts/
  10.txt, 100.txt, 1000.txt           # line-jumping graphs (`i: target` per line)
  10.path.txt, ...                     # ground-truth paths from line 1
scripts/
  build_prompts.py     # regenerate prompts/
  download_qwen.py     # snapshot_download Qwen3-0.6B into /srv/vllm/hf
  smoke_qwen3.py       # one-prompt sanity check (no HTTP)
  bench.py             # concurrency × prefix throughput sweep (mirrors setup/spark/bench_vllm.py)
  grade_prompts.py     # check model output against ground-truth paths
serve.sh               # nohup background server on port 1433 (no auth by default)
```

## Design

KV cache management is pure Python; performance is secondary to readability.

1. **Block manager** (`engine/block_manager.py`) — fixed-size KV blocks
   (default 64 tokens), refcount, FIFO free queue, `xxh64(parent_hash ||
   tokens)` for prefix dedup. Released blocks keep their content + hash and
   are only physically reset when something else needs the slot — so a
   finished request still helps the next one via prefix-hash hits.

2. **Scheduler** (`engine/scheduler.py`) — prefill-priority continuous
   batching; preempt + recompute when KV pool is exhausted under decode load
   (no swap-to-CPU).

3. **Paged attention** (`layers/attention.py`) — gather K/V from the block
   table and run `scaled_dot_product_attention` with `enable_gqa=True`. Two
   paths:
   - per-seq Python loop (prefill, and decode at B<8)
   - padded batched SDPA in one launch (decode at B≥8)

4. **YaRN** is enabled by default (`factor=4.0`, `original_max=32768` →
   131072 positions), so the model can handle prompts past the trained 40k
   ceiling without RoPE OOB errors.

5. **HTML observer** subscribes to a pub/sub event bus over WebSocket. Every
   alloc / hit / append / release / evict updates the grid live.

## Endpoints

- `GET  /v1/models`
- `POST /v1/completions` — supports `prompt` as a string or as a list of
  token IDs; honours `min_tokens` and `ignore_eos` (vLLM extensions).
- `POST /v1/chat/completions` — `messages` + chat template via
  `transformers.AutoTokenizer.apply_chat_template`. SSE streaming via
  `stream:true`.
- `GET  /observer/`        — KV grid UI
- `GET  /observer/snapshot`
- `WS   /observer/ws`
- `GET  /observer/decode?ids=1,2,3` — detokenize selected block contents
- `GET  /prompts/{N}.txt`  — saved demo prompts

## Running

```sh
uv sync                                  # install deps
HF_HOME=/srv/vllm/hf uv run python scripts/download_qwen.py
./serve.sh                               # foregrounds nothing — writes serve.log + serve.pid
uv run --frozen pytest tests/            # 14 tests, ~3 s
uv run --frozen ruff check src/ tests/ scripts/
```

## Throughput

See `SPEED.md` for the full concurrency × prefix table compared against
docker `vllm.service`. tl;dr at L=1 N=64 we hit ~830 tok/s on Qwen3-0.6B
(vLLM gets ~6800); at L=4096 N=64 the gap is much wider because our
attention is gather + SDPA in Python rather than a fused kernel
(flash-attn / paged-attention Triton). Closing the gap requires writing or
linking a fused decode kernel — listed as deferred work in `SPEED.md`.

## Features (mapped to requirements)

| Requirement                                                  | Where                                                          |
| ---                                                          | ---                                                            |
| Live HTML observer of every KV block                         | `web/static/index.html`, `app.js`; `server.py:/observer/*`     |
| Hover to see token IDs, click to detokenize                  | `app.js:decodeAndShow`; `server.py:/observer/decode`           |
| Color-coded block states: free / evictable / owned / shared / just-evicted | `app.js:classFor`, `style.css:.block.*`              |
| Demo buttons (10 / 100 / 1000 / shared-prefix pair)          | `index.html`, `app.js:DEMOS`                                   |
| Saved demo prompts in `prompts/{N}.txt` + `{N}.path.txt`     | `scripts/build_prompts.py` (compact `i: target`, no cycles, path = N/2) |
| Reasoning task: model traces a path through the rules graph  | `prompts/{N}.txt` task footer; graded by `scripts/grade_prompts.py` |
| Streaming via SSE                                            | `server.py:_stream_chat`, `app.js:streamChat`                  |
| Concurrent requests batched in one forward pass              | `engine/scheduler.py:_admit_prefill_batch`; verified at 4-way   |
| No artificial concurrency cap                                | `EngineConfig.max_num_seqs=1024`                               |
| Deferred (non-immediate) cache eviction                      | `engine/block_manager.py:_deref` / `_allocate_block` LRU        |
| YaRN extended RoPE → 131k positions                          | `layers/rotary.py`; `models/qwen3.py` enables by default       |
| Continuous-batching scheduler with preemption                | `engine/scheduler.py`                                          |
| OpenAI-compatible HTTP, no auth by default                   | `server.py`; `serve.sh` unsets `VLLM_API_KEY`                  |
| Port 1433, kill-on-port + nohup                              | `serve.sh` (patterned after `tonghuikang/nemotron/serve.sh`)   |
| Benchmark: concurrency × prefix table                        | `scripts/bench.py` (prompt sent as token IDs; honours `min_tokens` / `ignore_eos`) |
| Comparison vs `vllm.service` (Qwen3-0.6B AND gpt-oss-20b)    | `SPEED.md`                                                     |
| TTFT + tok/s in UI                                           | `app.js:streamChat` per-run card                               |
| `.txt` artifacts gitignored except saved prompts             | `.gitignore`                                                   |

## Open work

Everything below is in flight, not deferred:

- **gpt-oss-20b** — MoE + alternating sliding-window + MXFP4 dequant. The
  per-arch model file is the only thing left; the engine, scheduler, KV
  cache, and sliding-window attention path are already plumbed.
- **Fused attention kernel** — flash-attn varlen / Triton paged attention.
  Biggest remaining throughput win; current per-seq SDPA loop is ~10-100×
  slower than vLLM's fused PagedAttention on long-prefix decode.
- **`torch.compile` / CUDA graph capture** — needs static block-table
  shapes; the cached `_block_table_tensor` on `Sequence` is the prereq
  refactor that's already landed.
