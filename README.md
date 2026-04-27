# inference

A small, hackable, OpenAI-compatible LLM server modelled on
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). Two architectures —
**Qwen3-0.6B** (dense) and **gpt-oss-20b** (MoE + sliding-window attention) —
share a single Python engine where the **KV cache allocator, block manager,
scheduler, and paged attention** are written to be read, instrumented, and
modified.

The original goal was to replace an opaque `vllm serve` docker container with
something where you can *see* exactly how the KV cache is allocated. To that
end the server ships with a live HTML observer at `/observer/` that renders
every physical KV block in the pool, color-coded by status, with hover-to-see
the decoded tokens stored inside each block.

## Repo layout

```
src/inference/
  __init__.py
  config.py              # ModelConfig, EngineConfig, SamplingParams
  server.py              # FastAPI: /v1/models, /v1/chat/completions, /v1/completions (+SSE)
                         #         + /observer/ static mount + /observer/ws + /observer/snapshot
  engine/
    llm_engine.py        # top-level loop: pull requests -> schedule -> run -> emit tokens
    scheduler.py         # continuous batching: admission, prefill/decode mix, preemption
    block_manager.py     # KV block pool, ref counts, xxhash-based prefix dedup
    sequence.py          # Sequence, SequenceGroup, status, block_table
    model_runner.py      # batched input prep, forward call, KV cache write-back
  layers/
    attention.py         # PagedAttention wrapper (flash-attn varlen or Triton) + sliding-window variant
    rotary.py            # RoPE
    rmsnorm.py
    sampler.py           # temperature / top-p / top-k / greedy
    moe.py               # top-k gate + per-expert grouped GEMM (gpt_oss only)
    quant_mxfp4.py       # MXFP4 dequant for gpt-oss weight tensors
  models/
    __init__.py          # registry: HF config.architectures[0] -> model class
    qwen3.py             # Qwen3ForCausalLM (dense)
    gpt_oss.py           # GptOssForCausalLM (MoE + alternating sliding window)
  utils/
    loader.py            # safetensors -> module load with name remapping per arch
    kv_observer.py       # pub/sub event bus: alloc/free/hit/append. Sinks: log file + WebSocket.
  web/static/
    index.html           # KV block grid UI
    app.js               # WS client, grid renderer, block-detail pane
    style.css
scripts/
  run_server.py          # entrypoint: uv run python -m inference.server
  download_qwen.py       # huggingface_hub.snapshot_download -> /srv/vllm/hf
tests/
  test_block_manager.py  # alloc/free/ref-count/hash dedup
  test_paged_attention.py
  test_qwen3_smoke.py
  test_gpt_oss_smoke.py
  test_observer_ui.py    # puppeteer-driven HTML smoke (per CLAUDE.md)
```

## Design

Architecture-specific code lives in `models/<arch>.py`; everything else is
shared. Keeping the engine in pure, readable Python is the point — performance
is secondary to being able to step through the KV cache lifecycle.

### 1. Block manager — the observability anchor

Mirrors nano-vllm's allocator: fixed-size blocks (default 16 tokens),
ref-counted, free queue, `hash(prefix tokens) -> block_id` map for prefix
reuse. Every alloc / free / hit / append emits an event through
`kv_observer.py` so a `tail -f` of allocator events can run alongside
inference.

### 2. Two block pools per model when needed

For gpt-oss the scheduler keeps a separate pool (or stride-aware view) for
sliding-window layers so windowed evictions don't fight full-attention reuse.
`BlockManager(num_blocks, block_size, window=None | int)`; `LlmEngine`
instantiates one per attention-pattern group.

### 3. Paged attention

Wraps `flash_attn.flash_attn_varlen_func` with `block_table` (flash-attn ≥ 2.5)
when available. Triton kernel fallback if flash-attn isn't buildable on
GB10/aarch64. Models call the wrapper and stay clean of cache-layout details.

### 4. Continuous batching scheduler

Prefill-priority admission until the running batch's KV demand exceeds free
blocks, then decode-only steps until a slot frees. Preempt + recompute when a
sequence runs out of room mid-decode (no swap-to-CPU in v1).

### 5. Two model files, shared layers

`qwen3.py` and `gpt_oss.py` both use `attention.PagedAttention`, `rotary.RoPE`,
`rmsnorm.RMSNorm`, and `sampler`. `gpt_oss.py` additionally uses `layers/moe.py`
and `layers/quant_mxfp4.py`. Each model file is self-contained and readable
top-to-bottom — no inheritance gymnastics.

### 6. Weight loading

`utils/loader.py` reads HF safetensors index, applies a per-arch name-remap
dict (defined in the model file), and copies tensors into the module. MXFP4
tensors for gpt-oss stay packed and dequantize on the fly inside
`quant_mxfp4.linear`.

### 7. OpenAI-compatible HTTP surface

- `GET  /v1/models` — lists the two loaded models.
- `POST /v1/chat/completions` — `messages`, `temperature`, `top_p`, `max_tokens`,
  `stream`. Applies the model's chat template via
  `transformers.AutoTokenizer.apply_chat_template`.
- `POST /v1/completions` — raw prompt path.
- Bearer-token auth via `VLLM_API_KEY` env var (matches existing client config).
- Streaming = SSE in OpenAI's chunk format.

### 8. Both models loaded at startup

`LlmEngine` holds `model_id -> ModelInstance(model, tokenizer, block_manager,
scheduler)`. Requests are routed by the `model` field. Memory budget:
gpt-oss-20b ≈ 13 GB (MXFP4) + KV, Qwen3-0.6B ≈ 1.2 GB BF16 + KV — comfortable
on the GB10's 120 GB unified memory. `gpu_memory_utilization` config caps
total.

### 9. HTML KV-block observer

`server.py` mounts `src/inference/web/static/` at `/observer/` and exposes a
read-only WebSocket at `/observer/ws`. The page renders one cell per physical
KV block, color-coded:

- grey = free
- blue = allocated to one sequence
- green = shared via prefix-hash dedup (refcount > 1)
- red flash = just freed (decays over ~500 ms)

Hovering shows `block_id`, `refcount`, `hash`, owning sequence IDs, and the
decoded token text in that block (the block manager keeps `token_ids:
list[int]` per block; the server runs them through the tokenizer for display).
Clicking pins the detail pane.

The UI subscribes to the `kv_observer` event bus over WebSocket; every
`alloc/free/hit/append` emits `{event, block_id, seq_id, hash, tokens}` and
the page diff-applies it to the grid. `GET /observer/snapshot` returns the
initial state on page load.

### Out of scope for v1

- Tensor parallelism / multi-GPU
- CUDA graph capture
- Tool / function-calling parsing on the server side
- Prometheus metrics (replaced by `kv_observer` log tail)
- Swap-to-CPU on preemption (recompute only)
- Multimodal inputs

## Dependencies

Managed via `uv` (see `CLAUDE.md` for project rules — no `pip`).

- `torch` (CUDA 13 / sm_120 Blackwell)
- `transformers` (tokenizers + config classes only; no modeling code)
- `safetensors`
- `xxhash` (block content hashing)
- `flash-attn` (varlen + block_table; aarch64 wheel may need source build)
- `triton` (fallback paged-attention kernel)
- `fastapi`, `uvicorn`, `sse-starlette`
- `huggingface_hub`

## Migration / cutover

The previous setup ran `vllm serve openai/gpt-oss-20b` inside docker via the
systemd unit at `/etc/systemd/system/vllm.service`.

1. Stop the existing service:
   ```
   sudo systemctl stop vllm
   docker ps   # confirm container is gone
   ```
2. Download Qwen3-0.6B into the shared HF cache:
   ```
   HF_HOME=/srv/vllm/hf uv run python scripts/download_qwen.py
   ```
3. Start the new server:
   ```
   uv run python -m inference.server --port 8000
   ```
4. Existing clients keep working — same port, same `VLLM_API_KEY`, same OpenAI
   surface. Model IDs are now `openai/gpt-oss-20b` and `Qwen/Qwen3-0.6B`.

The systemd unit is left on disk (disabled) so `systemctl start vllm` reverts
to the old docker server instantly if the new one has problems.

## Verification

1. **Unit tests** — `uv run --frozen pytest tests/`:
   - `test_block_manager.py` — alloc / free / refcount overflow / prefix hash
     dedup / free queue ordering.
   - `test_paged_attention.py` — numerical match vs. an unpaged reference
     (small shapes, fp32) within 1e-3.
2. **Smoke** — both models load and emit a deterministic completion at
   `temperature=0`:
   ```
   curl -s http://127.0.0.1:8000/v1/chat/completions \
     -H "Authorization: Bearer $VLLM_API_KEY" \
     -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}]}'
   ```
   Same for `openai/gpt-oss-20b`.
3. **KV observability — log tail.** Run a 3-request workload with overlapping
   prompt prefixes; `kv_observer.log` should show `ALLOC`, `HIT` (prefix
   reuse), and `FREE` lines with consistent `block_id` and `refcount`.
4. **KV observability — HTML.** Smoke-tested via puppeteer MCP per
   `CLAUDE.md`: navigate to `http://127.0.0.1:8000/observer/`, fire a long
   prompt, screenshot the grid filling with blue cells; fire a second request
   sharing a prefix and confirm those cells flip green; let the first finish
   and confirm red-flash → grey decay.
5. **Concurrency** — 8 parallel chat requests; scheduler interleaves
   prefill / decode without OOM.
6. **Streaming** — `stream:true` produces incremental SSE chunks matching
   OpenAI format (verified with the `openai` Python SDK pointed at our
   endpoint).
7. **Lint / type** — `uv run --frozen ruff check src/ tests/` and
   `uv run --frozen mypy src/` clean.
