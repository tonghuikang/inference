# inference

A stripped-down OpenAI-compatible LLM server for the GB10 (sm_121,
aarch64, CUDA 13). Modelled on
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). Default model
is Qwen3-0.6B; gpt-oss-20b support is being built (see `plan.md`).

The whole point: a live HTML observer at `/observer/` that renders every
physical KV block, color-coded by status, with hover/click to see the
decoded tokens — so you can *see* prefill, prefix sharing, deferred
eviction, and continuous batching happening in real time.

**Status / open work: see `plan.md` and `SPEED.md`.** Throughput is not
yet matching vLLM and gpt-oss-20b is not yet validated with real
weights — the plan documents both.

## Quick start

```sh
./serve.sh                              # http://localhost:1433
open http://localhost:1433/observer/    # KV grid + demo buttons
PYTHONPATH=src uv run --frozen python scripts/bench.py
PYTHONPATH=src uv run --frozen pytest tests/   # 16 should pass
```

## Repo layout

```
src/inference/
  config.py              # ModelConfig, EngineConfig, SamplingParams
  server.py              # FastAPI: /v1/{models,completions,chat/completions} + SSE
                         #          /observer/{snapshot,ws,decode} + /observer/static
                         #          /prompts/{N}.txt static mount
  engine/
    llm_engine.py        # top loop: scheduler → runner → emit tokens
    scheduler.py         # continuous batching, prefill/decode mix, preempt+recompute
    block_manager.py     # KV pool, refcount, xxhash prefix dedup, LRU evictable
    sequence.py          # Sequence with cached block_table tensor
    model_runner.py      # batched input prep, contextvar metadata, forward call
    context.py           # AttentionContext (set per step, read in PagedAttention)
  layers/
    attention.py         # PagedAttention: per-seq SDPA prefill; Triton kernel decode
    paged_attn_triton.py # fused decode kernel (online softmax, GQA, sliding window)
    rotary.py            # RoPE + YaRN
    rmsnorm.py
    sampler.py           # greedy / temp / top-p / top-k
    moe.py               # top-k router + per-expert SwiGLU (gpt-oss)
    quant_mxfp4.py       # MXFP4 dequant (gpt-oss expert weights)
  models/
    qwen3.py             # Qwen3ForCausalLM — dense, GQA, QK-norm
    gpt_oss.py           # GptOssForCausalLM — built; UNTESTED with real weights
  utils/
    loader.py            # safetensors load with per-arch name remap
    kv_observer.py       # pub/sub event bus
  web/static/
    index.html, app.js, style.css
prompts/
  10.txt, 100.txt, 1000.txt          # line-jumping graphs (`i: target`)
  10.path.txt, 100.path.txt, 1000.path.txt
scripts/
  build_prompts.py     # regenerate prompts/
  download_qwen.py     # snapshot_download Qwen3-0.6B
  smoke_qwen3.py       # one-prompt sanity check (no HTTP)
  bench.py             # concurrency × prefix sweep
  grade_prompts.py     # output vs ground-truth path
serve.sh, plan.md, SPEED.md, tests/
```

## Endpoints

- `GET  /v1/models`
- `POST /v1/completions` — `prompt` accepts string or list[int]; honours
  `min_tokens`, `ignore_eos` (vLLM extensions).
- `POST /v1/chat/completions` — `messages`, chat template via
  `transformers.AutoTokenizer.apply_chat_template`. SSE via
  `stream:true`.
- `GET  /observer/`         — KV grid UI
- `GET  /observer/snapshot` — initial state JSON
- `WS   /observer/ws`       — live KV events
- `GET  /observer/decode?ids=...` — detokenize block contents
- `GET  /prompts/{N}.txt`   — saved demo prompts

## Constraints

- `uv` only — never `pip` or `uv pip` (per `CLAUDE.md`).
- aarch64 + CUDA 13 + sm_121.
- HF cache lives at `/srv/vllm/hf`.
- `.gitignore` ignores `*.txt` except `prompts/*.txt`.
