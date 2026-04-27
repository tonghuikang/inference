"""OpenAI-compatible HTTP server.

Endpoints:
    GET  /v1/models                     -> list loaded models.
    POST /v1/completions                -> raw prompt completion (+SSE).
    POST /v1/chat/completions           -> chat completion (+SSE).
    GET  /observer/                     -> KV block grid UI.
    GET  /observer/snapshot             -> initial cache state JSON.
    WS   /observer/ws                   -> live KV events.

Auth: bearer token via VLLM_API_KEY env var (matches existing client config).

Engine model: one synchronous LlmEngine per model_id, driven by a single
background thread per engine that pulls requests off an in-memory queue and
fans tokens out to per-request asyncio queues. Requests are served by FastAPI
async handlers that consume those queues.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference.config import EngineConfig, ModelConfig, SamplingParams
from inference.engine.llm_engine import LlmEngine, StepOutput
from inference.utils.kv_observer import KVEvent, get_observer

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Async glue: the engine's `step()` is synchronous; we run it on a thread and
# bridge to per-request asyncio queues.
# -----------------------------------------------------------------------------


@dataclass
class _Pending:
    queue: asyncio.Queue[StepOutput | None]
    seq_id: str = ""
    prompt_tokens: int = 0


@dataclass
class EngineWorker:
    """Wraps an LlmEngine so callers add (prompt, params, asyncio.Queue) and
    receive StepOutputs on the queue. None sentinel signals completion."""

    engine: LlmEngine
    loop: asyncio.AbstractEventLoop
    pending: dict[str, _Pending] = field(default_factory=dict)  # seq_id -> _Pending
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop_body, name="engine-step", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    async def submit(self, prompt: str | list[int], params: SamplingParams) -> _Pending:
        q: asyncio.Queue[StepOutput | None] = asyncio.Queue()
        pend = _Pending(queue=q)

        def _add_on_engine() -> None:
            seq = self.engine.add_request(prompt, params)
            pend.seq_id = seq.seq_id
            pend.prompt_tokens = seq.num_prompt
            self.pending[seq.seq_id] = pend

        # Adding to the scheduler must happen on the engine thread.
        await self.loop.run_in_executor(None, _add_on_engine)
        return pend

    def _loop_body(self) -> None:
        while not self._stop.is_set():
            if not self.engine.has_pending():
                time.sleep(0.005)
                continue
            outputs = self.engine.step()
            for out in outputs:
                pend = self.pending.get(out.seq_id)
                if pend is None:
                    continue
                self.loop.call_soon_threadsafe(pend.queue.put_nowait, out)
                if out.finished:
                    self.loop.call_soon_threadsafe(pend.queue.put_nowait, None)
                    self.pending.pop(out.seq_id, None)


# -----------------------------------------------------------------------------
# Pydantic request/response models (OpenAI shapes, only the fields we honour).
# -----------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[int]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    min_tokens: int = 0  # vLLM extension: minimum decode count before EOS may fire.
    ignore_eos: bool = False  # vLLM extension: never stop on EOS, only on max_tokens.
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "inference"


# -----------------------------------------------------------------------------
# App + auth
# -----------------------------------------------------------------------------


def _make_app(workers: dict[str, EngineWorker], static_dir: Path) -> FastAPI:
    app = FastAPI(title="inference", version="0.1.0")
    api_key = os.environ.get("VLLM_API_KEY")

    def auth(request: Request) -> None:
        if api_key is None:
            return
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or header[7:] != api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # -------- /v1 --------------------------------------------------------
    @app.get("/v1/models")
    async def list_models(_: None = Depends(auth)) -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [ModelCard(id=mid).model_dump() for mid in workers],
            }
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, _: None = Depends(auth)):
        worker = workers.get(req.model)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"unknown model {req.model}")
        params = _params_from(
            req.temperature,
            req.top_p,
            req.top_k,
            req.max_tokens,
            req.stop,
            req.seed,
            min_tokens=req.min_tokens,
            ignore_eos=req.ignore_eos,
        )
        pend = await worker.submit(req.prompt, params)
        if req.stream:
            return StreamingResponse(
                _stream_completion(req.model, pend), media_type="text/event-stream"
            )
        return await _collect_completion(req.model, pend)

    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest, _: None = Depends(auth)):
        worker = workers.get(req.model)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"unknown model {req.model}")
        prompt = worker.engine.tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = _params_from(
            req.temperature, req.top_p, req.top_k, req.max_tokens, req.stop, req.seed
        )
        pend = await worker.submit(prompt, params)
        if req.stream:
            return StreamingResponse(
                _stream_chat(req.model, pend), media_type="text/event-stream"
            )
        return await _collect_chat(req.model, pend)

    # -------- /observer --------------------------------------------------
    # Register dynamic endpoints BEFORE the static mount, otherwise StaticFiles
    # would intercept /observer/snapshot and /observer/ws and return 404.
    @app.get("/observer/decode")
    async def observer_decode(ids: str) -> JSONResponse:
        """Detokenize a comma-separated list of token IDs. Used by the block
        detail pane to show the actual text inside a clicked block."""
        worker = next(iter(workers.values()))
        try:
            id_list = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="bad ids list")
        text = worker.engine.tokenizer.decode(id_list, skip_special_tokens=False)
        return JSONResponse({"text": text, "ids": id_list})

    @app.get("/observer/snapshot")
    async def observer_snapshot() -> JSONResponse:
        # Only one engine's snapshot at a time — UI passes ?model=...; default
        # is "first loaded".
        # (For v1 we only run one engine in practice.)
        worker = next(iter(workers.values()))
        snap = worker.engine.block_mgr.snapshot()
        return JSONResponse(
            {
                "model": worker.engine.model_cfg.model_id,
                "block_size": worker.engine.engine_cfg.block_size,
                "num_blocks": worker.engine.engine_cfg.num_kv_blocks,
                "blocks": snap,
            }
        )

    @app.websocket("/observer/ws")
    async def observer_ws(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[KVEvent] = asyncio.Queue(maxsize=1024)

        def push(ev: KVEvent) -> None:
            loop.call_soon_threadsafe(_safe_put, queue, ev)

        unsub = get_observer().subscribe(push)
        try:
            while True:
                ev = await queue.get()
                await ws.send_text(ev.to_json())
        except WebSocketDisconnect:
            pass
        finally:
            unsub()

    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/observer", StaticFiles(directory=str(static_dir), html=True), name="observer"
    )

    # Saved prompt files (the line-jumping demos) — repo-root /prompts.
    # parents[2] = project root (src/inference/server.py -> src -> root).
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    if prompts_dir.exists():
        app.mount("/prompts", StaticFiles(directory=str(prompts_dir)), name="prompts")

    return app


def _safe_put(queue: asyncio.Queue[KVEvent], item: KVEvent) -> None:
    if queue.full():
        # Drop oldest to keep up.
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(item)


def _params_from(
    temp: float,
    top_p: float,
    top_k: int,
    max_tok: int,
    stop: list[str] | str | None,
    seed: int | None,
    *,
    min_tokens: int = 0,
    ignore_eos: bool = False,
) -> SamplingParams:
    stop_list = [stop] if isinstance(stop, str) else (stop or [])
    return SamplingParams(
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tok,
        min_tokens=min_tokens,
        ignore_eos=ignore_eos,
        stop=stop_list,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# Streaming helpers (OpenAI SSE shape)
# -----------------------------------------------------------------------------


async def _drain(pend: _Pending) -> AsyncIterator[StepOutput]:
    while True:
        out = await pend.queue.get()
        if out is None:
            return
        yield out


async def _stream_completion(model: str, pend: _Pending) -> AsyncIterator[bytes]:
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    async for out in _drain(pend):
        chunk = {
            "id": cmpl_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": out.text_delta,
                    "finish_reason": out.finish_reason if out.finished else None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _stream_chat(model: str, pend: _Pending) -> AsyncIterator[bytes]:
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    async for out in _drain(pend):
        chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": out.text_delta} if out.text_delta else {},
                    "finish_reason": out.finish_reason if out.finished else None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _collect_completion(model: str, pend: _Pending) -> JSONResponse:
    text = ""
    finish = None
    n_out = 0
    async for out in _drain(pend):
        text += out.text_delta
        n_out += 1
        if out.finished:
            finish = out.finish_reason
    return JSONResponse(
        {
            "id": f"cmpl-{uuid.uuid4().hex[:16]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": pend.prompt_tokens,
                "completion_tokens": n_out,
                "total_tokens": pend.prompt_tokens + n_out,
            },
        }
    )


async def _collect_chat(model: str, pend: _Pending) -> JSONResponse:
    text = ""
    finish = None
    n_out = 0
    async for out in _drain(pend):
        text += out.text_delta
        n_out += 1
        if out.finished:
            finish = out.finish_reason
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": pend.prompt_tokens,
                "completion_tokens": n_out,
                "total_tokens": pend.prompt_tokens + n_out,
            },
        }
    )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def _resolve_snapshot(repo_id: str, hf_home: Path) -> Path:
    """Find the latest snapshot dir for an HF repo in /srv/vllm/hf/hub layout."""
    folder = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    snaps = sorted(folder.iterdir())
    if not snaps:
        raise FileNotFoundError(f"no snapshot under {folder}")
    return snaps[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Repo id; resolved against HF_HOME/hub.",
    )
    parser.add_argument("--num-kv-blocks", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--kv-log", default=None, help="Optional path to append KV events."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    hf_home = Path(os.environ.get("HF_HOME", "/srv/vllm/hf"))
    snapshot = _resolve_snapshot(args.model, hf_home)
    log.info("resolved %s -> %s", args.model, snapshot)

    if args.kv_log:
        get_observer().attach_logfile(Path(args.kv_log))

    engine_cfg = EngineConfig(
        block_size=args.block_size,
        num_kv_blocks=args.num_kv_blocks,
        max_num_seqs=args.max_num_seqs,
    )
    model_cfg = ModelConfig(model_id=args.model, path=snapshot)
    engine = LlmEngine(model_cfg, engine_cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    worker = EngineWorker(engine=engine, loop=loop)
    worker.start()

    static_dir = Path(__file__).parent / "web" / "static"
    app = _make_app({args.model: worker}, static_dir)

    import uvicorn

    config = uvicorn.Config(app, host=args.host, port=args.port, loop="asyncio")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
