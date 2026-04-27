"""Top-level engine: takes requests, drives scheduler+runner, emits tokens.

One LlmEngine per loaded model. The HTTP server keeps a dict of engines keyed
by model_id and routes requests by the OpenAI `model` field.

This is a synchronous step loop. The server wraps it in a background asyncio
task that yields output tokens to per-request queues — see server.py.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from inference.config import EngineConfig, ModelConfig, SamplingParams
from inference.layers.sampler import sample
from inference.models import get as get_model_loader

from .block_manager import BlockManager
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence

log = logging.getLogger(__name__)


@dataclass
class StepOutput:
    seq_id: str
    token_id: int
    text_delta: str
    finished: bool
    finish_reason: str | None


class LlmEngine:
    def __init__(self, model_cfg: ModelConfig, engine_cfg: EngineConfig) -> None:
        self.model_cfg = model_cfg
        self.engine_cfg = engine_cfg
        self.device = torch.device("cuda")

        log.info("loading tokenizer for %s", model_cfg.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_cfg.path))
        self.eos_token_id = self.tokenizer.eos_token_id

        log.info("loading weights for %s", model_cfg.model_id)
        loader = get_model_loader(_arch_from_path(model_cfg.path))
        self.model = loader(
            model_cfg.path,
            num_kv_blocks=engine_cfg.num_kv_blocks,
            block_size=engine_cfg.block_size,
        )
        # Linear layers default to fp32; loader copies bf16 weights into them
        # which casts UP to fp32 storage. Force the whole module to bf16 so
        # forward activations stay bf16 and match the KV cache dtype.
        self.model.to(self.device).to(torch.bfloat16)
        self.model.eval()

        # Compile each decoder layer with mode="default" — reduce-overhead/
        # CUDA-graph capture would require static shapes that we don't have
        # (num_tokens varies per step, block_tables are dynamic). Default
        # mode still gets us inductor-fused linears + RMSNorm + RoPE.
        if not engine_cfg.enforce_eager:
            for layer in self.model.layers:
                layer.compile(mode="default", dynamic=True)

        self.block_mgr = BlockManager(
            num_blocks=engine_cfg.num_kv_blocks,
            block_size=engine_cfg.block_size,
            layer_group="full",
        )
        self.scheduler = Scheduler(engine_cfg, self.block_mgr)
        self.runner = ModelRunner(self.model, engine_cfg, self.device)
        self._streamed_offsets: dict[str, int] = {}  # seq_id -> chars already emitted.

    # -- public API -----------------------------------------------------------
    def add_request(self, prompt: str | list[int], params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            token_ids = list(prompt)  # already tokenized.
        seq = Sequence.new(token_ids, params)
        self.scheduler.add(seq)
        self._streamed_offsets[seq.seq_id] = 0
        return seq

    def has_pending(self) -> bool:
        return self.scheduler.num_pending() > 0

    def step(self) -> list[StepOutput]:
        """Run one scheduler step. Returns one StepOutput per running seq
        (or per just-admitted prefill seq); finished seqs included."""
        step = self.scheduler.schedule()
        if step is None:
            return []

        logits = self.runner.run(step.seqs, is_prefill=step.is_prefill)
        params = [s.sampling for s in step.seqs]
        new_tokens = sample(logits, params)

        finish_flags = [
            self._would_finish(s, t) for s, t in zip(step.seqs, new_tokens, strict=True)
        ]
        finished = self.scheduler.post_step(step, new_tokens, finish_flags)

        outputs = []
        for seq, tok in zip(step.seqs, new_tokens, strict=True):
            text_delta = self._render_delta(seq)
            outputs.append(
                StepOutput(
                    seq_id=seq.seq_id,
                    token_id=tok,
                    text_delta=text_delta,
                    finished=seq.is_finished(),
                    finish_reason=seq.finish_reason,
                )
            )

        # Free finished seqs' KV blocks.
        for seq in finished:
            self.block_mgr.free(seq.seq_id, seq.block_table)
            self._streamed_offsets.pop(seq.seq_id, None)

        return outputs

    def stream(self, prompt: str, params: SamplingParams) -> Iterator[StepOutput]:
        """Convenience for direct (non-HTTP) usage. Drives the engine until
        the given request finishes, yielding only its outputs."""
        seq = self.add_request(prompt, params)
        while True:
            for out in self.step():
                if out.seq_id == seq.seq_id:
                    yield out
                    if out.finished:
                        return

    # -- internals ------------------------------------------------------------
    def _would_finish(self, seq: Sequence, new_tok: int) -> bool:
        n_out_after = len(seq.output_token_ids) + 1
        if n_out_after >= seq.sampling.max_tokens:
            seq.finish_reason = "length"
            return True
        if seq.sampling.ignore_eos:
            return False
        if new_tok == self.eos_token_id and n_out_after >= seq.sampling.min_tokens:
            seq.finish_reason = "eos"
            return True
        return False

    def _render_delta(self, seq: Sequence) -> str:
        """Decode all output tokens and return only the chars produced since
        the last call. Tokenizers don't decode well token-by-token (multibyte
        UTF-8, BPE merges), so we always re-decode the full output and diff
        on character offset."""
        full = self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        prev = self._streamed_offsets.get(seq.seq_id, 0)
        delta = full[prev:]
        self._streamed_offsets[seq.seq_id] = len(full)
        return delta


def _arch_from_path(model_dir: Path) -> str:
    import json

    with (model_dir / "config.json").open() as f:
        return json.load(f)["architectures"][0]
