"""Per-step attention context. Set once per forward in ModelRunner; read
inside every PagedAttention layer so we don't recompute padded block-tables
+ attention masks once per layer.

Pattern lifted from nano-vllm's `nanovllm.utils.context`. Kept in a
contextvar so it works whether the engine runs synchronously on one thread
or in a thread executor.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass

import torch


@dataclass
class AttentionContext:
    is_prefill: bool
    block_tables: list[torch.Tensor]
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    slot_mapping: torch.Tensor
    # Precomputed for the batched-decode fast path. None means per-seq.
    padded_block_tables: torch.Tensor | None = None
    decode_attn_mask: torch.Tensor | None = None


_CTX: contextvars.ContextVar[AttentionContext | None] = contextvars.ContextVar(
    "attn_context", default=None
)


def set_context(ctx: AttentionContext) -> contextvars.Token:
    return _CTX.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _CTX.reset(token)


def get_context() -> AttentionContext:
    ctx = _CTX.get()
    if ctx is None:
        raise RuntimeError("AttentionContext not set; call set_context() before forward")
    return ctx
