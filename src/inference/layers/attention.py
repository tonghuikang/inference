"""Paged attention — readability-first implementation.

KV cache layout per attention layer:
    K, V: (num_blocks, block_size, num_kv_heads, head_dim)

The block manager owns *which* block_id is used by which sequence; we just
gather/scatter at known offsets.

Two passes for clarity rather than a fused kernel:

    1. SCATTER. For every input token in the current step, write its K and V
       into the KV cache at slot_mapping[i] = block_id * block_size + slot.

    2. GATHER + SDPA. For each sequence in the batch, gather the K and V
       chunks for the blocks in its block_table, slice down to seq_len, and
       run torch.nn.functional.scaled_dot_product_attention.

This is much slower than flash-attn varlen, but the data flow is obvious in
five lines of Python — which is the whole point of the project.

Sliding window: if `window` is set, gather only the last `window` KV positions.
This is what gpt-oss's alternating layers need.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


@torch.no_grad()
def write_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """k, v: (num_tokens, num_kv_heads, head_dim).
    {k,v}_cache: (num_blocks, block_size, num_kv_heads, head_dim).
    slot_mapping: (num_tokens,) int64; absolute slot = block_id * block_size + slot."""
    flat_k = k_cache.view(
        -1, *k_cache.shape[2:]
    )  # (num_blocks*block_size, kv_heads, head_dim)
    flat_v = v_cache.view(-1, *v_cache.shape[2:])
    flat_k.index_copy_(0, slot_mapping, k)
    flat_v.index_copy_(0, slot_mapping, v)


def _gather_kv_for_seq(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_ids: torch.Tensor,  # (num_blocks_for_seq,)
    seq_len: int,
    window: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the seq's blocks, slice to seq_len. Apply sliding window if set."""
    k = k_cache[block_ids]  # (num_blocks_for_seq, block_size, kv_heads, head_dim)
    v = v_cache[block_ids]
    block_size = k.shape[1]
    k = k.reshape(-1, *k.shape[2:])[:seq_len]  # (seq_len, kv_heads, head_dim)
    v = v.reshape(-1, *v.shape[2:])[:seq_len]
    if window is not None and seq_len > window:
        k = k[-window:]
        v = v[-window:]
    return k, v
    _ = block_size  # silence linter; kept for readability above.


class PagedAttention(nn.Module):
    """Stateless module: just packages the gather + SDPA logic. The KV cache
    tensors live on the model and are passed in per-call so we can reuse this
    layer across full and sliding-window heads."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
        window: int | None = None,
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (head_dim**-0.5)
        self.window = window
        assert num_q_heads % num_kv_heads == 0, (
            "GQA: num_q_heads must be divisible by num_kv_heads"
        )
        self.gqa_groups = num_q_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,  # (num_tokens, num_q_heads, head_dim)
        k: torch.Tensor,  # (num_tokens, num_kv_heads, head_dim)
        v: torch.Tensor,
        k_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,  # (num_tokens,)
        block_tables: list[torch.Tensor],  # per-seq block IDs
        seq_lens: torch.Tensor,  # (batch_size,) total length AFTER this step's writes
        query_lens: torch.Tensor,  # (batch_size,) tokens contributed by this step (1 in decode, prompt_len in prefill)
        is_prefill: bool,
    ) -> torch.Tensor:
        write_kv_cache(k, v, k_cache, v_cache, slot_mapping)

        out = torch.empty_like(q)
        offset = 0
        for i, (block_ids, seq_len, q_len) in enumerate(
            zip(block_tables, seq_lens.tolist(), query_lens.tolist(), strict=True)
        ):
            q_chunk = q[offset : offset + q_len]
            offset += q_len

            k_seq, v_seq = _gather_kv_for_seq(
                k_cache, v_cache, block_ids, seq_len, self.window
            )

            # Promote KV heads to Q heads for GQA. SDPA in PyTorch handles GQA via
            # `enable_gqa=True` in newer versions, but we expand manually for clarity.
            if self.gqa_groups > 1:
                k_seq = k_seq.repeat_interleave(self.gqa_groups, dim=1)
                v_seq = v_seq.repeat_interleave(self.gqa_groups, dim=1)

            # SDPA expects (heads, seq, head_dim) per query/key/value.
            q_t = q_chunk.transpose(0, 1)  # (heads, q_len, head_dim)
            k_t = k_seq.transpose(0, 1)  # (heads, kv_len, head_dim)
            v_t = v_seq.transpose(0, 1)

            # Causal masking only for prefill (q_len > 1). For decode, q_len=1 so
            # attention to all KV positions is correct.
            attn = F.scaled_dot_product_attention(
                q_t.unsqueeze(0),
                k_t.unsqueeze(0),
                v_t.unsqueeze(0),
                is_causal=is_prefill and q_len > 1,
                scale=self.scale,
            ).squeeze(0)  # (heads, q_len, head_dim)

            out[offset - q_len : offset] = attn.transpose(0, 1)
            _ = i

        return out
