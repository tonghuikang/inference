"""Paged attention.

KV cache layout per attention layer:
    K, V: (num_blocks, block_size, num_kv_heads, head_dim)

Per call we do:
    1. SCATTER. Write each input token's K and V into the cache at
       slot_mapping[i] = block_id * block_size + slot.
    2. GATHER + SDPA. Read each sequence's K/V via its block_table and run
       torch.nn.functional.scaled_dot_product_attention with `enable_gqa=True`.

For decode (q_len==1 for all seqs) we batch into a single padded SDPA call
when B >= 8 AND B*max_seq_len <= 4096 (the per-step block-table padding +
attn-mask are precomputed once on `AttentionContext` so each layer doesn't
redo them).

Sliding window: gpt-oss alternates full and windowed attention. Both paths
mask via `attn_mask` / `seq[-window:]`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from inference.engine.context import get_context


@torch.no_grad()
def write_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    flat_k = k_cache.view(-1, *k_cache.shape[2:])
    flat_v = v_cache.view(-1, *v_cache.shape[2:])
    flat_k.index_copy_(0, slot_mapping, k)
    flat_v.index_copy_(0, slot_mapping, v)


class PagedAttention(nn.Module):
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
        self._enable_gqa = self.gqa_groups > 1

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        ctx = get_context()
        write_kv_cache(k, v, k_cache, v_cache, ctx.slot_mapping)
        if not ctx.is_prefill and ctx.padded_block_tables is not None:
            return self._decode_batched(q, k_cache, v_cache, ctx)
        return self._per_seq(q, k_cache, v_cache, ctx)

    # ----------------------------------------------------------- per-seq path
    def _per_seq(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        offset = 0
        for block_ids, seq_len, q_len in zip(
            ctx.block_tables,
            ctx.seq_lens.tolist(),
            ctx.query_lens.tolist(),
            strict=True,
        ):
            q_chunk = q[offset : offset + q_len]
            offset += q_len
            k_seq, v_seq = self._gather(k_cache, v_cache, block_ids, seq_len)

            q_t = q_chunk.transpose(0, 1).unsqueeze(0)
            k_t = k_seq.transpose(0, 1).unsqueeze(0)
            v_t = v_seq.transpose(0, 1).unsqueeze(0)

            attn = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                is_causal=ctx.is_prefill and q_len > 1,
                scale=self.scale,
                enable_gqa=self._enable_gqa,
            ).squeeze(0)

            out[offset - q_len : offset] = attn.transpose(0, 1)
        return out

    # ----------------------------------------------------------- batched decode
    def _decode_batched(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        padded = ctx.padded_block_tables  # (B, max_blocks)
        B, max_blocks = padded.shape
        block_size = k_cache.shape[1]
        kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]

        K_full = k_cache[padded].reshape(B, max_blocks * block_size, kv_heads, head_dim)
        V_full = v_cache[padded].reshape(B, max_blocks * block_size, kv_heads, head_dim)

        attn_mask = ctx.decode_attn_mask
        if self.window is not None:
            kv_len = max_blocks * block_size
            positions = torch.arange(kv_len, device=q.device).unsqueeze(0)
            sl = ctx.seq_lens.unsqueeze(1)
            window_mask = positions >= sl - self.window
            attn_mask = attn_mask & window_mask.unsqueeze(1).unsqueeze(1)

        q_b = q.view(B, 1, self.num_q_heads, head_dim).transpose(1, 2)
        k_b = K_full.transpose(1, 2)
        v_b = V_full.transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
            enable_gqa=self._enable_gqa,
        )
        return attn.squeeze(2)

    def _gather(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_ids: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = k_cache[block_ids].reshape(-1, *k_cache.shape[2:])[:seq_len]
        v = v_cache[block_ids].reshape(-1, *v_cache.shape[2:])[:seq_len]
        if self.window is not None and seq_len > self.window:
            k = k[-self.window :]
            v = v[-self.window :]
        return k, v
