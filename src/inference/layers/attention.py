"""Paged attention.

KV cache layout per attention layer:
    K, V: (num_blocks, block_size, num_kv_heads, head_dim)

Per call we do:
    1. SCATTER. Write each input token's K and V into the cache at
       slot_mapping[i] = block_id * block_size + slot.
    2. GATHER + SDPA. For each sequence in the batch, gather K/V via its
       block_table and run torch.nn.functional.scaled_dot_product_attention
       with `enable_gqa=True` so we don't materialise the kv-head expansion.

For decode (q_len==1 for all seqs) we batch into a single padded SDPA call
when B >= 8 (where per-seq launch overhead starts to dominate); below that
the per-seq loop is faster because padding waste outweighs launch cost.
Threshold tuned on the L=4096 sweep.

Sliding window: gpt-oss alternates full and windowed attention. Both paths
mask via `attn_mask` / `seq[-window:]`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# Use the batched padded-SDPA decode path only when total KV traffic stays
# below this many tokens (B * max_kv_len). Above it the per-seq SDPA path
# wins because the padded approach materialises a big (B, kv_len, ...)
# tensor and the per-seq path doesn't. Tuned empirically:
#   - L=1 N=64 (B*kv_len ≈ 64): batched is ~3× faster.
#   - L=4096 N=16 (B*kv_len ≈ 65k): batched is ~3× SLOWER.
# So 4096 is a safe threshold.
_BATCH_KV_THRESHOLD = 4096
_MIN_BATCH_SIZE = 8


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
    slot_mapping: (num_tokens,) int64; absolute slot = block_id*block_size + slot."""
    flat_k = k_cache.view(-1, *k_cache.shape[2:])
    flat_v = v_cache.view(-1, *v_cache.shape[2:])
    flat_k.index_copy_(0, slot_mapping, k)
    flat_v.index_copy_(0, slot_mapping, v)


def _pad_block_tables(
    block_tables: list[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Right-pad to (B, max_blocks). Padding uses block_id=0; we mask its
    contribution in attn_mask."""
    max_blocks = max(bt.shape[0] for bt in block_tables)
    B = len(block_tables)
    padded = torch.zeros(B, max_blocks, dtype=torch.long, device=device)
    for i, bt in enumerate(block_tables):
        padded[i, : bt.shape[0]] = bt
    return padded


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
        slot_mapping: torch.Tensor,
        block_tables: list[torch.Tensor],
        seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        write_kv_cache(k, v, k_cache, v_cache, slot_mapping)
        B = len(block_tables)
        if not is_prefill and B >= _MIN_BATCH_SIZE:
            max_seq_len = int(seq_lens.max().item())
            if B * max_seq_len <= _BATCH_KV_THRESHOLD:
                return self._decode_batched(q, k_cache, v_cache, block_tables, seq_lens)
        return self._per_seq(
            q, k_cache, v_cache, block_tables, seq_lens, query_lens, is_prefill
        )

    # ----------------------------------------------------------- per-seq path
    def _per_seq(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: list[torch.Tensor],
        seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        offset = 0
        for block_ids, seq_len, q_len in zip(
            block_tables, seq_lens.tolist(), query_lens.tolist(), strict=True
        ):
            q_chunk = q[offset : offset + q_len]
            offset += q_len
            k_seq, v_seq = self._gather(k_cache, v_cache, block_ids, seq_len)

            q_t = q_chunk.transpose(0, 1).unsqueeze(0)  # (1, q_heads, q_len, head_dim)
            k_t = k_seq.transpose(0, 1).unsqueeze(0)  # (1, kv_heads, kv_len, head_dim)
            v_t = v_seq.transpose(0, 1).unsqueeze(0)

            attn = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                is_causal=is_prefill and q_len > 1,
                scale=self.scale,
                enable_gqa=self._enable_gqa,
            ).squeeze(0)  # (heads, q_len, head_dim)

            out[offset - q_len : offset] = attn.transpose(0, 1)
        return out

    # ----------------------------------------------------------- batched decode
    def _decode_batched(
        self,
        q: torch.Tensor,  # (B, num_q_heads, head_dim) — q_len=1 per seq.
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: list[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        device = q.device
        B = len(block_tables)
        block_size = k_cache.shape[1]
        kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]

        padded = _pad_block_tables(block_tables, device)
        max_blocks = padded.shape[1]
        kv_len = max_blocks * block_size

        K_full = k_cache[padded].reshape(B, kv_len, kv_heads, head_dim)
        V_full = v_cache[padded].reshape(B, kv_len, kv_heads, head_dim)

        positions = torch.arange(kv_len, device=device).unsqueeze(0)  # (1, kv_len)
        sl = seq_lens.unsqueeze(1)
        mask = positions < sl
        if self.window is not None:
            mask = mask & (positions >= sl - self.window)
        attn_mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, kv_len)

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
        )  # (B, num_q_heads, 1, head_dim)
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
