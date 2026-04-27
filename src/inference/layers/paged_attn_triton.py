"""Triton paged-attention decode kernel.

One forward kernel call replaces the per-seq Python loop + SDPA from
attention.py for the decode case. Mirrors `flash_attn_with_kvcache` shape
without depending on flash-attn (which is unbuildable on aarch64 + sm_121
in any reasonable time).

Inputs:
    q:           (B, num_q_heads, head_dim) — one query token per seq
    k_cache, v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    block_tables: (B, max_blocks) int64 — physical block IDs (zero-padded)
    seq_lens:     (B,) int64 — true KV length per seq (caps padded blocks)

Output:
    out: (B, num_q_heads, head_dim) bf16

Algorithm: one program per (batch, query head). Streams through the seq's
blocks and accumulates online softmax (Flash-style). GQA mapping
`kv_h = q_h * num_kv_heads // num_q_heads` baked in. Sliding window
optional via `window` arg.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    out_ptr,
    # strides (in elements)
    stride_q_b, stride_q_h,
    stride_k_blk, stride_k_slot, stride_k_h,
    stride_v_blk, stride_v_slot, stride_v_h,
    stride_bt_b,
    stride_o_b, stride_o_h,
    # scalars
    scale,
    num_kv_heads,
    num_q_heads,
    window,
    # constants
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + b)
    kv_h = (h * num_kv_heads) // num_q_heads

    # Load Q vector for (b, h).
    d_offsets = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + b * stride_q_b + h * stride_q_h + d_offsets).to(tl.float32)

    # Online-softmax state.
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Sliding window cutoff: positions >= seq_len - window are kept; else masked.
    win_start = tl.where(window > 0, seq_len - window, tl.zeros_like(seq_len))

    # Number of blocks this seq actually uses.
    n_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    s_offsets = tl.arange(0, BLOCK_SIZE)
    # Dynamic range — at runtime, not statically unrolled. Static unroll
    # blows up the PTX for long sequences (L=4096 → 128 inlined copies).
    for blk_idx in range(0, n_blocks):
        block_id = tl.load(block_tables_ptr + b * stride_bt_b + blk_idx)
        block_start = blk_idx * BLOCK_SIZE
        pos = block_start + s_offsets
        valid = (pos < seq_len) & (pos >= win_start)

        k_off = (
            block_id * stride_k_blk
            + s_offsets[:, None] * stride_k_slot
            + kv_h * stride_k_h
            + d_offsets[None, :]
        )
        v_off = (
            block_id * stride_v_blk
            + s_offsets[:, None] * stride_v_slot
            + kv_h * stride_v_h
            + d_offsets[None, :]
        )
        k = tl.load(k_cache_ptr + k_off, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + v_off, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))

        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + b * stride_o_b + h * stride_o_h + d_offsets, out.to(tl.bfloat16))


def paged_decode_attn(
    q: torch.Tensor,  # (B, num_q_heads, head_dim) bf16
    k_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,  # (B, max_blocks) int64
    seq_lens: torch.Tensor,  # (B,) int64
    scale: float,
    window: int = 0,
) -> torch.Tensor:
    """One Triton launch covers all (B, num_q_heads) outputs."""
    B, num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks = block_tables.shape[1]
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == torch.bfloat16
    assert v_cache.dtype == torch.bfloat16

    out = torch.empty_like(q)
    grid = (B, num_q_heads)
    _paged_decode_attn_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        out,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        scale,
        num_kv_heads,
        num_q_heads,
        window,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=max_blocks,
    )
    return out
