"""Numerical equivalence: Triton paged-decode kernel vs gather + SDPA."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.layers.paged_attn_triton import paged_decode_attn  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs CUDA")
def test_triton_decode_matches_sdpa_no_gqa():
    torch.manual_seed(0)
    B = 4
    num_q_heads = 4
    num_kv_heads = 4
    head_dim = 64
    block_size = 16
    seq_lens_list = [17, 64, 31, 16]
    max_blocks = max((sl + block_size - 1) // block_size for sl in seq_lens_list)
    num_blocks = 32
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(B, num_q_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Random block_tables (each seq picks distinct blocks).
    block_tables = torch.zeros(B, max_blocks, dtype=torch.long, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long, device=device)
    for i, sl in enumerate(seq_lens_list):
        n = (sl + block_size - 1) // block_size
        block_tables[i, :n] = torch.tensor(
            [i * 4 + j for j in range(n)], device=device, dtype=torch.long
        )

    scale = head_dim**-0.5
    out_triton = paged_decode_attn(q, k_cache, v_cache, block_tables, seq_lens, scale, 0)

    # Reference: gather + SDPA per seq.
    out_ref = torch.empty_like(q)
    for i, sl in enumerate(seq_lens_list):
        n = (sl + block_size - 1) // block_size
        ks = k_cache[block_tables[i, :n]].reshape(-1, num_kv_heads, head_dim)[:sl]
        vs = v_cache[block_tables[i, :n]].reshape(-1, num_kv_heads, head_dim)[:sl]
        # (1, heads, q_len=1, head_dim) attended to (1, heads, sl, head_dim)
        q_ = q[i].unsqueeze(0).transpose(0, 1).unsqueeze(0)  # (1, heads, 1, dim)
        k_ = ks.transpose(0, 1).unsqueeze(0)  # (1, heads, sl, dim)
        v_ = vs.transpose(0, 1).unsqueeze(0)
        attn = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, scale=scale)
        out_ref[i] = attn.squeeze(0).squeeze(1)

    torch.testing.assert_close(out_triton, out_ref, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs CUDA")
def test_triton_decode_matches_sdpa_gqa():
    """GQA: 8 q_heads, 2 kv_heads (4× ratio)."""
    torch.manual_seed(1)
    B = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    seq_lens_list = [17, 30]
    max_blocks = 2
    num_blocks = 8
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(B, num_q_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    block_tables = torch.tensor(
        [[0, 1], [2, 3]], dtype=torch.long, device=device
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long, device=device)
    scale = head_dim**-0.5

    out_triton = paged_decode_attn(q, k_cache, v_cache, block_tables, seq_lens, scale, 0)

    # Reference using SDPA with enable_gqa=True.
    out_ref = torch.empty_like(q)
    for i, sl in enumerate(seq_lens_list):
        n = (sl + block_size - 1) // block_size
        ks = k_cache[block_tables[i, :n]].reshape(-1, num_kv_heads, head_dim)[:sl]
        vs = v_cache[block_tables[i, :n]].reshape(-1, num_kv_heads, head_dim)[:sl]
        q_ = q[i].unsqueeze(0).transpose(0, 1).unsqueeze(0)
        k_ = ks.transpose(0, 1).unsqueeze(0)
        v_ = vs.transpose(0, 1).unsqueeze(0)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, scale=scale, enable_gqa=True
        )
        out_ref[i] = attn.squeeze(0).squeeze(1)

    torch.testing.assert_close(out_triton, out_ref, atol=5e-3, rtol=5e-3)
