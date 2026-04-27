"""End-to-end smoke test that exercises engine + paged attention + scheduler
without needing real model weights.

We build a tiny Qwen3 with random weights, run a 1-token prefill + a couple
of decodes, and check we get sensible token IDs and the block manager
allocated/freed predictably."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.config import EngineConfig, SamplingParams  # noqa: E402
from inference.engine.block_manager import BlockManager  # noqa: E402
from inference.engine.model_runner import ModelRunner  # noqa: E402
from inference.engine.scheduler import Scheduler  # noqa: E402
from inference.engine.sequence import Sequence  # noqa: E402
from inference.layers.sampler import sample  # noqa: E402
from inference.models.qwen3 import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs CUDA")
def test_tiny_qwen3_prefill_decode():
    # Tiny architecture so we can run with random weights in seconds.
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=63,
    )
    eng_cfg = EngineConfig(block_size=4, num_kv_blocks=16, max_num_seqs=4)
    device = torch.device("cuda")
    model = Qwen3ForCausalLM(
        cfg, num_kv_blocks=eng_cfg.num_kv_blocks, block_size=eng_cfg.block_size
    )
    model.to(device).to(torch.bfloat16).eval()

    bm = BlockManager(eng_cfg.num_kv_blocks, eng_cfg.block_size)
    sched = Scheduler(eng_cfg, bm)
    runner = ModelRunner(model, eng_cfg, device)

    seq = Sequence.new(
        prompt=[1, 2, 3, 4, 5], sampling=SamplingParams(temperature=0.0, max_tokens=4)
    )
    sched.add(seq)

    # Prefill.
    step = sched.schedule()
    assert step is not None and step.is_prefill
    logits = runner.run(step.seqs, is_prefill=True)
    assert logits.shape == (1, cfg.vocab_size)
    new_toks = sample(logits, [seq.sampling])
    sched.post_step(step, new_toks, [False])
    assert seq.output_token_ids == new_toks

    # Decode loop a few steps.
    for _ in range(3):
        step = sched.schedule()
        assert step is not None and not step.is_prefill
        logits = runner.run(step.seqs, is_prefill=False)
        new_toks = sample(logits, [seq.sampling])
        sched.post_step(step, new_toks, [False])

    assert len(seq.output_token_ids) == 4


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs CUDA")
def test_paged_attention_matches_unpaged_reference():
    """Sanity check: gather-then-SDPA matches the same SDPA on a plain dense
    K/V tensor. Tiny shapes, fp32, tight tolerance."""
    from inference.engine.context import AttentionContext, reset_context, set_context
    from inference.layers.attention import PagedAttention

    torch.manual_seed(0)
    block_size = 4
    num_blocks = 4
    num_q_heads = 2
    num_kv_heads = 2
    head_dim = 8
    seq_len = 7  # not a multiple of block_size, exercises tail handling.

    device = torch.device("cuda")
    dtype = torch.float32

    q = torch.randn(seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    k_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.tensor(
        [0, 1], dtype=torch.long, device=device
    )  # 2 blocks suffice (8 slots).
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)

    attn = PagedAttention(num_q_heads, num_kv_heads, head_dim).to(device).to(dtype)
    token = set_context(
        AttentionContext(
            is_prefill=True,
            block_tables=[block_table],
            seq_lens=torch.tensor([seq_len], device=device),
            query_lens=torch.tensor([seq_len], device=device),
            slot_mapping=slot_mapping,
        )
    )
    try:
        out = attn(q, k, v, k_cache, v_cache)
    finally:
        reset_context(token)

    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            scale=head_dim**-0.5,
        )
        .squeeze(0)
        .transpose(0, 1)
    )

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
