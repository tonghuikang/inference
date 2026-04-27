"""Qwen3 dense decoder. Designed to be read top-to-bottom.

Notable Qwen3 detail: QK-norm. Q and K are RMSNormed per-head before RoPE.
Other than that it's a vanilla Llama-style block (RMSNorm, GQA attention,
RoPE, SwiGLU MLP, residual).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from inference.layers.attention import PagedAttention
from inference.layers.rmsnorm import RMSNorm
from inference.layers.rotary import RotaryEmbedding
from inference.utils.loader import load_weights

from . import register


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int

    @classmethod
    def from_json(cls, path: Path) -> Qwen3Config:
        with path.open() as f:
            d = json.load(f)
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=float(d["rope_theta"]),
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            bos_token_id=d["bos_token_id"],
            eos_token_id=d["eos_token_id"],
        )


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.num_q = cfg.num_attention_heads
        self.num_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            cfg.hidden_size, self.num_kv * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size, self.num_kv * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_q * self.head_dim, cfg.hidden_size, bias=False)
        # Qwen3 QK-norm: per-head RMSNorm on Q and K (over head_dim).
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.attn = PagedAttention(self.num_q, self.num_kv, self.head_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        rotary: RotaryEmbedding,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        n_tok = hidden.shape[0]
        q = self.q_proj(hidden).view(n_tok, self.num_q, self.head_dim)
        k = self.k_proj(hidden).view(n_tok, self.num_kv, self.head_dim)
        v = self.v_proj(hidden).view(n_tok, self.num_kv, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rotary(q, k, positions)
        attn_out = self.attn(q, k, v, k_cache, v_cache)
        return self.o_proj(attn_out.reshape(n_tok, self.num_q * self.head_dim))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        rotary: RotaryEmbedding,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_attn(
            self.input_layernorm(hidden), positions, rotary, k_cache, v_cache
        )
        hidden = hidden + h
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class Qwen3ForCausalLM(nn.Module):
    """The full model. KV cache tensors live as parameters so they get moved
    to GPU with .to(device); the block manager addresses them by block_id."""

    def __init__(self, cfg: Qwen3Config, num_kv_blocks: int, block_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None  # use embed_tokens.weight in the forward.
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # YaRN-extend Qwen3 to ~131k tokens by default (the published config
        # supports it via `rope_scaling.type=yarn`, factor=4.0). We bake it in
        # so the 10000-line demo prompt fits without users wiring rope_scaling
        # themselves.
        yarn_factor = 4.0
        yarn_orig_max_pos = 32768
        extended_max_pos = int(yarn_orig_max_pos * yarn_factor)
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            extended_max_pos,
            base=cfg.rope_theta,
            yarn_factor=yarn_factor,
            yarn_orig_max_pos=yarn_orig_max_pos,
        )

        # KV cache pool, registered as buffers so .to(device) moves them.
        # Per-layer K and V each: (num_blocks, block_size, num_kv_heads, head_dim).
        kv_shape = (num_kv_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        for i in range(cfg.num_hidden_layers):
            self.register_buffer(
                f"k_cache_{i}", torch.zeros(kv_shape, dtype=torch.bfloat16)
            )
            self.register_buffer(
                f"v_cache_{i}", torch.zeros(kv_shape, dtype=torch.bfloat16)
            )

    def kv_for_layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_buffer(f"k_cache_{i}"), self.get_buffer(f"v_cache_{i}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        from inference.engine.context import get_context

        ctx = get_context()
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = self.kv_for_layer(i)
            hidden = layer(hidden, positions, self.rotary, k_cache, v_cache)
        hidden = self.norm(hidden)

        # We only need the LAST query token of each sequence to sample.
        last_idx = torch.cumsum(ctx.query_lens, dim=0) - 1
        last_hidden = hidden[last_idx]

        weight = (
            self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        )
        return last_hidden @ weight.T


# HF Qwen3 weights are prefixed `model.` for the body and `lm_head.weight` for the head.
# Our module nests body params directly (no `model.` prefix), so we strip it.
_QWEN3_REMAP: dict[str, str] = {}


def _build_remap(cfg: Qwen3Config) -> dict[str, str]:
    rm: dict[str, str] = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
    }
    for i in range(cfg.num_hidden_layers):
        for hf, ours in [
            (
                f"model.layers.{i}.input_layernorm.weight",
                f"layers.{i}.input_layernorm.weight",
            ),
            (
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"layers.{i}.post_attention_layernorm.weight",
            ),
            (
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"layers.{i}.self_attn.q_proj.weight",
            ),
            (
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"layers.{i}.self_attn.k_proj.weight",
            ),
            (
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"layers.{i}.self_attn.v_proj.weight",
            ),
            (
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"layers.{i}.self_attn.o_proj.weight",
            ),
            (
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"layers.{i}.self_attn.q_norm.weight",
            ),
            (
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"layers.{i}.self_attn.k_norm.weight",
            ),
            (
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"layers.{i}.mlp.gate_proj.weight",
            ),
            (f"model.layers.{i}.mlp.up_proj.weight", f"layers.{i}.mlp.up_proj.weight"),
            (
                f"model.layers.{i}.mlp.down_proj.weight",
                f"layers.{i}.mlp.down_proj.weight",
            ),
        ]:
            rm[hf] = ours
    if not cfg.tie_word_embeddings:
        rm["lm_head.weight"] = "lm_head.weight"
    return rm


@register("Qwen3ForCausalLM")
def load(model_dir: Path, num_kv_blocks: int, block_size: int) -> Qwen3ForCausalLM:
    cfg = Qwen3Config.from_json(model_dir / "config.json")
    model = Qwen3ForCausalLM(cfg, num_kv_blocks=num_kv_blocks, block_size=block_size)
    load_weights(
        model,
        model_dir,
        remap=_build_remap(cfg),
        dtype=torch.bfloat16,
        strict=False,  # buffers (k_cache_*, v_cache_*) won't be in the file.
    )
    return model
