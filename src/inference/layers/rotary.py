"""RoPE — rotary position embedding (HuggingFace split-halves variant) +
optional YaRN extension so we can serve Qwen3 at its full 131k context.

cos/sin tables are (max_position, head_dim). At application time we split
q/k into halves and rotate:

    q_embed = q * cos + rotate_half(q) * sin
    rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])

YaRN ("NTK-by-parts" with attention temperature scaling) replaces the
standard inv_freq with a per-dim blend of extrapolation and interpolation,
plus a small magnitude factor on cos/sin that compensates the softmax for
the longer attention window. This is what Qwen3 ships in its
`rope_scaling.type=yarn` config, and we enable it by default with
factor=4.0 (40k -> 160k) so the 10000-line demo prompt fits.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ---------------------------------------------------------------------------
# YaRN helpers — verbatim algorithm from huggingface/transformers'
# `modeling_rope_utils._compute_yarn_parameters`. Kept inline so the whole
# RoPE story lives in one file.
# ---------------------------------------------------------------------------


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, device: torch.device) -> torch.Tensor:
    if low == high:
        high += 0.001  # avoid div-by-zero.
    x = torch.arange(dim, dtype=torch.float32, device=device)
    return torch.clamp((x - low) / (high - low), 0.0, 1.0)


def _build_inv_freq(
    head_dim: int,
    base: float,
    device: torch.device,
    yarn_factor: float | None,
    yarn_orig_max_pos: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """Standard inv_freq when yarn_factor is None; otherwise the NTK-by-parts blend."""
    half = head_dim // 2
    pos_freqs = base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    inv_freq = 1.0 / pos_freqs

    if yarn_factor is None or yarn_factor == 1.0:
        return inv_freq

    inv_freq_extrapolation = inv_freq
    inv_freq_interpolation = inv_freq / yarn_factor
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, head_dim, base, yarn_orig_max_pos
    )
    extrapolation_factor = 1.0 - _yarn_linear_ramp_mask(low, high, half, device)
    return inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor


def _build_cos_sin(
    head_dim: int,
    max_position: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
    yarn_factor: float | None,
    yarn_orig_max_pos: int,
    yarn_attention_factor: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = _build_inv_freq(head_dim, base, device, yarn_factor, yarn_orig_max_pos)
    t = torch.arange(max_position, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_position, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_position, head_dim)

    # YaRN attention magnitude scaling: equivalent to bumping softmax temperature
    # by `attention_factor`. Folding it into cos/sin is convenient and matches HF.
    if yarn_factor and yarn_factor != 1.0:
        if yarn_attention_factor is None:
            yarn_attention_factor = 0.1 * math.log(yarn_factor) + 1.0
        cos = (emb.cos() * yarn_attention_factor).to(dtype)
        sin = (emb.sin() * yarn_attention_factor).to(dtype)
    else:
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """x: (..., n_heads, head_dim). cos/sin: (max_position, head_dim).
    positions: (...,) integer tensor with the absolute position of each token."""
    cos_p = cos[positions].unsqueeze(-2)
    sin_p = sin[positions].unsqueeze(-2)
    return (x * cos_p) + (_rotate_half(x) * sin_p)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        yarn_factor: float | None = None,
        yarn_orig_max_pos: int | None = None,
        yarn_attention_factor: float | None = None,
    ) -> None:
        super().__init__()
        device = device or torch.device("cuda")
        cos, sin = _build_cos_sin(
            head_dim,
            max_position,
            base,
            device,
            dtype,
            yarn_factor=yarn_factor,
            yarn_orig_max_pos=yarn_orig_max_pos or max_position,
            yarn_attention_factor=yarn_attention_factor,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_rotary(q, self.cos, self.sin, positions),
            apply_rotary(k, self.cos, self.sin, positions),
        )
