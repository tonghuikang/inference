"""MXFP4 dequant for gpt-oss-20b expert weights.

gpt-oss-20b ships expert MLP weights in MXFP4: every 32 elements share one
8-bit power-of-two scale (E8M0), and each element is a 4-bit float (E2M1).
Two elements are packed per byte. We dequantise once at load time into bf16,
keeping the rest of the pipeline simple.

Format reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Layout in the safetensors file:
    weights/MoE.gate_up_proj_blocks: uint8 (num_experts, hidden, 2*int/2)
    weights/MoE.gate_up_proj_scales: uint8 (num_experts, hidden, 2*int/32)
    weights/MoE.down_proj_blocks:    uint8 (num_experts, int, hidden/2)
    weights/MoE.down_proj_scales:    uint8 (num_experts, int, hidden/32)

Each scale byte stores the E8M0 exponent: real_scale = 2 ** (s - 127).
"""

from __future__ import annotations

import torch

# E2M1 lookup: 16 4-bit codes → fp32 values.
# Sign in bit3, 2-bit exponent in bits 2-1, 1-bit mantissa in bit 0.
# Layout: 0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0,
#         8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_mxfp4(
    blocks: torch.Tensor,  # uint8, last dim packs two 4-bit values per byte
    scales: torch.Tensor,  # uint8, E8M0 exponent
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequant a block-quantised tensor. The last axis of `blocks` is half
    the unpacked length (two values per byte) and `scales` covers it in
    groups of 32 unpacked values (16 packed bytes)."""
    device = blocks.device
    lut = _E2M1_LUT.to(device)

    # Unpack each byte into two 4-bit codes along a new last axis.
    low = (blocks & 0x0F).long()
    high = ((blocks >> 4) & 0x0F).long()
    # Interleave: original order is (low_of_byte_0, high_of_byte_0, low_of_byte_1, ...).
    interleaved = torch.stack((low, high), dim=-1).reshape(*blocks.shape[:-1], -1)
    values = lut[interleaved]  # (..., 2*last_dim_bytes)

    # Scales: each covers 32 unpacked values along the last dim.
    real_scales = torch.pow(2.0, scales.float() - 127.0)
    real_scales = real_scales.unsqueeze(-1).expand(*scales.shape, 32)
    real_scales = real_scales.reshape(*scales.shape[:-1], -1)
    # Trim to values' last-dim size in case of padding mismatch.
    real_scales = real_scales[..., : values.shape[-1]]

    return (values * real_scales).to(out_dtype)
