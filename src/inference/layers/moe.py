"""Mixture-of-Experts layer: top-k router + per-expert SwiGLU MLP.

Implements the gpt-oss-20b style: 32 experts, 4 active per token, router
chooses experts via softmax(top_k(logits)). The expert MLPs are the bulk
of the parameters and ship MXFP4-quantised in the gpt-oss checkpoint —
`weights/MoE.gate_up_proj` and `weights/MoE.down_proj` arrive as packed
4-bit tensors plus per-block scale factors. We dequantise to bf16 once at
load time (the engine is bf16 throughout); see `quant_mxfp4.dequant_mxfp4`.

The forward is the simple "expert-loop" form — one matmul per active
expert per step. With 32 experts and top-k=4, that's at most 4 expert
calls per token (route-grouped). For the small batch sizes we care about,
the loop is fine; production-grade MoE needs grouped-GEMM, but this is a
v1 readable implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        swiglu_limit: float = 7.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.swiglu_limit = swiglu_limit
        # Router: hidden -> num_experts logits. Stored full-precision.
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        # Stacked expert weights — one row per expert.
        # gate_up_proj: (num_experts, 2 * intermediate_size, hidden_size)
        # down_proj:    (num_experts, hidden_size, intermediate_size)
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_tokens, hidden_size)
        n_tok = x.shape[0]
        router_logits = self.router(x)  # (num_tokens, num_experts)
        # Top-k experts per token.
        topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1, dtype=torch.float32).to(x.dtype)

        out = torch.zeros_like(x)
        # Loop the active expert set; skip experts that no token routed to.
        flat_idx = topk_idx.flatten()
        for e in flat_idx.unique().tolist():
            # Find which (token, k-slot) entries chose this expert.
            mask = topk_idx == e  # (num_tokens, top_k)
            tok_idx, slot = mask.nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            x_e = x[tok_idx]  # (n, hidden)
            w_gate_up = self.gate_up_proj[e]  # (2*int, hidden)
            w_down = self.down_proj[e]  # (hidden, int)
            gate_up = x_e @ w_gate_up.T  # (n, 2*int)
            gate, up = gate_up.chunk(2, dim=-1)
            # SwiGLU with optional clamp (gpt-oss uses swiglu_limit).
            up = up.clamp(max=self.swiglu_limit)
            mlp_out = (F.silu(gate) * up) @ w_down.T  # (n, hidden)
            # Weight by router prob and accumulate.
            w = topk_weights[tok_idx, slot].unsqueeze(-1)  # (n, 1)
            out.index_add_(0, tok_idx, mlp_out * w)
        _ = n_tok
        return out
