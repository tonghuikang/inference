"""Token sampler: greedy / temperature / top-p / top-k."""

from __future__ import annotations

import torch

from inference.config import SamplingParams


def sample(logits: torch.Tensor, params_per_seq: list[SamplingParams]) -> list[int]:
    """logits: (batch, vocab_size). Returns one token per batch row."""
    out: list[int] = []
    for i, sp in enumerate(params_per_seq):
        row = logits[i]
        if sp.greedy:
            out.append(int(row.argmax().item()))
            continue

        if sp.temperature != 1.0:
            row = row / max(sp.temperature, 1e-5)

        if sp.top_k > 0:
            v, _ = torch.topk(row, sp.top_k)
            row = torch.where(row < v[-1], torch.full_like(row, float("-inf")), row)

        probs = torch.softmax(row, dim=-1)

        if 0.0 < sp.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > sp.top_p
            # Keep at least one token (the most likely).
            mask[0] = False
            sorted_probs[mask] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
            probs = probs / probs.sum()

        token = int(torch.multinomial(probs, num_samples=1).item())
        out.append(token)
    return out
