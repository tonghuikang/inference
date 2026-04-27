"""Prepare batched tensors for the model and run a forward pass.

For each scheduler step we build:

    input_ids:    (num_tokens,)            tokens to feed this step.
    positions:    (num_tokens,)            absolute position of each token.
    slot_mapping: (num_tokens,)            block_id*block_size + slot for KV write.
    block_tables: list[Tensor[num_blocks_for_seq]]  per-seq physical block IDs.
    seq_lens:     (batch,)                 total length AFTER this step's writes.
    query_lens:   (batch,)                 tokens contributed by this step.

In prefill: input_ids = full prompt for each admitted seq, positions = 0..L-1.
In decode:  input_ids = last sampled token per running seq, positions = L-1.
"""

from __future__ import annotations

from typing import TypedDict

import torch

from inference.config import EngineConfig

from .sequence import Sequence


class ModelInputs(TypedDict):
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: list[torch.Tensor]
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    is_prefill: bool


def _absolute_slot(block_id: int, slot: int, block_size: int) -> int:
    return block_id * block_size + slot


def build_inputs(
    seqs: list[Sequence], is_prefill: bool, block_size: int, device: torch.device
) -> ModelInputs:
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    block_tables: list[torch.Tensor] = []
    seq_lens: list[int] = []
    query_lens: list[int] = []

    for seq in seqs:
        if is_prefill:
            tokens = seq.prompt_token_ids
            base_pos = 0
            q_len = len(tokens)
            total_len = len(tokens)  # we're writing the full prompt now.
        else:
            tokens = (
                [seq.output_token_ids[-1]]
                if seq.output_token_ids
                else [seq.prompt_token_ids[-1]]
            )
            base_pos = seq.num_total - 1
            q_len = 1
            total_len = seq.num_total

        input_ids.extend(tokens)
        positions.extend(range(base_pos, base_pos + q_len))
        for offset in range(q_len):
            abs_pos = base_pos + offset
            block_idx, slot = divmod(abs_pos, block_size)
            slot_mapping.append(
                _absolute_slot(seq.block_table[block_idx], slot, block_size)
            )
        block_tables.append(seq.block_table_tensor(device))
        seq_lens.append(total_len)
        query_lens.append(q_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "positions": torch.tensor(positions, dtype=torch.long, device=device),
        "slot_mapping": torch.tensor(slot_mapping, dtype=torch.long, device=device),
        "block_tables": block_tables,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long, device=device),
        "query_lens": torch.tensor(query_lens, dtype=torch.long, device=device),
        "is_prefill": is_prefill,
    }


class ModelRunner:
    def __init__(
        self, model: torch.nn.Module, cfg: EngineConfig, device: torch.device
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor:
        inputs = build_inputs(seqs, is_prefill, self.cfg.block_size, self.device)
        return self.model(**inputs)  # (batch, vocab)
