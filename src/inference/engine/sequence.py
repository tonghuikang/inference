"""Per-request state: token IDs, status, KV block table.

A `Sequence` is a single in-flight generation. The block table is a list of
physical block IDs in the order they're consumed; logical token position
maps to a (block_idx, slot) pair via integer division by `block_size`.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

import torch

from inference.config import SamplingParams


class SeqStatus(enum.Enum):
    WAITING = enum.auto()  # Admitted, no KV yet (queued for prefill).
    RUNNING = enum.auto()  # Has block_table, in scheduler's run set.
    FINISHED = enum.auto()  # Hit max_tokens / stop / EOS.
    PREEMPTED = enum.auto()  # KV freed; needs recompute on next admission.


_SEQ_COUNTER = itertools.count()


def next_seq_id() -> str:
    return f"seq-{next(_SEQ_COUNTER):08d}"


@dataclass
class Sequence:
    seq_id: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)  # physical block IDs.
    status: SeqStatus = SeqStatus.WAITING
    cumulative_logprob: float = 0.0
    finish_reason: str | None = None  # "length" | "stop" | "eos"

    # Cached on-device tensor of `block_table`. Rebuilt only when the table
    # actually grows (~once per `block_size` decode steps), not every step.
    # Hot-path optimisation — see model_runner.build_inputs.
    _block_table_tensor: torch.Tensor | None = field(default=None, repr=False)
    _cached_block_table_len: int = field(default=0, repr=False)

    def block_table_tensor(self, device: torch.device) -> torch.Tensor:
        n = len(self.block_table)
        cached = self._block_table_tensor
        if cached is None or cached.device != device or self._cached_block_table_len != n:
            self._block_table_tensor = torch.tensor(
                self.block_table, dtype=torch.long, device=device
            )
            self._cached_block_table_len = n
        return self._block_table_tensor

    @classmethod
    def new(cls, prompt: list[int], sampling: SamplingParams) -> Sequence:
        return cls(
            seq_id=next_seq_id(), prompt_token_ids=list(prompt), sampling=sampling
        )

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_prompt(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_total(self) -> int:
        return self.num_prompt + len(self.output_token_ids)

    def num_blocks_needed(self, block_size: int) -> int:
        return (self.num_total + block_size - 1) // block_size

    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED
