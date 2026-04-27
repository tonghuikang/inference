"""Continuous batching scheduler.

Each step decides what runs in the next forward pass:
- If the running set is small and the waiting queue has prompts that fit, do
  a prefill step (admit one or more prompts).
- Otherwise do a decode step over the running set (one token per running seq).

If the running set runs out of KV blocks mid-decode, the youngest sequence is
preempted: its blocks are freed and it goes back to the waiting queue marked
PREEMPTED. Recompute on next admission.

This is the simplest scheduler that handles the things we care about
(prefix dedup, OOM under load, fairness across requests). It's intentionally
not the fastest — readability first.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

from inference.config import EngineConfig

from .block_manager import BlockManager
from .sequence import Sequence, SeqStatus


@dataclass
class SchedulerStep:
    """What ModelRunner should run this turn."""

    is_prefill: bool
    seqs: list[Sequence]


class Scheduler:
    def __init__(self, cfg: EngineConfig, block_mgr: BlockManager) -> None:
        self.cfg = cfg
        self.block_mgr = block_mgr
        self.waiting: collections.deque[Sequence] = collections.deque()
        self.running: list[Sequence] = []

    # -- request entry --------------------------------------------------------
    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def num_pending(self) -> int:
        return len(self.waiting) + len(self.running)

    # -- main loop ------------------------------------------------------------
    def schedule(self) -> SchedulerStep | None:
        """Pick prefill or decode. Returns None if there's nothing to do."""
        # Try to admit waiting sequences if we have headroom (prefill priority).
        if self.waiting and len(self.running) < self.cfg.max_num_seqs:
            admitted = self._admit_prefill_batch()
            if admitted:
                return SchedulerStep(is_prefill=True, seqs=admitted)

        # Otherwise decode whatever's running.
        if not self.running:
            return None
        # Each running seq needs at most 1 new block this step.
        while not self.block_mgr.can_allocate(len(self.running)):
            if not self._preempt_one():
                # Couldn't even free one block — implies a single sequence is
                # already at the pool size limit. That's an unrecoverable OOM.
                raise RuntimeError("KV pool exhausted with no preemptable sequence")
        return SchedulerStep(is_prefill=False, seqs=list(self.running))

    # -- post-step bookkeeping ------------------------------------------------
    def post_step(
        self,
        step: SchedulerStep,
        new_token_per_seq: list[int],
        finish_flags: list[bool],
    ) -> list[Sequence]:
        """Apply the step's outputs: append tokens to KV, mark finishes,
        promote prefill admittees to RUNNING. Returns finished seqs (caller
        emits responses + frees blocks)."""
        finished: list[Sequence] = []

        if step.is_prefill:
            # Prefill produces the first sampled token for each admitted seq.
            for seq, tok, done in zip(
                step.seqs, new_token_per_seq, finish_flags, strict=True
            ):
                seq.output_token_ids.append(tok)
                self.block_mgr.append_token(
                    seq.seq_id, seq.block_table, tok, position=seq.num_total - 1
                )
                seq.status = SeqStatus.RUNNING
                self.running.append(seq)
                if done:
                    self._finish(seq, finished)

        else:
            # Decode: each running seq emitted exactly one new token.
            for seq, tok, done in zip(
                step.seqs, new_token_per_seq, finish_flags, strict=True
            ):
                seq.output_token_ids.append(tok)
                self.block_mgr.append_token(
                    seq.seq_id, seq.block_table, tok, position=seq.num_total - 1
                )
                if done:
                    self._finish(seq, finished)

        return finished

    # -- internals ------------------------------------------------------------
    def _admit_prefill_batch(self) -> list[Sequence]:
        """Admit as many waiting seqs as fit in the per-step token budget AND
        the KV block budget (after counting prefix-dedup hits)."""
        admitted: list[Sequence] = []
        token_budget = self.cfg.max_num_batched_tokens

        while (
            self.waiting and len(self.running) + len(admitted) < self.cfg.max_num_seqs
        ):
            seq = self.waiting[0]
            n_tok = seq.num_prompt
            if n_tok > token_budget and admitted:
                break  # save it for the next prefill step.

            # Try to allocate. Hits don't consume free blocks; only fresh allocs do.
            # Cheap upper bound on free-block need:
            n_blocks_needed = seq.num_blocks_needed(self.block_mgr.block_size)
            if not self.block_mgr.can_allocate(n_blocks_needed):
                break

            try:
                seq.block_table = self.block_mgr.allocate_for_prompt(
                    seq.seq_id, seq.prompt_token_ids
                )
            except RuntimeError:
                break

            self.waiting.popleft()
            admitted.append(seq)
            token_budget -= n_tok

        return admitted

    def _preempt_one(self) -> bool:
        """Free the most-recently-admitted running seq's blocks and requeue it."""
        if not self.running:
            return False
        victim = self.running.pop()
        self.block_mgr.free(victim.seq_id, victim.block_table)
        victim.status = SeqStatus.PREEMPTED
        # Put back at front so it's retried first.
        self.waiting.appendleft(victim)
        return True

    def _finish(self, seq: Sequence, sink: list[Sequence]) -> None:
        seq.status = SeqStatus.FINISHED
        if seq in self.running:
            self.running.remove(seq)
        sink.append(seq)
