"""KV cache block manager — the central observability surface.

A pool of `num_blocks` physical KV blocks, each holding `block_size` tokens'
worth of K and V. Blocks are reference-counted so a single physical block can
be shared across sequences whose prefixes hash to the same content.

Lifecycle:
    fresh  --alloc--> owned (refcount=1)
    owned --ref()--> shared (refcount>1)
    shared --deref--> owned
    owned --deref--> evictable (refcount=0; CONTENT KEPT)
    evictable --hit/ref--> owned (rescued before another seq pulls the slot)
    evictable --evict--> fresh, then alloc (some other seq needed the slot)

Evictable means "still has its contents and hash, just nobody is using it
right now". A second sequence whose prompt prefix matches this block's hash
can rescue it for free. We only physically clear (`reset()`) a block when
something else actually pulls it off the LRU.

`evictable` is an OrderedDict so popping the head gives us LRU-order
eviction. Insertion-order is the recency of last-deref-to-zero.

Every state transition emits a KVEvent so the observer log + UI can render
the cache live.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import xxhash

from inference.utils.kv_observer import KVEvent, get_observer


@dataclass
class Block:
    block_id: int
    block_size: int
    refcount: int = 0
    block_hash: int | None = None  # xxh64 of (parent_hash, token_ids); None if partial.
    token_ids: list[int] = field(default_factory=list)
    owners: set[str] = field(
        default_factory=set
    )  # seq_ids holding a ref (for UI display).

    def reset(self) -> None:
        self.refcount = 0
        self.block_hash = None
        self.token_ids.clear()
        self.owners.clear()


def compute_block_hash(parent_hash: int | None, tokens: tuple[int, ...]) -> int:
    """Rolling hash: hash(parent_hash || tokens). Stable across positions so
    two sequences with the same prefix produce the same per-block hashes."""
    h = xxhash.xxh64()
    if parent_hash is not None:
        h.update(parent_hash.to_bytes(8, "little", signed=False))
    h.update(b"".join(t.to_bytes(4, "little", signed=False) for t in tokens))
    return h.intdigest()


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        layer_group: str = "full",
        window: int | None = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.layer_group = layer_group
        self.window = window  # If set, this pool serves sliding-window layers.
        self.blocks: list[Block] = [Block(i, block_size) for i in range(num_blocks)]
        # All blocks start evictable (they have no content yet — they'll be
        # picked up in alloc order). OrderedDict makes head=oldest, so we
        # always evict the LRU when we need a fresh slot.
        self.evictable: collections.OrderedDict[int, None] = collections.OrderedDict(
            (i, None) for i in range(num_blocks)
        )
        self.hash_to_block: dict[int, int] = {}  # block_hash -> block_id (full blocks only).
        self.observer = get_observer()

    # -- introspection --------------------------------------------------------
    @property
    def num_free(self) -> int:
        """How many blocks could be allocated right now (i.e. evictable)."""
        return len(self.evictable)

    def snapshot(self) -> list[dict]:
        """Return everything the UI needs to render the grid."""
        return [
            {
                "block_id": b.block_id,
                "refcount": b.refcount,
                "block_hash": b.block_hash,
                "token_ids": list(b.token_ids),
                "owners": sorted(b.owners),
                "evictable": b.refcount == 0 and b.block_hash is not None,
            }
            for b in self.blocks
        ]

    def can_allocate(self, n: int) -> bool:
        return self.num_free >= n

    # -- alloc / free ---------------------------------------------------------
    def allocate_for_prompt(self, seq_id: str, token_ids: list[int]) -> list[int]:
        """Allocate (or reuse via prefix-hash) blocks covering all prompt tokens."""
        bs = self.block_size
        block_table: list[int] = []
        parent_hash: int | None = None

        n_full = len(token_ids) // bs
        for i in range(n_full):
            window = tuple(token_ids[i * bs : (i + 1) * bs])
            h = compute_block_hash(parent_hash, window)
            cached_id = self.hash_to_block.get(h)
            if cached_id is not None and self.blocks[cached_id].block_hash == h:
                self._ref(cached_id, seq_id)
                self.observer.emit(
                    KVEvent(
                        kind="hit",
                        block_id=cached_id,
                        seq_id=seq_id,
                        refcount=self.blocks[cached_id].refcount,
                        block_hash=h,
                        tokens=list(window),
                        layer_group=self.layer_group,
                    )
                )
            else:
                cached_id = self._allocate_block(seq_id, list(window), block_hash=h)
            block_table.append(cached_id)
            parent_hash = h

        tail = token_ids[n_full * bs :]
        if tail:
            block_id = self._allocate_block(seq_id, list(tail), block_hash=None)
            block_table.append(block_id)
        return block_table

    def append_token(
        self, seq_id: str, block_table: list[int], token_id: int, position: int
    ) -> int | None:
        """Place one freshly-generated token into the cache. Returns a NEW
        block_id if we had to allocate one; else None."""
        bs = self.block_size
        block_idx, _slot = divmod(position, bs)

        if block_idx >= len(block_table):
            block_id = self._allocate_block(seq_id, [token_id], block_hash=None)
            block_table.append(block_id)
            return block_id

        blk = self.blocks[block_table[block_idx]]
        blk.token_ids.append(token_id)

        # When we just filled this block, finalize its hash so future seqs can dedup.
        if len(blk.token_ids) == bs and blk.block_hash is None:
            parent_hash = (
                self.blocks[block_table[block_idx - 1]].block_hash
                if block_idx > 0
                else None
            )
            blk.block_hash = compute_block_hash(parent_hash, tuple(blk.token_ids))
            self.hash_to_block.setdefault(blk.block_hash, blk.block_id)

        self.observer.emit(
            KVEvent(
                kind="append",
                block_id=blk.block_id,
                seq_id=seq_id,
                refcount=blk.refcount,
                block_hash=blk.block_hash,
                tokens=list(blk.token_ids),
                layer_group=self.layer_group,
            )
        )
        return None

    def free(self, seq_id: str, block_table: list[int]) -> None:
        """Drop the seq's references. Blocks with refcount→0 stay in the cache
        (evictable) until something else needs them — that's how a finished
        request still helps the next one via prefix-hash hits."""
        for block_id in block_table:
            self._deref(block_id, seq_id)
        block_table.clear()

    # -- internals ------------------------------------------------------------
    def _allocate_block(
        self, seq_id: str, token_ids: list[int], block_hash: int | None
    ) -> int:
        if not self.evictable:
            raise RuntimeError(
                f"out of KV blocks (pool={self.num_blocks}, group={self.layer_group}); "
                "scheduler should have preempted before calling _allocate_block",
            )
        # popitem(last=False) → LRU.
        block_id, _ = self.evictable.popitem(last=False)
        blk = self.blocks[block_id]

        # If we're repurposing a block that was holding cached content, evict
        # its hash entry first so future lookups don't point at stale data.
        if blk.block_hash is not None and self.hash_to_block.get(blk.block_hash) == block_id:
            del self.hash_to_block[blk.block_hash]
            self.observer.emit(
                KVEvent(
                    kind="evict",
                    block_id=block_id,
                    seq_id=seq_id,  # the seq taking over the slot
                    refcount=0,
                    block_hash=blk.block_hash,
                    tokens=list(blk.token_ids),
                    layer_group=self.layer_group,
                )
            )

        blk.reset()
        blk.refcount = 1
        blk.block_hash = block_hash
        blk.token_ids = list(token_ids)
        blk.owners.add(seq_id)
        if block_hash is not None:
            self.hash_to_block.setdefault(block_hash, block_id)

        self.observer.emit(
            KVEvent(
                kind="alloc",
                block_id=block_id,
                seq_id=seq_id,
                refcount=1,
                block_hash=block_hash,
                tokens=list(token_ids),
                layer_group=self.layer_group,
            )
        )
        return block_id

    def _ref(self, block_id: int, seq_id: str) -> None:
        blk = self.blocks[block_id]
        if blk.refcount == 0:
            # Rescue from evictable set — somebody hit the prefix cache.
            self.evictable.pop(block_id, None)
        blk.refcount += 1
        blk.owners.add(seq_id)

    def _deref(self, block_id: int, seq_id: str) -> None:
        blk = self.blocks[block_id]
        blk.refcount -= 1
        blk.owners.discard(seq_id)
        if blk.refcount == 0:
            # Mark as evictable but KEEP content + hash so the next request can
            # prefix-hit it. Move-to-end so it's the youngest in LRU order.
            self.evictable[block_id] = None
            self.evictable.move_to_end(block_id)
        # Emit the deref so the UI can update colors (refcount changed).
        self.observer.emit(
            KVEvent(
                kind="release",
                block_id=block_id,
                seq_id=seq_id,
                refcount=blk.refcount,
                block_hash=blk.block_hash,
                tokens=list(blk.token_ids),
                layer_group=self.layer_group,
            )
        )
