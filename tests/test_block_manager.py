"""Block manager unit tests — alloc / free / refcount / prefix-hash dedup
/ LRU eviction / deferred-eviction prefix rescue."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.engine.block_manager import BlockManager, compute_block_hash  # noqa: E402


def test_alloc_uses_lru_in_fifo_order():
    bm = BlockManager(num_blocks=4, block_size=4)
    bt = bm.allocate_for_prompt("seq-A", token_ids=list(range(4)) + list(range(4, 8)))
    assert bt == [0, 1]  # first two blocks pulled in order from the LRU.
    assert bm.num_free == 2  # 2 still evictable.


def test_partial_block_at_tail():
    bm = BlockManager(num_blocks=4, block_size=4)
    bt = bm.allocate_for_prompt("seq-A", token_ids=list(range(6)))  # 4+2
    assert len(bt) == 2
    # Tail block has no hash.
    assert bm.blocks[bt[1]].block_hash is None
    assert bm.blocks[bt[1]].token_ids == [4, 5]


def test_prefix_hash_dedup_shares_block():
    bm = BlockManager(num_blocks=8, block_size=4)
    prompt = list(range(16))  # 4 full blocks worth.
    bt_a = bm.allocate_for_prompt("seq-A", prompt)
    free_after_a = bm.num_free
    bt_b = bm.allocate_for_prompt("seq-B", prompt)
    assert bt_a == bt_b
    assert bm.num_free == free_after_a  # no fresh blocks consumed for B.
    for bid in bt_a:
        assert bm.blocks[bid].refcount == 2


def test_deref_keeps_block_evictable_with_content_intact():
    """When a sequence finishes, its blocks should NOT be wiped — they stay
    cached until something else needs the slot, so a follow-up request can
    rescue them via prefix-hash."""
    bm = BlockManager(num_blocks=4, block_size=4)
    prompt = list(range(8))
    bt_a = bm.allocate_for_prompt("seq-A", prompt)
    block_ids = list(bt_a)
    bm.free("seq-A", bt_a)
    # Pool is "free" (evictable) but the blocks still hold their content.
    assert bm.num_free == 4
    for bid in block_ids:
        assert bm.blocks[bid].refcount == 0
        assert bm.blocks[bid].token_ids != []
        assert bm.blocks[bid].block_hash is not None


def test_finished_request_blocks_get_rescued_by_next_matching_prefix():
    bm = BlockManager(num_blocks=4, block_size=4)
    prompt = list(range(8))
    bt_a = bm.allocate_for_prompt("seq-A", prompt)
    bm.free("seq-A", bt_a)
    # Same prefix: should re-use the same physical blocks (no fresh alloc).
    bt_b = bm.allocate_for_prompt("seq-B", prompt)
    assert sorted(bt_b) == sorted([0, 1])  # same physical IDs as A used.
    for bid in bt_b:
        assert bm.blocks[bid].refcount == 1
    # Pool is now "in use" by B — only the still-unused blocks remain free.
    assert bm.num_free == 2


def test_lru_evicts_oldest_finished_block_when_pool_pressured():
    bm = BlockManager(num_blocks=4, block_size=4)
    # Fill pool with seq-A, free it. All 4 blocks are evictable with content.
    bt_a = bm.allocate_for_prompt("seq-A", list(range(16)))
    a_blocks = list(bt_a)
    bm.free("seq-A", bt_a)
    assert bm.num_free == 4

    # Brand new content (no prefix overlap) needs 4 blocks → must evict A's.
    bt_b = bm.allocate_for_prompt("seq-B", list(range(100, 116)))
    assert sorted(bt_b) == sorted(a_blocks)  # same physical pool.
    # And A's hashes are gone now (we repurposed those slots).
    assert bm.num_free == 0


def test_compute_block_hash_is_position_chained():
    h0 = compute_block_hash(None, (1, 2, 3))
    h1 = compute_block_hash(h0, (4, 5, 6))
    h0b = compute_block_hash(None, (1, 2, 3))
    h1b = compute_block_hash(h0b, (4, 5, 6))
    assert h0 == h0b
    assert h1 == h1b
    h_swapped = compute_block_hash(h1, (1, 2, 3))
    assert h_swapped != h0


def test_append_token_finalizes_hash_at_block_boundary():
    bm = BlockManager(num_blocks=4, block_size=4)
    bt = bm.allocate_for_prompt("seq-A", token_ids=[1, 2])
    bm.append_token("seq-A", bt, 3, position=2)
    bm.append_token("seq-A", bt, 4, position=3)
    block = bm.blocks[bt[0]]
    assert len(block.token_ids) == 4
    assert block.block_hash is not None
