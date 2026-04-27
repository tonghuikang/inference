"""Load HF safetensors weights into a torch.nn.Module via a name-remap table.

Each model file declares a `WEIGHT_REMAP: dict[str, str]` mapping HF state-dict
keys -> our module's parameter names. Anything missing from the remap is loaded
as-is.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


def _load_index(model_dir: Path) -> dict[str, str] | None:
    """Return tensor -> shard-filename map if a sharded index exists."""
    idx = model_dir / "model.safetensors.index.json"
    if not idx.exists():
        return None
    with idx.open() as f:
        return json.load(f)["weight_map"]


def _iter_state_dict(model_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (key, tensor) pairs from one or more safetensors shards."""
    index = _load_index(model_dir)
    if index is None:
        single = model_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"no safetensors at {model_dir}")
        with safe_open(single, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)
        return

    shards = sorted(set(index.values()))
    for shard in shards:
        path = model_dir / shard
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def load_weights(
    model: nn.Module,
    model_dir: Path,
    remap: dict[str, str] | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> list[str]:
    """Load weights from `model_dir` into `model`. Returns list of HF keys
    that were not consumed (useful for catching naming mismatches early)."""
    remap = remap or {}
    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    used: set[str] = set()
    leftover: list[str] = []

    for hf_key, tensor in _iter_state_dict(model_dir):
        target = remap.get(hf_key, hf_key)
        if target not in own:
            leftover.append(hf_key)
            continue
        param = own[target]
        if param.shape != tensor.shape:
            raise ValueError(
                f"shape mismatch for {hf_key} -> {target}: "
                f"have {param.shape}, file has {tensor.shape}"
            )
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        param.data.copy_(tensor)
        used.add(target)

    if strict:
        missing = sorted(set(own) - used)
        if missing:
            raise ValueError(
                f"missing weights: {missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
    return leftover
