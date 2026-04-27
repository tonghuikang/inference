"""Model architecture registry. Each value is a (loader_callable, default_window) pair."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

ModelLoader = Callable[..., nn.Module]
_REGISTRY: dict[str, ModelLoader] = {}


def register(arch: str) -> Callable[[ModelLoader], ModelLoader]:
    def deco(fn: ModelLoader) -> ModelLoader:
        _REGISTRY[arch] = fn
        return fn

    return deco


def get(arch: str) -> ModelLoader:
    if arch not in _REGISTRY:
        raise KeyError(
            f"no model loader registered for {arch!r}; have {sorted(_REGISTRY)}"
        )
    return _REGISTRY[arch]


# Eager imports so registration happens at package load.
from . import qwen3 as _qwen3  # noqa: E402, F401
from . import gpt_oss as _gpt_oss  # noqa: E402, F401
