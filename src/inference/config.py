"""Runtime configuration: model, engine, sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Per-model serving config."""

    model_id: str  # e.g. "Qwen/Qwen3-0.6B" — also the HF repo path on disk.
    path: Path  # Resolved local snapshot directory.
    dtype: str = "bfloat16"
    max_model_len: int = 8192


@dataclass
class EngineConfig:
    """Global engine config; one instance shared across loaded models."""

    block_size: int = 64  # Tokens per KV block. Larger blocks = fewer cells in
    # the observer grid (more legible) at the cost of less granular prefix
    # dedup. 64 still hits well on long shared prompts.
    num_kv_blocks: int = 1024  # Pool size; tune to GPU memory.
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 1024  # Concurrency cap. There's no fundamental limit
    # — the engine batches every running sequence into a single forward pass.
    # The cap just bounds memory growth from scheduler bookkeeping.
    max_num_batched_tokens: int = 32768  # Per-step token budget for prefill admission.
    enforce_eager: bool = True  # No CUDA graphs in v1.
    kv_observer_log: Path | None = None  # If set, append KV events to this file.


@dataclass
class SamplingParams:
    """Per-request sampling knobs."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled.
    max_tokens: int = 512
    min_tokens: int = 0  # Generate at least this many tokens before EOS can fire.
    ignore_eos: bool = False  # If true, never stop on EOS — only on max_tokens.
    stop: list[str] = field(default_factory=list)
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0
