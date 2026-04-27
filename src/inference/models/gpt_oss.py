"""gpt-oss-20b MoE model — STUB.

Full implementation deferred to v2 (alternating full/sliding-window attention,
top-k MoE routing, MXFP4-packed weights). The KV cache infrastructure that
ships in v1 is built so this file is the only thing that needs writing to
add the second architecture.

Until that lands, the model registry surfaces a clear NotImplementedError so
loading gpt-oss raises early at server startup rather than silently failing.
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from . import register


@register("GptOssForCausalLM")
def load(model_dir: Path, num_kv_blocks: int, block_size: int) -> nn.Module:
    raise NotImplementedError(
        "gpt-oss-20b model file not yet implemented. v1 ships with Qwen3 only; "
        "the docker vllm at /etc/systemd/system/vllm.service can serve gpt-oss "
        "until the MoE + sliding-window port lands."
    )
