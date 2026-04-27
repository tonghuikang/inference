"""Run one short generation against Qwen3-0.6B end-to-end without the HTTP layer.

Useful for validating that real weights load and produce coherent output before
launching the FastAPI server.

    HF_HOME=/srv/vllm/hf uv run python scripts/smoke_qwen3.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from inference.config import EngineConfig, ModelConfig, SamplingParams
from inference.engine.llm_engine import LlmEngine

REPO_ID = "Qwen/Qwen3-0.6B"
HF_HOME = Path(os.environ.get("HF_HOME", "/srv/vllm/hf"))


def _resolve(repo: str) -> Path:
    folder = HF_HOME / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    return sorted(folder.iterdir())[-1]


def main() -> int:
    snap = _resolve(REPO_ID)
    print(f"loading {REPO_ID} from {snap}")
    eng_cfg = EngineConfig(num_kv_blocks=1024, block_size=16, max_num_seqs=4)
    engine = LlmEngine(ModelConfig(model_id=REPO_ID, path=snap), eng_cfg)

    prompt = "<|im_start|>user\nWrite a haiku about transformers.<|im_end|>\n<|im_start|>assistant\n"
    params = SamplingParams(temperature=0.0, max_tokens=64)

    print("---")
    text = ""
    for out in engine.stream(prompt, params):
        sys.stdout.write(out.text_delta)
        sys.stdout.flush()
        text += out.text_delta
    print("\n---")
    print(f"finished: {len(text)} chars, {engine.block_mgr.num_blocks - engine.block_mgr.num_free} blocks in use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
