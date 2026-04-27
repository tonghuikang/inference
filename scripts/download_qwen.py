"""Fetch Qwen3-0.6B into the shared HF cache at /srv/vllm/hf.

Run:
    HF_HOME=/srv/vllm/hf uv run python scripts/download_qwen.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-0.6B"
CACHE = Path(os.environ.get("HF_HOME", "/srv/vllm/hf"))


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=str(CACHE / "hub"),
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"downloaded {REPO_ID} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
