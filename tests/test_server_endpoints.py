"""HTTP-level smoke tests using FastAPI's TestClient.

We skip anything that requires real weights and instead build a fake worker
that just exposes a block_manager + small chat-completion path. This catches
routing, auth, and JSON-schema regressions without needing a 1.5 GB download
sitting on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.config import EngineConfig, ModelConfig  # noqa: E402
from inference.engine.block_manager import BlockManager  # noqa: E402
from inference.server import EngineWorker, _make_app  # noqa: E402


@pytest.fixture()
def fake_worker() -> EngineWorker:
    """Construct an EngineWorker whose engine is mocked but whose block_mgr is real."""
    bm = BlockManager(num_blocks=8, block_size=4)
    # Simulate two allocated blocks so /observer/snapshot returns something interesting.
    bm.allocate_for_prompt("seq-test", token_ids=list(range(8)))

    engine = MagicMock()
    engine.block_mgr = bm
    engine.model_cfg = ModelConfig(model_id="fake/model", path=Path("/tmp"))
    engine.engine_cfg = EngineConfig(num_kv_blocks=8, block_size=4)
    engine.tokenizer = MagicMock()

    worker = EngineWorker(engine=engine, loop=None)  # type: ignore[arg-type]
    return worker


def test_models_endpoint(fake_worker: EngineWorker, tmp_path: Path):
    app = _make_app({"fake/model": fake_worker}, tmp_path)
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "fake/model"


def test_observer_snapshot(fake_worker: EngineWorker, tmp_path: Path):
    app = _make_app({"fake/model": fake_worker}, tmp_path)
    client = TestClient(app)
    resp = client.get("/observer/snapshot")
    assert resp.status_code == 200
    snap = resp.json()
    assert snap["model"] == "fake/model"
    assert snap["block_size"] == 4
    assert snap["num_blocks"] == 8
    assert len(snap["blocks"]) == 8
    # Two blocks should be allocated (seq-test took 2 full blocks of 4 tokens).
    owned = [b for b in snap["blocks"] if b["refcount"] >= 1]
    assert len(owned) == 2
    assert "seq-test" in owned[0]["owners"]


def test_observer_static_index(fake_worker: EngineWorker, tmp_path: Path):
    """The /observer/ mount should serve the index.html shipped in src/."""
    static = (
        Path(__file__).resolve().parents[1] / "src" / "inference" / "web" / "static"
    )
    app = _make_app({"fake/model": fake_worker}, static)
    client = TestClient(app)
    resp = client.get("/observer/")
    assert resp.status_code == 200
    assert "<title>KV cache observer</title>" in resp.text
    # The JS + CSS should load too.
    assert client.get("/observer/app.js").status_code == 200
    assert client.get("/observer/style.css").status_code == 200


def test_unknown_model_returns_404(fake_worker: EngineWorker, tmp_path: Path):
    app = _make_app({"fake/model": fake_worker}, tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/v1/completions",
        json={"model": "nope", "prompt": "hi", "max_tokens": 1},
    )
    assert resp.status_code == 404
