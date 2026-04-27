"""Pub/sub event bus for KV cache lifecycle events.

The block manager calls `emit(event)` on every alloc / free / hit / append.
Subscribers are: a log-file sink (text) and the WebSocket server (live UI).
Kept tiny on purpose — this is a learning aid, not a metrics framework.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

EventKind = Literal["alloc", "release", "hit", "append", "evict"]


@dataclass
class KVEvent:
    kind: EventKind
    block_id: int
    seq_id: str = ""
    refcount: int = 0
    block_hash: int | None = None
    tokens: list[int] = field(default_factory=list)
    layer_group: str = "default"  # e.g. "full" / "window-128" for gpt-oss.

    def to_json(self) -> str:
        return json.dumps(asdict(self))


Subscriber = Callable[[KVEvent], None]


class KVObserver:
    """Single global event bus. Thread-safe (block ops can happen off the
    main thread when async streaming wakes the engine)."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._lock = threading.Lock()
        self._log_handle = None

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return unsubscribe

    def attach_logfile(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = path.open("a", buffering=1)  # line-buffered
        logging.getLogger(__name__).info("kv-observer log -> %s", path)

    def emit(self, event: KVEvent) -> None:
        with self._lock:
            subs = list(self._subs)
        if self._log_handle is not None:
            self._log_handle.write(event.to_json() + "\n")
        for fn in subs:
            try:
                fn(event)
            except (RuntimeError, OSError, ConnectionError) as exc:
                # Subscribers are best-effort sinks (logfile, WebSocket queues).
                # A dead WebSocket or full asyncio queue surfaces as RuntimeError /
                # ConnectionError; we log and keep dispatching to the rest.
                logging.getLogger(__name__).warning("kv-observer sink failed: %s", exc)


_GLOBAL = KVObserver()


def get_observer() -> KVObserver:
    return _GLOBAL
