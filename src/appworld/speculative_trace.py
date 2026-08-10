"""Structured JSONL tracing for speculative execution."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

from appworld.common.time import freezegun_bypassed_datetime


@dataclass
class TraceRecorder:
    """Append process-safe-enough, line-buffered events to one JSONL file.

    The lock protects threads in one process. Isolated worker processes should
    write separate files and merge them by ``timestamp`` and ``round_id``.
    """

    file_path: str | Path
    run_id: str
    _sequence: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._sequence += 1
            payload = {
                "schema_version": 1,
                "timestamp": freezegun_bypassed_datetime().astimezone(UTC).isoformat(),
                "monotonic_ns": monotonic_ns(),
                "run_id": self.run_id,
                "sequence": self._sequence,
                "event": event,
                **fields,
            }
            with self.file_path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def monotonic_ns() -> int:
    """Return real monotonic time even while AppWorld freezes Python clocks."""

    if sys.platform.startswith("win"):
        return int(freezegun_bypassed_datetime().timestamp() * 1_000_000_000)

    class Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    library_path = ctypes.util.find_library("rt") or ctypes.util.find_library("c")
    if not library_path:
        raise RuntimeError("Cannot find libc/librt for a real monotonic clock.")
    library = ctypes.CDLL(library_path, use_errno=True)
    clock_monotonic = 6 if sys.platform == "darwin" else 1
    value = Timespec()
    if library.clock_gettime(clock_monotonic, ctypes.byref(value)) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "clock_gettime(CLOCK_MONOTONIC) failed")
    return value.tv_sec * 1_000_000_000 + value.tv_nsec


def elapsed_ms(start_ns: int) -> float:
    return round((monotonic_ns() - start_ns) / 1_000_000, 3)


def response_usage(response: Any) -> dict[str, Any] | None:
    """Convert SDK usage objects without depending on one SDK version."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return usage
    return {
        name: getattr(usage, name)
        for name in ("input_tokens", "output_tokens", "total_tokens")
        if getattr(usage, name, None) is not None
    }
