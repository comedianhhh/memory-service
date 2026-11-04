"""One JSON object per line on stderr: {"ts", "event", "duration_ms", ...fields}."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("memory_service")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "level": record.levelname.lower()}
        payload.update(getattr(record, "fields", {}) or {"msg": record.getMessage()})
        return json.dumps(payload, ensure_ascii=False)


def setup(level: int = logging.INFO) -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and log it. The block can add fields (e.g. hits, tokens) to the yielded dict."""
    extra: dict[str, Any] = dict(fields)
    start = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        extra["error"] = type(exc).__name__
        raise
    finally:
        extra["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
        logger.info(event, extra={"fields": {"event": event, **extra}})
