"""One JSON object per line on stderr: {"ts", "event", "duration_ms", ...fields}.

With tracing on, every line also carries trace_id / span_id, and every timed() block is also a span,
so a slow bar in the trace waterfall leads straight to that request's log lines and back.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .telemetry import current_ids

logger = logging.getLogger("memory_service")
_tracer = trace.get_tracer("memory_service")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "level": record.levelname.lower()}
        payload.update(getattr(record, "fields", {}) or {"msg": record.getMessage()})
        payload.update(current_ids())  # formatting runs in the caller's thread, inside its span
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
    """Time a block, log it, and record it as a span named `event`.
    The block can add fields (e.g. hits, tokens) to the yielded dict; they become span attributes too."""
    extra: dict[str, Any] = dict(fields)
    start = time.perf_counter()
    with _tracer.start_as_current_span(event, record_exception=False, set_status_on_exception=False) as span:
        try:
            yield extra
        except Exception as exc:
            extra["error"] = type(exc).__name__
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            extra["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
            for key, value in extra.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(f"app.{key}", value)
            logger.info(event, extra={"fields": {"event": event, **extra}})
