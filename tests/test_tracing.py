"""timed() blocks become nested spans of one trace, and log lines carry that trace's id.
Uses an in-memory exporter, so no collector is needed."""

from __future__ import annotations

import io
import json
import logging

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from memory_service import logs

from .conftest import requires_db

pytestmark = requires_db

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _provider():
    # The global provider can be set once per process; nothing else in the suite sets one.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))  # Simple: exported at span end, no batching
    trace.set_tracer_provider(provider)
    yield


@pytest.fixture
def spans():
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


@pytest.fixture
def json_lines():
    """Log lines formatted at emit time, i.e. inside the active span (unlike caplog, which formats later)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logs.JsonFormatter())
    logs.logger.addHandler(handler)
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines()]
    logs.logger.removeHandler(handler)


def test_search_steps_are_children_of_one_search_span(client, account, spans, json_lines):
    a = account()
    client.post("/me/memories", headers=a["headers"], json={"text": "dentist on Friday"})
    spans.clear()

    client.get("/me/memories/search", params={"q": "dentist"}, headers=a["headers"])

    by_name: dict[str, list] = {}
    for s in spans.get_finished_spans():
        by_name.setdefault(s.name, []).append(s)
    (search,) = by_name["search"]
    children = by_name["retrieve"] + by_name["fuse"]
    assert {s.attributes["app.method"] for s in by_name["retrieve"]} == {"vector", "fts"}
    for child in children:
        assert child.context.trace_id == search.context.trace_id
        assert child.parent.span_id == search.context.span_id
    assert search.attributes["app.mode"] == "hybrid" and search.attributes["app.duration_ms"] > 0

    trace_hex = format(search.context.trace_id, "032x")
    step_lines = [l for l in json_lines() if l["event"] in ("search", "retrieve", "fuse")]
    assert step_lines and all(l["trace_id"] == trace_hex for l in step_lines)


def test_failed_step_marks_span_as_error(spans):
    with pytest.raises(ValueError):
        with logs.timed("boom"):
            raise ValueError("bad input")
    (s,) = [s for s in spans.get_finished_spans() if s.name == "boom"]
    assert s.status.status_code == trace.StatusCode.ERROR
    assert s.attributes["app.error"] == "ValueError"
    assert any(e.name == "exception" for e in s.events)
