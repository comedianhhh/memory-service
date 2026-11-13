"""OpenTelemetry tracing.

Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set (tests and plain local runs export nothing). When on:
- FastAPI: one server span per request (the root of the trace)
- psycopg: one span per SQL statement
- botocore: one span per AWS call (Secrets Manager, S3)
- logs.timed(): one span per pipeline step (embed, retrieve, fuse, rerank, export, job, ...)
Spans go to an OTLP/HTTP collector (Jaeger locally; on AWS, the ADOT collector or X-Ray).

Across the queue, the trace context rides inside the job payload (inject() at enqueue, resumed() in
the worker), the same W3C `traceparent` an HTTP call would carry in a header.

Everything except the setup functions uses only the opentelemetry API, a no-op until a provider is set.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import context, propagate, trace


def _install(default_service: str) -> bool:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", default_service)})
    )
    # Batch: spans are queued and sent in the background, so a slow collector never slows a request.
    # The exporter reads OTEL_EXPORTER_OTLP_ENDPOINT itself and appends /v1/traces.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    PsycopgInstrumentor().instrument(enable_commenter=False)
    return True


def setup(app) -> bool:
    """API. Call once at import, before the DB pool opens, so pooled connections are instrumented."""
    if not _install("memory-service"):
        return False
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health",  # health checks would drown the rest
        exclude_spans=["send", "receive"],  # one span per ASGI message is noise, keep the request span
    )
    BotocoreInstrumentor().instrument()
    return True


def setup_worker() -> bool:
    return _install("memory-worker")


def inject() -> dict[str, str]:
    """The current trace context as a dict ({"traceparent": ...}) to store with a job."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


@contextmanager
def resumed(carrier: dict[str, str] | None) -> Iterator[None]:
    """Make spans opened inside this block children of the context saved by inject()."""
    if not carrier:
        yield
        return
    token = context.attach(propagate.extract(carrier))
    try:
        yield
    finally:
        context.detach(token)


def current_ids() -> dict[str, str]:
    """trace_id / span_id of the active span, for log lines. Empty when no span is recording."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}
