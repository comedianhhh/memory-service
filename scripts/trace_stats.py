"""Drive real traffic through the api, then read the traces back from Jaeger and print p50 / p95 per
step (span name). Needs: docker compose -f docker-compose.yml -f docker-compose.otel.yml up -d --build

    .venv/Scripts/python scripts/trace_stats.py [--memories 40] [--searches 100]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime

API = "http://localhost:8000"
JAEGER = "http://localhost:16686"

MEMORIES = [
    "I am allergic to peanuts", "My sister lives in Vancouver", "I skip lunch on Mondays",
    "Dentist appointment on Friday at 3pm", "My cat Biscuit needs thyroid pills twice a day",
    "I prefer window seats on flights", "I'm learning French and Korean", "I run 5k every Sunday",
    "My manager is called Priya", "I don't drink coffee after 2pm", "My birthday is in March",
    "I moved to Toronto in 2024", "I use a standing desk", "I hate cilantro",
]
QUERIES = [
    "what food should I avoid", "where does my family live", "pet medication", "when is the dentist",
    "travel preferences", "what languages am I learning", "exercise routine", "who is my boss",
    "caffeine habits", "what herbs do I dislike",
]


def call(method: str, path: str, body: dict | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req) as r:
        return json.load(r) if r.status != 204 else {}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memories", type=int, default=40)
    ap.add_argument("--searches", type=int, default=100)
    args = ap.parse_args()

    cred = {"email": f"trace-{uuid.uuid4().hex[:8]}@example.test", "password": "trace-" + uuid.uuid4().hex}
    call("POST", "/auth/register", cred)
    token = call("POST", "/auth/login", cred)["access_token"]
    for i in range(args.memories):
        call("POST", "/me/memories", {"text": MEMORIES[i % len(MEMORIES)]}, token)
    call("GET", "/me/memories/search?q=warmup", token=token)  # first call loads the model; not counted
    time.sleep(1)
    since = datetime.now(UTC)
    for i in range(args.searches):
        q = urllib.parse.quote(QUERIES[i % len(QUERIES)])
        call("GET", f"/me/memories/search?q={q}", token=token)

    time.sleep(6)  # BatchSpanProcessor flushes every ~5 s
    # Jaeger v2 query API v3: returns OTLP JSON (resourceSpans -> scopeSpans -> spans).
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    params = urllib.parse.urlencode({
        "query.service_name": "memory-service",
        "query.operation_name": "GET /me/memories/search",  # search_depth caps traces of any kind
        "query.start_time_min": since.strftime(fmt),
        "query.start_time_max": datetime.now(UTC).strftime(fmt),
        "query.search_depth": args.searches + 50,
    })
    with urllib.request.urlopen(f"{JAEGER}/api/v3/traces?{params}") as r:
        result = json.load(r)["result"]

    by_trace: dict[str, list[dict]] = {}
    for rs in result["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for s in ss["spans"]:
                by_trace.setdefault(s["traceId"], []).append(s)

    durations: dict[str, list[float]] = {}
    n_traces = 0
    for spans in by_trace.values():
        if not any(s["name"] == "search" for s in spans):
            continue
        n_traces += 1
        for s in spans:
            ms = (int(s["endTimeUnixNano"]) - int(s["startTimeUnixNano"])) / 1e6
            durations.setdefault(s["name"], []).append(ms)

    print(f"{n_traces} search traces from Jaeger ({args.memories} memories, {args.searches} searches)\n")
    print(f"{'span':42} {'n':>5} {'p50 ms':>8} {'p95 ms':>8}")
    labels = {"memory": "(pool health-check ping, empty query)", "WITH": "SQL WITH (full-text query)"}
    for name, xs in sorted(durations.items(), key=lambda kv: -statistics.median(kv[1])):
        label = labels.get(name, name)
        print(f"{label[:42]:42} {len(xs):>5} {pct(xs, 50):>8.2f} {pct(xs, 95):>8.2f}")


if __name__ == "__main__":
    main()
