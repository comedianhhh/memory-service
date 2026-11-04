from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from memory_service.api import app, get_embedder
from memory_service.embed import FakeEmbedder

from .conftest import requires_db

pytestmark = requires_db


@pytest.fixture
def client():
    app.dependency_overrides[get_embedder] = FakeEmbedder
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["pool"]["size"] >= 1


def test_message_is_chunked_and_each_chunk_is_searchable(client, users):
    a, b = users
    text = (
        "My dentist moved the appointment to November 21. "
        + "Filler sentence about nothing in particular here. " * 20
        + "Also Biscuit the cat needs his thyroid pills at night."
    )
    r = client.post(f"/users/{a}/messages", json={"text": text})
    assert r.status_code == 201
    body = r.json()
    assert len(body["memory_ids"]) > 1
    assert all(i.startswith(body["message_id"]) for i in body["memory_ids"])

    first = client.get(f"/users/{a}/memories/search", params={"q": "dentist", "mode": "fts"}).json()["results"][0]
    last = client.get(f"/users/{a}/memories/search", params={"q": "thyroid", "mode": "fts"}).json()["results"][0]
    assert first["id"] == body["memory_ids"][0] and "dentist" in first["text"]
    assert last["id"] == body["memory_ids"][-1] and "thyroid" in last["text"]
    assert first["source_id"] == last["source_id"] == body["message_id"]
    assert client.get(f"/users/{b}/memories/search", params={"q": "dentist"}).json()["results"] == []


def test_write_then_search_each_mode(client, users):
    a, _ = users
    r = client.post(f"/users/{a}/memories", json={"text": "dentist appointment on November 14"})
    assert r.status_code == 201
    mid = r.json()["id"]
    client.post(f"/users/{a}/memories", json={"text": "winter tires go on mid November"})
    for mode in ("hybrid", "vector", "fts"):
        body = client.get(f"/users/{a}/memories/search", params={"q": "dentist", "mode": mode}).json()
        assert body["results"][0]["id"] == mid, mode


def test_search_is_isolated_per_user(client, users):
    a, b = users
    client.post(f"/users/{a}/memories", json={"text": "A only: flight AC 1207"})
    assert client.get(f"/users/{b}/memories/search", params={"q": "AC 1207"}).json()["results"] == []


def test_validation(client, users):
    a, _ = users
    assert client.get("/users/not-a-uuid/memories/search", params={"q": "x"}).status_code == 422
    assert client.post(f"/users/{a}/memories", json={"text": ""}).status_code == 422
    assert client.get(f"/users/{a}/memories/search", params={"q": "x", "mode": "bm25"}).status_code == 422


def test_logs_are_one_json_object_per_line(client, users, caplog):
    a, _ = users
    lg = logging.getLogger("memory_service")
    lg.addHandler(caplog.handler)
    try:
        client.get(f"/users/{a}/memories/search", params={"q": "dentist"})
    finally:
        lg.removeHandler(caplog.handler)
    from memory_service.logs import JsonFormatter

    lines = [json.loads(JsonFormatter().format(r)) for r in caplog.records]
    events = {line["event"] for line in lines}
    assert {"retrieve", "fuse", "search", "http"} <= events
    assert all(isinstance(line["duration_ms"], float) for line in lines)
