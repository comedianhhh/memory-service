from __future__ import annotations

import json
import logging

from .conftest import requires_db

pytestmark = requires_db


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["pool"]["size"] >= 1


def test_message_is_chunked_and_each_chunk_is_searchable(client, account):
    a, b = account(), account()
    text = (
        "My dentist moved the appointment to November 21. "
        + "Filler sentence about nothing in particular here. " * 20
        + "Also Biscuit the cat needs his thyroid pills at night."
    )
    r = client.post("/me/messages", json={"text": text}, headers=a["headers"])
    assert r.status_code == 201
    body = r.json()
    assert len(body["memory_ids"]) > 1
    assert all(i.startswith(body["message_id"]) for i in body["memory_ids"])

    s = lambda q, who: client.get("/me/memories/search", params={"q": q, "mode": "fts"}, headers=who["headers"])  # noqa: E731
    first, last = s("dentist", a).json()["results"][0], s("thyroid", a).json()["results"][0]
    assert first["id"] == body["memory_ids"][0] and "dentist" in first["text"]
    assert last["id"] == body["memory_ids"][-1] and "thyroid" in last["text"]
    assert first["source_id"] == last["source_id"] == body["message_id"]
    assert s("dentist", b).json()["results"] == []


def test_write_then_search_each_mode(client, account):
    a = account()
    mid = client.post("/me/memories", json={"text": "dentist appointment on November 14"}, headers=a["headers"]).json()["id"]
    client.post("/me/memories", json={"text": "winter tires go on mid November"}, headers=a["headers"])
    for mode in ("hybrid", "vector", "fts"):
        body = client.get("/me/memories/search", params={"q": "dentist", "mode": mode}, headers=a["headers"]).json()
        assert body["results"][0]["id"] == mid, mode


def test_validation(client, account):
    a = account()
    assert client.post("/me/memories", json={"text": ""}, headers=a["headers"]).status_code == 422
    assert client.get("/me/memories/search", params={"q": "x", "mode": "bm25"}, headers=a["headers"]).status_code == 422


def test_old_user_id_in_path_routes_are_gone(client, account):
    a = account()
    assert client.get(f"/users/{a['user_id']}/memories/search", params={"q": "x"}).status_code == 404


def test_logs_are_one_json_object_per_line(client, account, caplog):
    a = account()
    lg = logging.getLogger("memory_service")
    lg.addHandler(caplog.handler)
    try:
        client.get("/me/memories/search", params={"q": "dentist"}, headers=a["headers"])
    finally:
        lg.removeHandler(caplog.handler)
    from memory_service.logs import JsonFormatter

    lines = [json.loads(JsonFormatter().format(r)) for r in caplog.records]
    events = {line["event"] for line in lines}
    assert {"retrieve", "fuse", "search", "http"} <= events
    assert all(isinstance(line["duration_ms"], float) for line in lines if "duration_ms" in line)
