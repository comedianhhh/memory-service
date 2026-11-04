"""MemoryClient keeps the session alive by itself, and gives up cleanly when it can't."""

from __future__ import annotations

import httpx
import pytest

from memory_service.client import MemoryClient, SessionExpired

from .conftest import requires_db


@requires_db
def test_auto_refresh_when_access_token_expires(client, account, clock):
    a = account()
    mc = MemoryClient(http=client)
    mc.login(a["email"], a["password"])
    mid = mc.remember("Biscuit needs thyroid pills")
    first_refresh = mc.refresh_token

    clock.advance(minutes=16)  # access token is dead now
    hits = mc.search("thyroid", mode="fts")  # 401 token_expired -> refresh -> retry, invisibly

    assert hits[0]["id"] == mid
    assert mc.refresh_token != first_refresh  # it rotated


@requires_db
def test_session_end_surfaces_as_session_expired(client, account, clock):
    a = account()
    mc = MemoryClient(http=client)
    mc.login(a["email"], a["password"])
    clock.advance(days=8)  # past refresh idle timeout
    with pytest.raises(SessionExpired, match="refresh_token_expired"):
        mc.search("anything")
    assert mc.access_token is None and mc.refresh_token is None


@requires_db
def test_second_refresh_with_same_stale_token_is_a_noop(client, account, clock):
    """Two threads both see 401 with the same old access token. Only the first may refresh;
    a second refresh would replay a used refresh token and get the session revoked."""
    a = account()
    mc = MemoryClient(http=client)
    mc.login(a["email"], a["password"])
    stale = mc.access_token
    clock.advance(minutes=16)
    mc._refresh(stale)
    after_first = mc.refresh_token
    mc._refresh(stale)  # the "second thread"
    assert mc.refresh_token == after_first
    assert mc.search("x") == []  # session still alive


def test_transport_retry_only_for_idempotent_requests():
    calls = {"GET": 0, "POST": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls[request.method] += 1
        if calls[request.method] < 3:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"results": [], "id": "m1"})

    mc = MemoryClient(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t"), sleep=lambda _: None)
    mc.access_token = "t"
    assert mc.search("x") == []
    assert calls["GET"] == 3  # failed twice, retried, succeeded
    with pytest.raises(httpx.ConnectError):
        mc.remember("x")  # POST is not idempotent: a retry could store the memory twice
    assert calls["POST"] == 1
