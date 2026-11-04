"""Python client that keeps a session alive on its own.

- access token expired (401 token_expired) -> refresh once, retry the request once
- refresh fails (expired / revoked / reused)  -> SessionExpired: the caller must log in again
- connection dropped / timed out              -> retry with backoff, only for idempotent requests

The refresh is behind a lock: refresh tokens are single use, so two threads refreshing with the
same token would look like token theft and the server would revoke the whole session.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx

RETRYABLE_AUTH = {"token_expired"}
IDEMPOTENT = {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"}


class SessionExpired(Exception):
    """Refresh is no longer possible; log in again."""


class MemoryClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        http: httpx.Client | None = None,
        retries: int = 3,
        backoff: float = 0.2,
        sleep=time.sleep,
    ):
        self.http = http or httpx.Client(base_url=base_url, timeout=10)
        self.retries = retries
        self.backoff = backoff
        self._sleep = sleep
        self._lock = threading.Lock()
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    # ---- session ----------------------------------------------------------------------------------

    def login(self, email: str, password: str) -> None:
        r = self.http.post("/auth/login", json={"email": email, "password": password})
        if r.status_code == 401:
            raise SessionExpired(r.json().get("error"))
        r.raise_for_status()
        self._store(r.json())

    def logout(self) -> None:
        if self.access_token:
            self._send("POST", "/auth/logout")
        self.access_token = self.refresh_token = None

    def _store(self, body: dict[str, Any]) -> None:
        self.access_token = body["access_token"]
        self.refresh_token = body["refresh_token"]

    def _refresh(self, stale_access: str | None) -> None:
        with self._lock:
            if self.access_token != stale_access:
                return  # another thread already refreshed while we waited
            if not self.refresh_token:
                raise SessionExpired("not_logged_in")
            r = self.http.post("/auth/refresh", json={"refresh_token": self.refresh_token})
            if r.status_code == 401:
                self.access_token = self.refresh_token = None
                raise SessionExpired(r.json().get("error"))
            r.raise_for_status()
            self._store(r.json())

    # ---- transport --------------------------------------------------------------------------------

    def _send(self, method: str, url: str, **kw: Any) -> httpx.Response:
        """One request with auth, plus transport retries for idempotent methods."""
        attempts = self.retries if method in IDEMPOTENT else 1
        for i in range(attempts):
            headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
            try:
                return self.http.request(method, url, headers=headers, **kw)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                if i == attempts - 1:
                    raise
                self._sleep(self.backoff * 2**i)
        raise AssertionError("unreachable")

    def request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        used = self.access_token
        r = self._send(method, url, **kw)
        if r.status_code == 401 and r.json().get("error") in RETRYABLE_AUTH:
            self._refresh(used)
            r = self._send(method, url, **kw)
        if r.status_code == 401:
            raise SessionExpired(r.json().get("error"))
        r.raise_for_status()
        return r

    # ---- API --------------------------------------------------------------------------------------

    def remember(self, text: str, importance: float = 0.5) -> str:
        return self.request("POST", "/me/memories", json={"text": text, "importance": importance}).json()["id"]

    def add_message(self, text: str) -> dict[str, Any]:
        return self.request("POST", "/me/messages", json={"text": text}).json()

    def search(self, q: str, mode: str = "hybrid", k: int = 5) -> list[dict[str, Any]]:
        return self.request("GET", "/me/memories/search", params={"q": q, "mode": mode, "k": k}).json()["results"]
