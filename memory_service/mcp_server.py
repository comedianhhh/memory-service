"""memory-service as an MCP server: two tools, search_memory and write_memory.

Design:
- The tools go through the HTTP API (MemoryClient), not straight to Postgres. Auth, RLS and input
  limits all live in the API, so the MCP layer can't bypass them; it is just another client.
- stdio transport: the MCP host (Claude Desktop, MCP Inspector) starts this process and speaks
  JSON-RPC over stdin/stdout. So stdout belongs to the protocol. Never print(); logs go to stderr.
- Login is lazy (first tool call), so the server starts and lists its tools even when the API is
  down; the tool call then fails with a readable error instead of the process crashing.
- MEMORY_API_URL=inprocess runs the FastAPI app inside this process (Starlette TestClient) instead of
  calling a server over the network. Requests still go through the same routes, JWT auth and RLS;
  there is just no separate API process to keep running. The DB is whatever MEMORY_DSN points at.

Config (env): MEMORY_API_URL (http://... or "inprocess"; default http://localhost:8000), MEMORY_EMAIL,
MEMORY_PASSWORD. MEMORY_ENV_FILE=<path> loads KEY=VALUE lines from that file first (already-set env
wins), so secrets can stay in the repo's git-ignored .env instead of the MCP host's config.

Run:       python -m memory_service.mcp_server
Register:  python -m memory_service.mcp_server register   (creates the MEMORY_EMAIL account once)
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx
import psycopg
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from psycopg_pool import PoolTimeout

from .client import MemoryClient, SessionExpired

log = logging.getLogger("memory_service.mcp")


class ApiSession:
    """A MemoryClient that is built and logged in on first use, and logs in again once if the
    session ended. `client` may be a factory, so an in-process app only starts on the first call."""

    def __init__(
        self,
        client: MemoryClient | Callable[[], MemoryClient],
        email: str | None,
        password: str | None,
    ):
        self._client = client if isinstance(client, MemoryClient) else None
        self._make = None if isinstance(client, MemoryClient) else client
        self.email = email
        self.password = password

    @property
    def client(self) -> MemoryClient:
        if self._client is None:
            self._client = self._make()
        return self._client

    def _login(self) -> None:
        if not (self.email and self.password):
            raise ToolError("MEMORY_EMAIL and MEMORY_PASSWORD must be set for the MCP server")
        self.client.login(self.email, self.password)
        log.info("logged in to memory API as %s", self.email)

    def call(self, fn: Callable[[MemoryClient], Any]) -> Any:
        if self.client.access_token is None:
            self._login()
        try:
            return fn(self.client)
        except SessionExpired:  # refresh token expired/revoked: one fresh login, then give up
            log.info("session expired, logging in again")
            self._login()
            return fn(self.client)


def build_server(session: ApiSession) -> MCPServer:
    server = MCPServer(
        name="memory-service",
        instructions=(
            "Long-term memory for the user. Call search_memory before answering questions that may "
            "depend on things the user said in earlier conversations. Call write_memory when the user "
            "states a durable fact or preference worth remembering."
        ),
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def search_memory(
        query: str, k: int = 5, mode: Literal["hybrid", "vector", "fts"] = "hybrid"
    ) -> list[dict[str, Any]]:
        """Search the user's stored memories. Returns up to k hits, best first, each {id, text, source_id}.
        mode: hybrid (default, vector + keyword fused with RRF), vector (semantic only), fts (keyword only)."""
        return _api_errors(session.call, lambda c: c.search(query, mode=mode, k=k))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
    def write_memory(text: str, importance: float = 0.5) -> dict[str, str]:
        """Store one memory for the user (max 4000 chars). importance 0..1, default 0.5.
        Returns {id} of the new memory."""
        return {"id": _api_errors(session.call, lambda c: c.remember(text, importance=importance))}

    return server


def _api_errors(call, *args: Any, **kw: Any) -> Any:
    """Turn failures into short messages the model can read and act on. Only ToolError text
    reaches the client; any other exception shows up as a bare "Error executing tool"."""
    try:
        return call(*args, **kw)
    except httpx.ConnectError as e:
        raise ToolError(f"memory API unreachable at {os.environ.get('MEMORY_API_URL', 'http://localhost:8000')}") from e
    except (psycopg.OperationalError, PoolTimeout) as e:  # in-process mode: the database itself is down
        raise ToolError(f"memory database unreachable: {str(e).splitlines()[0] if str(e) else type(e).__name__}") from e
    except httpx.HTTPStatusError as e:
        r = e.response
        detail = r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "") else r.text
        raise ToolError(f"memory API {r.status_code}: {detail}") from e


def load_env_file(path: str) -> None:
    """KEY=VALUE lines; blank lines and # comments skipped; variables already set are kept."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def make_client(api_url: str) -> MemoryClient:
    if api_url != "inprocess":
        return MemoryClient(base_url=api_url)
    # Imported here: memory_service.db reads MEMORY_DSN at import time, after the env file is loaded.
    from starlette.testclient import TestClient

    from .api import app

    http = TestClient(app, base_url="http://inprocess")
    http.__enter__()  # runs the app's lifespan (opens the DB pool); lives as long as the process
    return MemoryClient(http=http)


def main() -> None:
    # stderr only: stdout is the JSON-RPC channel.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if os.environ.get("MEMORY_ENV_FILE"):
        load_env_file(os.environ["MEMORY_ENV_FILE"])
    api_url = os.environ.get("MEMORY_API_URL", "http://localhost:8000")
    email, password = os.environ.get("MEMORY_EMAIL"), os.environ.get("MEMORY_PASSWORD")

    if sys.argv[1:] == ["register"]:
        r = make_client(api_url).http.post("/auth/register", json={"email": email, "password": password})
        sys.stderr.write(f"register {email}: {r.status_code} {r.text}\n")
        sys.exit(0 if r.status_code in (201, 409) else 1)

    session = ApiSession(lambda: make_client(api_url), email, password)
    build_server(session).run("stdio")


if __name__ == "__main__":
    main()
