"""memory-service as an MCP server: two tools, search_memory and write_memory.

Design:
- The tools go through the HTTP API (MemoryClient), not straight to Postgres. Auth, RLS and input
  limits all live in the API, so the MCP layer can't bypass them; it is just another client.
- stdio transport: the MCP host (Claude Desktop, MCP Inspector) starts this process and speaks
  JSON-RPC over stdin/stdout. So stdout belongs to the protocol. Never print(); logs go to stderr.
- Login is lazy (first tool call), so the server starts and lists its tools even when the API is
  down; the tool call then fails with a readable error instead of the process crashing.

Config (env): MEMORY_API_URL (default http://localhost:8000), MEMORY_EMAIL, MEMORY_PASSWORD.

Run:  python -m memory_service.mcp_server
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .client import MemoryClient, SessionExpired

log = logging.getLogger("memory_service.mcp")


class ApiSession:
    """A MemoryClient that logs in on first use and logs in again once if the session ended."""

    def __init__(self, client: MemoryClient, email: str | None, password: str | None):
        self.client = client
        self.email = email
        self.password = password

    def _login(self) -> None:
        if not (self.email and self.password):
            raise ToolError("MEMORY_EMAIL and MEMORY_PASSWORD must be set for the MCP server")
        self.client.login(self.email, self.password)
        log.info("logged in to memory API as %s", self.email)

    def call(self, fn, *args: Any, **kw: Any) -> Any:
        if self.client.access_token is None:
            self._login()
        try:
            return fn(*args, **kw)
        except SessionExpired:  # refresh token expired/revoked: one fresh login, then give up
            log.info("session expired, logging in again")
            self._login()
            return fn(*args, **kw)


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
        return _api_errors(session.call, session.client.search, query, mode=mode, k=k)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
    def write_memory(text: str, importance: float = 0.5) -> dict[str, str]:
        """Store one memory for the user (max 4000 chars). importance 0..1, default 0.5.
        Returns {id} of the new memory."""
        return {"id": _api_errors(session.call, session.client.remember, text, importance=importance)}

    return server


def _api_errors(call, *args: Any, **kw: Any) -> Any:
    """Turn HTTP failures into short messages the model can read and act on. Only ToolError text
    reaches the client; any other exception shows up as a bare "Error executing tool"."""
    try:
        return call(*args, **kw)
    except httpx.ConnectError as e:
        raise ToolError(f"memory API unreachable at {os.environ.get('MEMORY_API_URL', 'http://localhost:8000')}") from e
    except httpx.HTTPStatusError as e:
        r = e.response
        detail = r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "") else r.text
        raise ToolError(f"memory API {r.status_code}: {detail}") from e


def main() -> None:
    # stderr only: stdout is the JSON-RPC channel.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    client = MemoryClient(base_url=os.environ.get("MEMORY_API_URL", "http://localhost:8000"))
    session = ApiSession(client, os.environ.get("MEMORY_EMAIL"), os.environ.get("MEMORY_PASSWORD"))
    build_server(session).run("stdio")


if __name__ == "__main__":
    main()
