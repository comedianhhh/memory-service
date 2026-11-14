"""The MCP server: tools are listed, stdout stays clean JSON-RPC, and calls reach the real API."""

from __future__ import annotations

import os
import sys

import anyio
import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from memory_service.client import MemoryClient
from memory_service.mcp_server import ApiSession, build_server

from .conftest import requires_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_stdio_lists_tools_and_reports_errors_as_tool_errors():
    """Spawn the server the way Claude Desktop does (subprocess over stdio). If anything printed to
    stdout, the client would fail to parse the stream. No API is running: the call must come back
    as an isError result, and the process must stay alive for the next request."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "memory_service.mcp_server"],
        cwd=ROOT,
        env={"MEMORY_API_URL": "http://127.0.0.1:9", "MEMORY_EMAIL": "x@example.test", "MEMORY_PASSWORD": "irrelevant"},
    )

    async def run():
        with open(os.devnull, "w") as quiet:  # server logs go to stderr; keep test output tidy
            async with stdio_client(params, errlog=quiet) as (read, write):
                async with ClientSession(read, write) as s:
                    await s.initialize()
                    tools = {t.name: t for t in (await s.list_tools()).tools}
                    assert set(tools) == {"search_memory", "write_memory"}
                    assert tools["search_memory"].annotations.read_only_hint is True
                    assert tools["search_memory"].input_schema["required"] == ["query"]

                    r = await s.call_tool("search_memory", {"query": "anything"})
                    assert r.is_error
                    assert "unreachable" in r.content[0].text

                    again = await s.list_tools()  # still alive after a failed call
                    assert len(again.tools) == 2

    anyio.run(run)


@requires_db
def test_write_then_search_through_the_tools(client, account):
    a = account()
    session = ApiSession(MemoryClient(http=client), a["email"], a["password"])
    server = build_server(session)

    async def run():
        w = await server.call_tool("write_memory", {"text": "Biscuit the cat takes thyroid pills", "importance": 0.8})
        mid = w.structured_content["id"]
        s = await server.call_tool("search_memory", {"query": "thyroid", "mode": "fts"})
        hits = s.structured_content["result"]
        assert hits[0]["id"] == mid and "thyroid" in hits[0]["text"]

    anyio.run(run)


@requires_db
def test_relogin_when_session_is_gone(client, account, clock):
    a = account()
    mc = MemoryClient(http=client)
    session = ApiSession(mc, a["email"], a["password"])
    server = build_server(session)

    async def run():
        await server.call_tool("write_memory", {"text": "first"})
        clock.advance(days=8)  # past the refresh idle timeout: refresh fails -> SessionExpired
        r = await server.call_tool("search_memory", {"query": "first", "mode": "fts"})
        assert r.structured_content["result"][0]["text"] == "first"

    anyio.run(run)


def test_missing_credentials_is_a_clear_error():
    session = ApiSession(MemoryClient(http=httpx.Client(base_url="http://127.0.0.1:9")), None, None)
    with pytest.raises(ToolError, match="MEMORY_EMAIL"):
        session.call(lambda: None)


@requires_db
def test_inprocess_mode_needs_no_api_server(account):
    """MEMORY_API_URL=inprocess: the subprocess hosts the FastAPI app itself and talks to the DB
    directly via MEMORY_DSN. Nothing listens on a port; auth and RLS still apply."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from memory_service.db import APP_DSN

    a = account()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "memory_service.mcp_server"],
        cwd=ROOT,
        env={
            "MEMORY_API_URL": "inprocess",
            "MEMORY_DSN": APP_DSN,
            "MEMORY_EMAIL": a["email"],
            "MEMORY_PASSWORD": a["password"],
        },
    )

    async def run():
        with open(os.devnull, "w") as quiet:
            async with stdio_client(params, errlog=quiet) as (read, write):
                async with ClientSession(read, write) as s:
                    await s.initialize()
                    w = await s.call_tool("write_memory", {"text": "inprocess mode stores this"})
                    assert not w.is_error, w.content
                    r = await s.call_tool("search_memory", {"query": "inprocess", "mode": "fts"})
                    assert r.structured_content["result"][0]["id"] == w.structured_content["id"]

    anyio.run(run)


def test_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    from memory_service.mcp_server import load_env_file

    f = tmp_path / ".env"
    f.write_text("# comment\nMS_TEST_A=from_file\nMS_TEST_B=x=y\n\n", encoding="utf-8")
    monkeypatch.setenv("MS_TEST_A", "already_set")
    monkeypatch.delenv("MS_TEST_B", raising=False)
    load_env_file(str(f))
    assert os.environ["MS_TEST_A"] == "already_set"
    assert os.environ["MS_TEST_B"] == "x=y"
