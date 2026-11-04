"""HTTP API. Run: uvicorn memory_service.api:app --port 8000"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import logs
from .db import make_pool, user_session
from .embed import Embedder, FastEmbedder
from .search import add_memory, add_message, search

logs.setup()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # a closed psycopg pool can't be reopened, so each app start gets a fresh one
    app.state.pool = make_pool()
    app.state.pool.open(wait=True, timeout=10)
    try:
        yield
    finally:
        app.state.pool.close()


app = FastAPI(title="memory-service", lifespan=lifespan)


def get_embedder() -> Embedder:
    return FastEmbedder()  # tests override this with FakeEmbedder


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    with request.app.state.pool.connection(timeout=5) as conn:  # app_user, RLS on; back to the pool afterwards
        yield conn


Conn = Annotated[psycopg.Connection, Depends(get_conn)]
Emb = Annotated[Embedder, Depends(get_embedder)]


@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    logs.logger.info(
        "http",
        extra={"fields": {
            "event": "http",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        }},
    )
    return response


class MemoryIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    importance: float = Field(0.5, ge=0, le=1)


class MemoryOut(BaseModel):
    id: str


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    importance: float = Field(0.5, ge=0, le=1)


class MessageOut(BaseModel):
    message_id: str
    memory_ids: list[str]


class Hit(BaseModel):
    id: str
    text: str
    source_id: str | None = None


class SearchOut(BaseModel):
    query: str
    mode: str
    results: list[Hit]


@app.get("/health")
def health(request: Request, conn: Conn) -> dict[str, object]:
    conn.execute("SELECT 1")
    s = request.app.state.pool.get_stats()
    return {"status": "ok", "pool": {"size": s.get("pool_size"), "available": s.get("pool_available")}}


# NOTE: user_id comes from the path for now; with real auth it must come from the verified token.


@app.post("/users/{user_id}/memories", status_code=201)
def create_memory(user_id: uuid.UUID, body: MemoryIn, conn: Conn, embedder: Emb) -> MemoryOut:
    with user_session(conn, user_id):
        memory_id = add_memory(conn, embedder, str(user_id), body.text, body.importance)
    return MemoryOut(id=memory_id)


@app.post("/users/{user_id}/messages", status_code=201)
def create_message(user_id: uuid.UUID, body: MessageIn, conn: Conn, embedder: Emb) -> MessageOut:
    """A raw chat message of any length: chunked, embedded, stored as several memories."""
    with user_session(conn, user_id):
        message_id, ids = add_message(conn, embedder, str(user_id), body.text, body.importance)
    return MessageOut(message_id=message_id, memory_ids=ids)


@app.get("/users/{user_id}/memories/search")
def search_memories(
    user_id: uuid.UUID,
    conn: Conn,
    embedder: Emb,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    mode: Literal["hybrid", "vector", "fts"] = "hybrid",
    k: Annotated[int, Query(ge=1, le=50)] = 5,
) -> SearchOut:
    with logs.timed("search", mode=mode, k=k) as log:
        with user_session(conn, user_id):
            ids = search(conn, embedder, q, mode=mode, limit=k)
            rows = conn.execute(
                "SELECT id, text, source_id FROM memories WHERE id = ANY(%s)", (ids,)
            ).fetchall()
        log["hits"] = len(ids)
    by_id = {r[0]: r for r in rows}
    if any(i not in by_id for i in ids):  # should be impossible inside one transaction
        raise HTTPException(500, "search returned ids that are not readable")
    return SearchOut(
        query=q, mode=mode, results=[Hit(id=i, text=by_id[i][1], source_id=by_id[i][2]) for i in ids]
    )
