"""HTTP API. Run: uvicorn memory_service.api:app --port 8000"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import logs
from .db import connect, user_session
from .embed import Embedder, FastEmbedder
from .search import add_memory, search

logs.setup()
app = FastAPI(title="memory-service")


def get_embedder() -> Embedder:
    return FastEmbedder()  # tests override this with FakeEmbedder


def get_conn() -> Iterator[psycopg.Connection]:
    conn = connect()  # app_user, RLS on
    try:
        yield conn
    finally:
        conn.close()


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


class Hit(BaseModel):
    id: str
    text: str


class SearchOut(BaseModel):
    query: str
    mode: str
    results: list[Hit]


@app.get("/health")
def health(conn: Conn) -> dict[str, str]:
    conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/users/{user_id}/memories", status_code=201)
def create_memory(user_id: uuid.UUID, body: MemoryIn, conn: Conn, embedder: Emb) -> MemoryOut:
    # NOTE: user_id comes from the path for now; with real auth it must come from the token instead.
    with user_session(conn, user_id):
        memory_id = add_memory(conn, embedder, str(user_id), body.text, body.importance)
    return MemoryOut(id=memory_id)


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
            texts = dict(conn.execute("SELECT id, text FROM memories WHERE id = ANY(%s)", (ids,)).fetchall())
        log["hits"] = len(ids)
    missing = [i for i in ids if i not in texts]
    if missing:  # should be impossible inside one transaction
        raise HTTPException(500, "search returned ids that are not readable")
    return SearchOut(query=q, mode=mode, results=[Hit(id=i, text=texts[i]) for i in ids])
