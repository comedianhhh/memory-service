"""HTTP API. Run: uvicorn memory_service.api:app --port 8000

Who you are comes only from the access token. There is no user id in any URL, so there is nothing
to tamper with: the token's `sub` goes into SET LOCAL app.user_id and RLS does the rest.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import auth, logs
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


@app.exception_handler(auth.AuthError)
async def auth_error(_: Request, exc: auth.AuthError) -> JSONResponse:
    # the code tells a client whether refreshing can help (token_expired) or it must log in again
    return JSONResponse({"error": exc.code}, status_code=401, headers={"WWW-Authenticate": "Bearer"})


# ---- dependencies ---------------------------------------------------------------------------------


def get_embedder() -> Embedder:
    return FastEmbedder()  # tests override with FakeEmbedder


def get_now() -> datetime:
    return datetime.now(UTC)  # tests override to move time forward


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    with request.app.state.pool.connection(timeout=5) as conn:  # app_user, RLS on; back to the pool afterwards
        yield conn


Conn = Annotated[psycopg.Connection, Depends(get_conn)]
Emb = Annotated[Embedder, Depends(get_embedder)]
Now = Annotated[datetime, Depends(get_now)]
_bearer = HTTPBearer(auto_error=False)


def current_principal(
    conn: Conn, now: Now, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> auth.Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise auth.AuthError("missing_token")
    return auth.authenticate(conn, creds.credentials, now)


Me = Annotated[auth.Principal, Depends(current_principal)]


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


# ---- models ---------------------------------------------------------------------------------------


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$")
    password: str = Field(min_length=8, max_length=256)


class RefreshIn(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=200)


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    session_expires_at: datetime


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


def _tokens(pair: auth.TokenPair, now: datetime) -> TokenOut:
    return TokenOut(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=int((pair.access_expires_at - now).total_seconds()),
        session_expires_at=pair.session_expires_at,
    )


# ---- routes ---------------------------------------------------------------------------------------


@app.get("/health")
def health(request: Request, conn: Conn) -> dict[str, object]:
    conn.execute("SELECT 1")
    s = request.app.state.pool.get_stats()
    return {"status": "ok", "pool": {"size": s.get("pool_size"), "available": s.get("pool_available")}}


@app.post("/auth/register", status_code=201)
def register(body: Credentials, conn: Conn) -> dict[str, str]:
    try:
        return {"user_id": auth.register(conn, body.email, body.password)}
    except auth.EmailTaken:
        raise HTTPException(409, "email already registered") from None


@app.post("/auth/login")
def login(body: Credentials, conn: Conn, now: Now) -> TokenOut:
    return _tokens(auth.login(conn, body.email, body.password, now), now)


@app.post("/auth/refresh")
def refresh(body: RefreshIn, conn: Conn, now: Now) -> TokenOut:
    return _tokens(auth.refresh(conn, body.refresh_token, now), now)


@app.post("/auth/logout", status_code=204)
def logout(me: Me, conn: Conn, now: Now) -> None:
    auth.logout(conn, me.session_id, now)


@app.get("/me")
def whoami(me: Me) -> dict[str, str]:
    return {"user_id": me.user_id, "session_id": me.session_id}


@app.post("/me/memories", status_code=201)
def create_memory(body: MemoryIn, me: Me, conn: Conn, embedder: Emb) -> MemoryOut:
    with user_session(conn, me.user_id):
        memory_id = add_memory(conn, embedder, me.user_id, body.text, body.importance)
    return MemoryOut(id=memory_id)


@app.post("/me/messages", status_code=201)
def create_message(body: MessageIn, me: Me, conn: Conn, embedder: Emb) -> MessageOut:
    """A raw chat message of any length: chunked, embedded, stored as several memories."""
    with user_session(conn, me.user_id):
        message_id, ids = add_message(conn, embedder, me.user_id, body.text, body.importance)
    return MessageOut(message_id=message_id, memory_ids=ids)


@app.get("/me/memories/search")
def search_memories(
    me: Me,
    conn: Conn,
    embedder: Emb,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    mode: Literal["hybrid", "vector", "fts"] = "hybrid",
    k: Annotated[int, Query(ge=1, le=50)] = 5,
) -> SearchOut:
    with logs.timed("search", mode=mode, k=k) as log:
        with user_session(conn, me.user_id):
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
