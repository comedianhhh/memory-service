"""Password login, short-lived access tokens, rotating refresh tokens, session expiry.

Three clocks:
  access token   15 min   JWT, checked on every request (signature + exp + session still active)
  refresh token  7 days   opaque, single use; each refresh returns a new one   (idle timeout)
  session        30 days  absolute; refresh can never extend past it           (hard timeout)

Why a session row at all when JWTs are "stateless": without it, logout can't take effect until the
access token expires, and a stolen refresh token can't be cut off. One indexed lookup per request
buys both.

Every function takes `now` so tests can move time forward instead of sleeping.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import jwt
import psycopg
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from .logs import logger

ACCESS_TTL = timedelta(minutes=15)
REFRESH_TTL = timedelta(days=7)
SESSION_TTL = timedelta(days=30)
ISSUER = "memory-service"
ALGORITHM = "HS256"

_DEV_SECRET = "dev-only-secret-change-me-dev-only-secret"  # 32+ bytes so HS256 isn't weak
JWT_SECRET = os.environ.get("JWT_SECRET", _DEV_SECRET)
if JWT_SECRET == _DEV_SECRET:
    logger.warning("jwt", extra={"fields": {"event": "config", "warning": "JWT_SECRET not set, using dev secret"}})

_hasher = PasswordHasher()  # argon2id with library defaults
# verify against this when the email doesn't exist, so "no such user" takes as long as "wrong password"
_DUMMY_HASH = _hasher.hash("not-a-real-password")


class AuthError(Exception):
    """Maps to HTTP 401. `code` goes to the client so it knows whether refreshing can help."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class EmailTaken(Exception):
    pass


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    session_expires_at: datetime


@dataclass(frozen=True)
class Principal:
    user_id: str
    session_id: str


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _sha256(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


def register(conn: psycopg.Connection, email: str, password: str) -> str:
    try:
        with conn.transaction():
            row = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                (normalize_email(email), _hasher.hash(password)),
            ).fetchone()
    except psycopg.errors.UniqueViolation as exc:
        raise EmailTaken from exc
    return str(row[0])


def _access_token(user_id: str, session_id: str, now: datetime) -> tuple[str, datetime]:
    exp = now + ACCESS_TTL
    claims = {
        "iss": ISSUER,
        "sub": user_id,
        "sid": session_id,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, JWT_SECRET, algorithm=ALGORITHM), exp


def _new_refresh(conn: psycopg.Connection, session_id: str, session_expires: datetime, now: datetime) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO refresh_tokens (token_hash, session_id, issued_at, expires_at) VALUES (%s, %s, %s, %s)",
        (_sha256(token), session_id, now, min(now + REFRESH_TTL, session_expires)),
    )
    return token


def login(conn: psycopg.Connection, email: str, password: str, now: datetime) -> TokenPair:
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE email = %s", (normalize_email(email),)
    ).fetchone()
    try:
        _hasher.verify(row[1] if row else _DUMMY_HASH, password)
    except (VerifyMismatchError, InvalidHashError):
        raise AuthError("invalid_credentials") from None
    if row is None:
        raise AuthError("invalid_credentials")
    user_id = str(row[0])
    session_expires = now + SESSION_TTL
    with conn.transaction():
        session_id = str(
            conn.execute(
                "INSERT INTO sessions (user_id, created_at, expires_at) VALUES (%s, %s, %s) RETURNING id",
                (user_id, now, session_expires),
            ).fetchone()[0]
        )
        refresh = _new_refresh(conn, session_id, session_expires, now)
    access, access_exp = _access_token(user_id, session_id, now)
    logger.info("auth", extra={"fields": {"event": "auth.login", "user_id": user_id, "session_id": session_id}})
    return TokenPair(access, refresh, access_exp, session_expires)


def _revoke(conn: psycopg.Connection, session_id: str, reason: str, now: datetime) -> None:
    conn.execute(
        "UPDATE sessions SET revoked_at = %s, revoke_reason = %s WHERE id = %s AND revoked_at IS NULL",
        (now, reason, session_id),
    )


def refresh(conn: psycopg.Connection, refresh_token: str, now: datetime) -> TokenPair:
    """Rotate: the presented token is burned, a new one comes back. Atomic, so two concurrent
    refreshes with the same token can't both win."""
    h = _sha256(refresh_token)
    # Errors are decided inside the transaction but raised after it commits: raising inside
    # `with conn.transaction()` rolls back, which would also undo the reuse-detection revoke.
    error: str | None = None
    with conn.transaction():
        row = conn.execute(
            """
            UPDATE refresh_tokens rt SET used_at = %s
            FROM sessions s
            WHERE rt.token_hash = %s AND rt.used_at IS NULL AND s.id = rt.session_id
            RETURNING rt.session_id, rt.expires_at, s.user_id, s.expires_at, s.revoked_at
            """,
            (now, h),
        ).fetchone()
        if row is None:
            seen = conn.execute("SELECT session_id FROM refresh_tokens WHERE token_hash = %s", (h,)).fetchone()
            if seen:
                # A used token came back: either the client replayed it or someone stole it.
                # We can't tell which copy is the attacker, so the whole session dies.
                _revoke(conn, str(seen[0]), "refresh_token_reuse", now)
                logger.warning("auth", extra={"fields": {"event": "auth.reuse_detected", "session_id": str(seen[0])}})
                error = "refresh_token_reused"
            else:
                error = "invalid_refresh_token"
        else:
            session_id, token_exp, user_id, session_exp, revoked_at = row
            session_id, user_id = str(session_id), str(user_id)
            if revoked_at is not None:
                error = "session_revoked"
            elif now >= session_exp:
                error = "session_expired"
            elif now >= token_exp:
                error = "refresh_token_expired"
            else:
                new_refresh = _new_refresh(conn, session_id, session_exp, now)
    if error:
        raise AuthError(error)  # the burned token (used_at) stays burned too
    access, access_exp = _access_token(user_id, session_id, now)
    return TokenPair(access, new_refresh, access_exp, session_exp)


def logout(conn: psycopg.Connection, session_id: str, now: datetime) -> None:
    with conn.transaction():
        _revoke(conn, session_id, "logout", now)


def authenticate(conn: psycopg.Connection, access_token: str, now: datetime) -> Principal:
    try:
        # exp is checked below against the injected clock, not the wall clock
        claims = jwt.decode(
            access_token,
            JWT_SECRET,
            algorithms=[ALGORITHM],  # pinned: never accept "none" or a different alg from the header
            issuer=ISSUER,
            options={"verify_exp": False, "require": ["exp", "sub", "sid", "iss"]},
        )
    except jwt.PyJWTError:
        raise AuthError("invalid_token") from None
    if now.timestamp() >= claims["exp"]:
        raise AuthError("token_expired")
    row = conn.execute(
        "SELECT user_id, expires_at, revoked_at FROM sessions WHERE id = %s", (claims["sid"],)
    ).fetchone()
    if row is None or str(row[0]) != claims["sub"]:
        raise AuthError("invalid_token")
    if row[2] is not None:
        raise AuthError("session_revoked")
    if now >= row[1]:
        raise AuthError("session_expired")
    return Principal(user_id=claims["sub"], session_id=claims["sid"])
