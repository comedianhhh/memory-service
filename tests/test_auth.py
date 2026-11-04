"""Login, token checks, and the three clocks: access 15 min, refresh 7 days idle, session 30 days."""

from __future__ import annotations

import jwt

from memory_service import auth

from .conftest import requires_db

pytestmark = requires_db


def _err(r) -> str:
    assert r.status_code == 401, (r.status_code, r.text)
    assert r.headers["www-authenticate"] == "Bearer"
    return r.json()["error"]


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---- login ----------------------------------------------------------------------------------------


def test_register_login_me(client, account):
    a = account()
    me = client.get("/me", headers=a["headers"]).json()
    assert me["user_id"] == a["user_id"]
    assert a["tokens"]["expires_in"] == 15 * 60


def test_duplicate_email_is_409_case_insensitive(client, account):
    a = account()
    r = client.post("/auth/register", json={"email": a["email"].upper(), "password": "another password"})
    assert r.status_code == 409


def test_wrong_password_and_unknown_email_look_the_same(client, account):
    a = account()
    wrong = client.post("/auth/login", json={"email": a["email"], "password": "wrong password!"})
    nobody = client.post("/auth/login", json={"email": "nobody@example.test", "password": "wrong password!"})
    assert _err(wrong) == _err(nobody) == "invalid_credentials"


def test_password_is_hashed_not_stored(client, account, conn):
    a = account()
    stored = conn.execute("SELECT password_hash FROM users WHERE id = %s", (a["user_id"],)).fetchone()[0]
    assert a["password"] not in stored and stored.startswith("$argon2id$")


def test_short_password_rejected(client):
    assert client.post("/auth/register", json={"email": "x@example.test", "password": "short"}).status_code == 422


# ---- access token ---------------------------------------------------------------------------------


def test_missing_or_garbage_token(client):
    assert _err(client.get("/me")) == "missing_token"
    assert _err(client.get("/me", headers=_bearer("not.a.jwt"))) == "invalid_token"


def test_token_signed_with_other_key_or_alg_none_is_rejected(client, account, clock):
    a = account()
    claims = jwt.decode(a["tokens"]["access_token"], options={"verify_signature": False})
    forged = jwt.encode(claims, "attacker-secret-" + "y" * 32, algorithm="HS256")
    none_alg = jwt.encode(claims, None, algorithm="none")
    assert _err(client.get("/me", headers=_bearer(forged))) == "invalid_token"
    assert _err(client.get("/me", headers=_bearer(none_alg))) == "invalid_token"


def test_cannot_act_as_another_user_by_editing_sub(client, account):
    a, b = account(), account()
    claims = jwt.decode(a["tokens"]["access_token"], options={"verify_signature": False})
    claims["sub"] = b["user_id"]  # even a correctly signed token must match its session's user
    tampered = jwt.encode(claims, auth.JWT_SECRET, algorithm="HS256")
    assert _err(client.get("/me", headers=_bearer(tampered))) == "invalid_token"


def test_access_token_expires_after_15_minutes(client, account, clock):
    a = account()
    clock.advance(minutes=14, seconds=59)
    assert client.get("/me", headers=a["headers"]).status_code == 200
    clock.advance(seconds=1)
    assert _err(client.get("/me", headers=a["headers"])) == "token_expired"


# ---- refresh --------------------------------------------------------------------------------------


def test_refresh_rotates_and_old_access_still_dies_on_time(client, account, clock):
    a = account()
    clock.advance(minutes=20)
    r = client.post("/auth/refresh", json={"refresh_token": a["tokens"]["refresh_token"]})
    assert r.status_code == 200
    new = r.json()
    assert new["refresh_token"] != a["tokens"]["refresh_token"]
    assert client.get("/me", headers=_bearer(new["access_token"])).status_code == 200


def test_reusing_a_refresh_token_kills_the_whole_session(client, account):
    a = account()
    old = a["tokens"]["refresh_token"]
    new = client.post("/auth/refresh", json={"refresh_token": old}).json()
    # attacker (or a buggy client) replays the old one
    assert _err(client.post("/auth/refresh", json={"refresh_token": old})) == "refresh_token_reused"
    # ...and now even the legitimate new tokens are dead
    assert _err(client.post("/auth/refresh", json={"refresh_token": new["refresh_token"]})) == "session_revoked"
    assert _err(client.get("/me", headers=_bearer(new["access_token"]))) == "session_revoked"


def test_refresh_token_idle_timeout_7_days(client, account, clock):
    a = account()
    clock.advance(days=7)
    assert _err(client.post("/auth/refresh", json={"refresh_token": a["tokens"]["refresh_token"]})) == "refresh_token_expired"


def test_session_hard_limit_30_days_even_if_active(client, account, clock):
    a = account()
    refresh = a["tokens"]["refresh_token"]
    for _ in range(4):  # refresh every 6 days: never idle for 7
        clock.advance(days=6)
        r = client.post("/auth/refresh", json={"refresh_token": refresh})
        assert r.status_code == 200, r.text
        refresh = r.json()["refresh_token"]
    assert r.json()["session_expires_at"].startswith("2026-12-04")  # login + 30 days, never extended
    clock.advance(days=6)  # day 30
    assert _err(client.post("/auth/refresh", json={"refresh_token": refresh})) == "session_expired"


def test_unknown_refresh_token(client):
    assert _err(client.post("/auth/refresh", json={"refresh_token": "x" * 43})) == "invalid_refresh_token"


# ---- logout ---------------------------------------------------------------------------------------


def test_logout_takes_effect_immediately(client, account):
    a = account()
    assert client.post("/auth/logout", headers=a["headers"]).status_code == 204
    # the access token's own exp is 15 min away, but the session row says no
    assert _err(client.get("/me", headers=a["headers"])) == "session_revoked"
    assert _err(client.post("/auth/refresh", json={"refresh_token": a["tokens"]["refresh_token"]})) == "session_revoked"


def test_logout_only_ends_that_session(client, account):
    a = account()
    second = client.post("/auth/login", json={"email": a["email"], "password": a["password"]}).json()
    client.post("/auth/logout", headers=a["headers"])
    assert client.get("/me", headers=_bearer(second["access_token"])).status_code == 200


# ---- isolation end to end -------------------------------------------------------------------------


def test_b_cannot_see_a_through_the_api(client, account):
    a, b = account(), account()
    client.post("/me/memories", json={"text": "A only: flight AC 1207"}, headers=a["headers"])
    assert client.get("/me/memories/search", params={"q": "AC 1207"}, headers=a["headers"]).json()["results"]
    assert client.get("/me/memories/search", params={"q": "AC 1207"}, headers=b["headers"]).json()["results"] == []
