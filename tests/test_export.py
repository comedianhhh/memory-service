"""S3 export and Secrets Manager config, against moto's in-process AWS (no LocalStack, no network),
so these run in CI too."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest
from moto import mock_aws

from memory_service import aws
from memory_service.aws import dsn_from_secret

from .conftest import requires_db

BUCKET = "test-exports"


@pytest.fixture
def s3_env(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)  # moto intercepts real-AWS endpoints only
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ca-central-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("EXPORT_BUCKET", BUCKET)
    aws.client.cache_clear()  # clients made outside the mock would talk to the real endpoint
    with mock_aws():
        aws.client("s3").create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "ca-central-1"}
        )
        yield aws.client("s3")
    aws.client.cache_clear()


@requires_db
def test_export_contains_only_my_memories_and_link_expires_in_15_min(client, account, s3_env):
    me, other = account(), account()
    for t in ("I am allergic to peanuts", "My sister lives in Vancouver"):
        client.post("/me/memories", headers=me["headers"], json={"text": t})
    client.post("/me/memories", headers=other["headers"], json={"text": "someone else's secret"})

    r = client.post("/me/export", headers=me["headers"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["count"] == 2 and out["expires_in"] == 900

    q = parse_qs(urlparse(out["url"]).query)
    assert q["X-Amz-Expires"] == ["900"] and "X-Amz-Signature" in q

    key = urlparse(out["url"]).path.split(f"/{BUCKET}/", 1)[1]
    assert key.startswith(f"exports/{me['user_id']}/")
    obj = s3_env.get_object(Bucket=BUCKET, Key=key)
    assert obj["ServerSideEncryption"] == "AES256"
    doc = json.loads(obj["Body"].read())
    assert doc["user_id"] == me["user_id"]
    assert sorted(m["text"] for m in doc["memories"]) == ["I am allergic to peanuts", "My sister lives in Vancouver"]


@requires_db
def test_export_without_bucket_is_503(client, account, monkeypatch):
    monkeypatch.delenv("EXPORT_BUCKET", raising=False)
    assert client.post("/me/export", headers=account()["headers"]).status_code == 503


@requires_db
def test_export_needs_a_token(client):
    assert client.post("/me/export").status_code == 401


def test_secret_lookup(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ca-central-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    aws.client.cache_clear()
    with mock_aws():
        aws.client("secretsmanager").create_secret(Name="memory/app-dsn", SecretString="postgresql://u:p@h:5432/memory")
        assert aws.get_secret("memory/app-dsn") == "postgresql://u:p@h:5432/memory"
    aws.client.cache_clear()


def test_dsn_from_plain_string_and_rds_json():
    assert dsn_from_secret("postgresql://u:p@h/db") == "postgresql://u:p@h/db"
    rds = json.dumps({"username": "app", "password": "p@ss/w:rd", "host": "x.rds.amazonaws.com", "port": 5432, "dbname": "memory"})
    # special characters in the password are percent-encoded, or they'd break the URL
    assert dsn_from_secret(rds) == "postgresql://app:p%40ss%2Fw%3Ard@x.rds.amazonaws.com:5432/memory"
