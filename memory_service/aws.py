"""AWS clients (boto3).

No endpoint is hard-coded: boto3 reads AWS_ENDPOINT_URL itself. So the same code talks to LocalStack
when that variable points at it, and to real AWS when it is unset. Region and credentials also come
from the standard AWS_* variables (or, on ECS, from the task role), never from this file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from urllib.parse import quote

import boto3
from botocore.config import Config

_CONFIG = Config(
    # Path style puts the bucket in the URL path (host/bucket/key) instead of a subdomain
    # (bucket.host/key), so it works with any local hostname, not just *.amazonaws.com.
    s3={"addressing_style": "path"},
    # Standard retry mode: exponential backoff with jitter on throttling and 5xx errors.
    retries={"mode": "standard", "max_attempts": 3},
)


@lru_cache(maxsize=None)
def client(service: str):
    # boto3 clients are thread-safe and costly to build, so make one per service per process.
    return boto3.client(service, config=_CONFIG)


def get_secret(secret_id: str) -> str:
    return client("secretsmanager").get_secret_value(SecretId=secret_id)["SecretString"]


def dsn_from_secret(secret: str) -> str:
    """Accept either a plain DSN string, or the JSON that RDS-managed secrets use:
    {"username", "password", "host", "port", "dbname"}."""
    if not secret.lstrip().startswith("{"):
        return secret
    s = json.loads(secret)
    user, pw = quote(s["username"], safe=""), quote(s["password"], safe="")
    return f"postgresql://{user}:{pw}@{s['host']}:{s.get('port', 5432)}/{s.get('dbname', 'memory')}"
