"""Export one user's memories to S3 and hand back a short-lived download link.

The bucket stays private. The presigned URL carries an HMAC-SHA256 signature over the method, bucket,
key, signing time and expiry, made with the server's AWS key. Whoever holds the link can GET that one
object until it expires. Changing any part of the link breaks the signature (403), and the key itself
never leaves the server.
"""

from __future__ import annotations

import json
from datetime import datetime

import psycopg

from .aws import client
from .logs import timed

EXPORT_TTL_S = 15 * 60


def read_memories(conn: psycopg.Connection) -> list[dict]:
    """Call inside user_session: there is no WHERE user_id, RLS limits the rows to that user."""
    rows = conn.execute(
        "SELECT id, text, importance, created_at, source_id FROM memories ORDER BY created_at, id"
    ).fetchall()
    return [
        {"id": str(r[0]), "text": r[1], "importance": r[2], "created_at": r[3].isoformat(), "source_id": r[4]}
        for r in rows
    ]


def upload_export(bucket: str, user_id: str, memories: list[dict], now: datetime) -> tuple[str, str]:
    """-> (key, presigned GET url). Runs after the DB transaction has closed, so a slow S3 call
    never holds a database connection open."""
    key = f"exports/{user_id}/{now:%Y%m%dT%H%M%SZ}.json"
    body = json.dumps(
        {"user_id": user_id, "exported_at": now.isoformat(), "count": len(memories), "memories": memories},
        ensure_ascii=False,
    ).encode()
    s3 = client("s3")
    with timed("export", n=len(memories), bytes=len(body)):
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ServerSideEncryption="AES256",  # encrypted at rest with S3-managed keys
        )
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=EXPORT_TTL_S
        )
    return key, url
