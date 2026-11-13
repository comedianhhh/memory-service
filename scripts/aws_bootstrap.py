"""Create the AWS resources the api expects: a private, encrypted export bucket with a 1-day expiry
rule, and the app's DSN in Secrets Manager. Idempotent, safe to run again.

On real AWS this is Terraform's job; here it sets up LocalStack:
    AWS_ENDPOINT_URL=http://localhost:4566 .venv/Scripts/python scripts/aws_bootstrap.py
(Defaults below point at LocalStack, so plain `python scripts/aws_bootstrap.py` does the same.)
"""

from __future__ import annotations

import os

os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:4566")
os.environ.setdefault("AWS_DEFAULT_REGION", "ca-central-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402

BUCKET = os.environ.get("EXPORT_BUCKET", "memory-exports")
SECRET_ID = os.environ.get("MEMORY_DSN_SECRET_ID", "memory/app-dsn")
# The DSN as the api container sees it (service name db, container port 5432). Local-only password.
APP_DSN = os.environ.get("BOOTSTRAP_APP_DSN", "postgresql://app_user:app_user@db:5432/memory")


def bucket(s3) -> None:
    region = os.environ["AWS_DEFAULT_REGION"]
    try:
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": region})
        print(f"created bucket {BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"bucket {BUCKET} exists")
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=BUCKET,
        ServerSideEncryptionConfiguration={"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
    )
    # Exports are personal data with a 15-minute link; nothing needs them after a day.
    s3.put_bucket_lifecycle_configuration(
        Bucket=BUCKET,
        LifecycleConfiguration={
            "Rules": [{"ID": "expire-exports", "Status": "Enabled", "Filter": {"Prefix": "exports/"}, "Expiration": {"Days": 1}}]
        },
    )


def secret(sm) -> None:
    try:
        sm.create_secret(Name=SECRET_ID, SecretString=APP_DSN)
        print(f"created secret {SECRET_ID}")
    except sm.exceptions.ResourceExistsException:
        sm.put_secret_value(SecretId=SECRET_ID, SecretString=APP_DSN)
        print(f"updated secret {SECRET_ID}")


if __name__ == "__main__":
    bucket(boto3.client("s3", config=Config(s3={"addressing_style": "path"})))
    secret(boto3.client("secretsmanager"))
