"""Load data/memories.jsonl for the demo user (wipes that user's rows first)."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_service.db import connect  # noqa: E402
from memory_service.embed import embed_passages, to_pgvector  # noqa: E402

DEMO_USER = "00000000-0000-0000-0000-000000000001"
DATA = ROOT / "data" / "memories.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = embed_passages([r["text"] for r in rows])
    with connect() as conn:
        conn.execute("DELETE FROM memories WHERE user_id = %s", (DEMO_USER,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO memories (id, user_id, text, embedding) VALUES (%s, %s, %s, %s::vector)",
                [(r["id"], DEMO_USER, r["text"], to_pgvector(v)) for r, v in zip(rows, vectors, strict=True)],
            )
    print(f"seeded {len(rows)} memories for {DEMO_USER}")


if __name__ == "__main__":
    main()
