"""The `report`, `aliases` and `query` commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient

from links2rag import reports
from links2rag.config import Settings
from links2rag.db import Database
from links2rag.embeddings import build_embedder
from links2rag.search import search


class CommandError(ValueError):
    """A command was given arguments it can't use."""


# Any: command output is JSON.
def run_command(
    settings: Settings,
    command: str,
    *,
    text: str = "",
    approve: int | None = None,
    reject: int | None = None,
    limit: int = 5,
    asset: str | None = None,
    sources: Sequence[str] = ("asset", "manual"),
) -> Any:
    if command == "query":
        if not text.strip():
            raise CommandError("query needs search text")
        return query(settings, text, limit=limit, asset=asset, sources=sources)
    with Database.connect(settings.database_url) as db:
        if command == "report":
            report = reports.build(db)
            reports.write(report, settings.data_dir)
            return report
        if approve is not None and reject is not None:
            raise CommandError("use --approve or --reject, not both")
        if approve is not None or reject is not None:
            queue_id = approve if approve is not None else reject
            row = db.resolve_alias(int(queue_id), approve=approve is not None)  # type: ignore[arg-type]
            if row is None:
                raise CommandError(f"no proposed alias with id {queue_id}")
            return {
                "id": row["id"],
                "asset_id": row["left_id"],
                "alias": row["right_id"],
                "decision": row["details"].get("decision"),
            }
        return [
            {"id": r["id"], "asset_id": r["left_id"], "alias": r["right_id"], **r["details"]}
            for r in db.alias_queue()
        ]


# Any: the results are JSON.
def query(
    settings: Settings,
    text: str,
    *,
    limit: int,
    asset: str | None,
    sources: Sequence[str],
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    # The embedder first: if it can't be built, no client is left open.
    embedder = build_embedder(settings)
    qdrant = client or QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    try:
        hits = search(
            qdrant,
            settings.member_collection,
            embedder,
            text,
            limit=limit,
            sources=sources,
            asset_id=asset,
        )
    finally:
        closer = getattr(embedder, "close", None)
        if closer is not None:
            closer()
        if client is None:
            qdrant.close()
    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "score": round(hit.score, 4),
                "source": payload.get("source"),
                "title": payload.get("title"),
                "heading_path": payload.get("heading_path"),
                "page": payload.get("page"),
                "url": payload.get("url"),
                "asset_ids": payload.get("asset_ids"),
                "manual_section": payload.get("manual_section"),
                "text": payload.get("text"),
            }
        )
    return results
