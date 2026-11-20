"""Pure helpers for point IDs, hashes and the shared payload.

``point_id`` and ``base_payload`` follow the shared architecture plan and
match the pipeline's other ingesters; keep them in step.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

TIER = "member"
ASSET_SOURCE = "asset"
MANUAL_SOURCE = "manual"
# Shared architecture: 5 policy/sign, 4 wiki, 3 asset/geowiki/event,
# 2 manual/repo/esphome, 1 slack fact.
AUTHORITY = {ASSET_SOURCE: 3, MANUAL_SOURCE: 2}


def point_id(source: str, doc_id: str, chunk_index: int) -> str:
    """uuid5(source:doc_id:chunk_index): stable, so a re-sync overwrites."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{doc_id}:{chunk_index}"))


def text_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# Any: JSON-ready values.
def json_hash(value: Any) -> str:
    return text_hash(json.dumps(value, sort_keys=True, default=str))


# Payload fields that change on every run without the point changing.
_VOLATILE = frozenset({"indexed_at", "sync_run_id", "payload_hash"})


# Any: payload values are JSON.
def payload_hash(payload: dict[str, Any]) -> str:
    """Hash of everything in a payload but the per-run fields."""
    return json_hash({k: v for k, v in payload.items() if k not in _VOLATILE})


def now_rfc3339() -> str:
    return datetime.now(UTC).isoformat()


def base_payload(
    *,
    source: str,
    doc_id: str,
    chunk_index: int,
    title: str,
    text: str,
    context: str,
    url: str | None,
    run_id: str,
    created_at: str | None,
    updated_at: str | None,
    heading_path: list[str] | None = None,
    page: int | None = None,
    lang: str | None = None,
    topics: list[str] | None = None,
    links: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """The payload every text collection shares (see the shared architecture plan)."""
    links = links or {}
    return {
        "source": source,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "title": title,
        "heading_path": heading_path or [],
        "text": text,
        "context": context,
        "url": url,
        "tier": TIER,
        "authority": AUTHORITY[source],
        "created_at": created_at,
        "updated_at": updated_at,
        "indexed_at": now_rfc3339(),
        "stale_after": None,
        "superseded_by": None,
        "content_hash": text_hash(context, text),
        "sync_run_id": run_id,
        "asset_ids": links.get("asset_ids", []),
        "location_ids": links.get("location_ids", []),
        "work_ids": links.get("work_ids", []),
        "event_ids": links.get("event_ids", []),
        "training_ids": links.get("training_ids", []),
        "topics": topics or [],
        "section": None,
        "page": page,
        "lang": lang,
    }
