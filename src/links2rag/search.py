"""Hybrid search of hs_member: dense and sparse results fused with reciprocal rank fusion.

Used by `links2rag query` and by the nickname pass. The retrieval service
does the same before reranking; when a query names an asset it also searches
that asset's manuals with a filter on ``asset_ids``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from links2rag.embeddings import Embedder
from links2rag.qdrant_store import DENSE, SPARSE
from links2rag.sparse import sparse_vector


def search(
    client: QdrantClient,
    collection: str,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 5,
    sources: Sequence[str] = (),
    asset_id: str | None = None,
) -> list[qm.ScoredPoint]:
    must: list[qm.Condition] = []
    if sources:
        must.append(qm.FieldCondition(key="source", match=qm.MatchAny(any=list(sources))))
    if asset_id:
        must.append(qm.FieldCondition(key="asset_ids", match=qm.MatchValue(value=asset_id)))
    selector = qm.Filter(must=must) if must else None
    dense = embedder.encode([query], kind="query")[0]
    prefetch = max(limit * 3, 20)
    # Any: query_points returns a typed response; the points are what we want.
    response: Any = client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE, filter=selector, limit=prefetch),
            qm.Prefetch(query=sparse_vector(query), using=SPARSE, filter=selector, limit=prefetch),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return list(response.points)
