from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from links2rag.qdrant_store import (
    COMMON_INDEXES,
    LINKS_INDEXES,
    CollectionMismatchError,
    QdrantStore,
    _distance,
)
from links2rag.sparse import sparse_vector

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
META = {"embedding_model": "m1"}


def _store(client: QdrantClient, **kwargs: Any) -> QdrantStore:
    args: dict[str, Any] = {
        "collection": "c",
        "vector_size": 2,
        "distance": "cosine",
        "metadata": META,
    }
    args.update(kwargs)
    return QdrantStore(client=client, **args)


def _point(n: int, source: str = "calibre", run: str = "r1", doc: str = "d") -> qm.PointStruct:
    return qm.PointStruct(
        id=f"00000000-0000-0000-0000-{n:012d}",
        vector={"dense": [1.0, float(n)], "sparse": sparse_vector(f"word{n}")},
        payload={"source": source, "doc_id": doc, "sync_run_id": run, "content_hash": f"h{n}"},
    )


def test_creates_hybrid_collection_with_metadata() -> None:
    client = QdrantClient(location=":memory:")
    _store(client)
    info = client.get_collection("c")
    assert info.config.metadata == META
    assert isinstance(info.config.params.vectors, dict)
    assert info.config.params.vectors["dense"].size == 2
    assert info.config.params.sparse_vectors is not None
    assert info.config.params.sparse_vectors["sparse"].modifier == qm.Modifier.IDF
    _store(client)  # reopening an existing matching collection is fine


def test_mismatches_are_errors() -> None:
    client = QdrantClient(location=":memory:")
    _store(client)
    with pytest.raises(CollectionMismatchError, match="VECTOR_SIZE is 3"):
        _store(client, vector_size=3)
    with pytest.raises(CollectionMismatchError, match="was embedded with 'm1'"):
        _store(client, metadata={"embedding_model": "m2"})
    client.create_collection(
        "old", vectors_config=qm.VectorParams(size=2, distance=qm.Distance.COSINE)
    )
    with pytest.raises(CollectionMismatchError, match="without the 'dense'"):
        _store(client, collection="old")


def test_adds_metadata_to_a_collection_without_it() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "shared",
        vectors_config={"dense": qm.VectorParams(size=2, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )
    _store(client, collection="shared")
    assert client.get_collection("shared").config.metadata == META


def test_upsert_hashes_payloads_and_deletes() -> None:
    client = QdrantClient(location=":memory:")
    store = _store(client)
    store.upsert([])
    store.upsert([_point(1), _point(2, doc="e"), _point(3, source="wiki")])
    ids = [str(_point(n).id) for n in (1, 2, 9)]
    assert store.existing_hashes(ids) == {ids[0]: "h1", ids[1]: "h2"}

    store.set_payload([ids[0]], {"sync_run_id": "r2"})
    store.set_payload([], {"x": 1})
    store.set_payload_for_doc("calibre", "e", {"sync_run_id": "r2"})
    assert store.delete_stale("calibre", "r2") == 0
    store.set_payload([ids[0]], {"sync_run_id": "r3"})
    assert store.delete_stale("calibre", "r3") == 1
    assert store.count("calibre") == 1
    assert store.count() == 2

    store.delete_docs("calibre", [])
    store.delete_docs("calibre", ["d"])
    assert store.count("calibre") == 0
    store.delete_ids([])
    store.delete_ids([str(_point(3).id)])
    assert store.payloads() == []


def test_existing_state_and_replace_payloads() -> None:
    client = QdrantClient(location=":memory:")
    store = _store(client)
    store.upsert([_point(1)])
    pid = str(_point(1).id)
    assert store.existing_state([pid, str(_point(2).id)]) == {pid: ("h1", None)}
    store.replace_payloads(
        [(pid, {"source": "calibre", "content_hash": "h9", "payload_hash": "p"})]
    )
    assert store.existing_state([pid]) == {pid: ("h9", "p")}


def test_doc_point_ids() -> None:
    client = QdrantClient(location=":memory:")
    store = _store(client)
    assert store.doc_point_ids("calibre", []) == set()
    store.upsert([_point(n, doc="d") for n in range(1, 600)] + [_point(700, doc="e")])
    assert len(store.doc_point_ids("calibre", ["d"])) == 599
    assert store.doc_point_ids("wiki", ["d"]) == set()


def test_payloads_scrolls_everything() -> None:
    client = QdrantClient(location=":memory:")
    store = _store(client)
    store.upsert([_point(n) for n in range(1, 301)])
    assert len(store.payloads("calibre")) == 300


def test_indexes_are_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QdrantClient(location=":memory:")
    requested: list[str] = []
    monkeypatch.setattr(
        client,
        "create_payload_index",
        lambda collection_name, field_name, field_schema: requested.append(field_name),
    )
    _store(client, indexes=COMMON_INDEXES | LINKS_INDEXES)
    for name in (
        "source",
        "tier",
        "asset_ids",
        "location_ids",
        "topics",
        "updated_at",
        "section",
        "doc_id",
        "manual_section",
        "authority",
    ):
        assert name in requested


def test_distance() -> None:
    assert _distance("DOT") == qm.Distance.DOT
    assert _distance("euclid") == qm.Distance.EUCLID
    with pytest.raises(ValueError, match="Unsupported"):
        _distance("manhattan")
