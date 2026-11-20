from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from links2rag import pipeline as pipeline_module
from links2rag.config import Settings
from links2rag.db import Database
from links2rag.documents import point_id
from links2rag.llm import LLM, AssetAnalysis, LLMError
from links2rag.pipeline import Pipeline, load_taxonomy
from tests.fakes import HOSTNAME, IEEE, IP, OWNER, SERIAL, FakeEmbedder, FakeLinksClient, thing

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
SettingsFactory = Callable[..., Settings]


class FakeLLM:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []
        self.model = "fake"

    def analyze_asset(self, *, facts: str, taxonomy: list[str]) -> AssetAnalysis:
        self.calls.append(facts)
        if self.fail:
            raise LLMError("down")
        return AssetAnalysis(description="A laser cutter for members.", topics=["laser-cutting"])

    def confirm_link(self, *, text: str, alias: str, entity_type: str, entity_id: str) -> bool:
        if self.fail:
            raise LLMError("down")
        return entity_id == "geowiki:rooms/a"


def _pipeline(
    db: Database,
    settings: Settings,
    client: FakeLinksClient,
    *,
    qdrant: QdrantClient | None = None,
    embedder: FakeEmbedder | None = None,
    llm: Any = None,
    stop: threading.Event | None = None,
) -> Pipeline:
    return Pipeline(
        settings=settings,
        db=db,
        embedder=embedder or FakeEmbedder(),
        qdrant=qdrant or QdrantClient(location=":memory:"),
        client=client,
        llm=llm,
        stop=stop,
    )


def _payloads(p: Pipeline, source: str = "asset") -> list[dict[str, Any]]:
    return p.member.payloads(source)


def test_full_run_indexes_assets_without_private_fields(
    db: Database, make_settings: SettingsFactory
) -> None:
    client = FakeLinksClient(
        thing(owner=OWNER, notes=f"{OWNER} says: cut at 10.42.0.17? no, serial {SERIAL}"),
        thing("efgh5678", thing_id=2, name="Bandsaw", manufacturer="Rikon", model="10-305"),
    )
    p = _pipeline(db, make_settings(), client)
    result = p.run("full")
    assert result.status == "ok"
    assert result.counts.seen == 2
    assert result.counts.added == 2
    payloads = _payloads(p)
    assert {pl["doc_id"] for pl in payloads} == {"abcd1234", "efgh5678"}
    for pl in payloads:
        assert pl["tier"] == "member"
        assert pl["authority"] == 3
        assert pl["asset_ids"] == [pl["doc_id"]]
    dump = json.dumps(payloads)
    for canary in (SERIAL, OWNER, IP, HOSTNAME, IEEE):
        assert canary not in dump
    laser = next(pl for pl in payloads if pl["doc_id"] == "abcd1234")
    assert laser["location_ids"] == ["geowiki:resources/12"]
    assert laser["ownership"] == "member-loan"
    assert laser["url"] == "https://links.example.com/things/abcd1234"
    # Only the member collection exists: nothing goes to hs_public.
    assert [c.name for c in p.qdrant.get_collections().collections] == ["hs_member"]
    staged = db.staged_document("asset", "abcd1234")
    assert staged is not None
    assert staged["raw_content"]["serial_number"] == SERIAL  # staged, never indexed
    assert {r["asset_id"] for r in db.assets()} == {"abcd1234", "efgh5678"}
    aliases = {r["alias"] for r in db.aliases() if r["entity_id"] == "efgh5678"}
    assert "Rikon 10-305" in aliases


def test_unchanged_things_are_not_restaged_or_reembedded(
    db: Database, make_settings: SettingsFactory
) -> None:
    client = FakeLinksClient(thing())
    embedder = FakeEmbedder()
    p = _pipeline(db, make_settings(), client, embedder=embedder)
    p.run("full")
    first = len(embedder.texts)
    client.things["abcd1234"]["qr_scan_count"] = 50  # a scan is not a change
    result = p.run("incremental")
    assert result.counts.updated == 0
    assert len(embedder.texts) == first
    result = p.run("full")
    assert len(embedder.texts) == first
    assert p.member.count("asset") == 1
    # Forced: restaged and re-embedded.
    result = p.run("full", force=True)
    assert result.counts.updated == 1
    assert len(embedder.texts) == first * 2


def test_changes_and_deletions_propagate(db: Database, make_settings: SettingsFactory) -> None:
    client = FakeLinksClient(thing(), thing("efgh5678", thing_id=2, name="Bandsaw"))
    p = _pipeline(db, make_settings(), client)
    p.run("full")
    client.things["abcd1234"]["notes"] = "Now with a rotary attachment."
    del client.things["efgh5678"]
    result = p.run("incremental")
    assert result.counts.updated == 1
    assert result.counts.deleted == 1
    payloads = _payloads(p)
    assert [pl["doc_id"] for pl in payloads] == ["abcd1234"]
    assert "rotary attachment" in payloads[0]["text"]
    assert db.staged_document("asset", "efgh5678") is None
    assert [r["asset_id"] for r in db.assets()] == ["abcd1234"]
    assert all(r["entity_id"] != "efgh5678" for r in db.aliases())


def test_shrinking_record_drops_leftover_chunks(
    db: Database, make_settings: SettingsFactory
) -> None:
    client = FakeLinksClient(thing(description="Cuts things well. " * 120))
    p = _pipeline(db, make_settings(chunk_size=300, chunk_overlap=30), client)
    p.run("full")
    assert p.member.count("asset") > 2
    client.things["abcd1234"]["description"] = "Short now."
    p.run("incremental")
    assert p.member.count("asset") == 1


def test_safety_valve_blocks_deletes(db: Database, make_settings: SettingsFactory) -> None:
    things = [thing(f"key{n:05d}", thing_id=n, name=f"Thing {n}") for n in range(6)]
    client = FakeLinksClient(*things)
    p = _pipeline(db, make_settings(), client)
    p.run("full")
    for key in list(client.things)[:4]:
        del client.things[key]
    result = p.run("full")
    assert result.status == "valve"
    assert p.member.count("asset") == 6
    assert len(db.assets()) == 6


def test_integration_things_are_staged_not_indexed(
    db: Database, make_settings: SettingsFactory
) -> None:
    client = FakeLinksClient(
        thing(), thing("zig00001", thing_id=2, name="Door sensor", integration_source="zigbee2mqtt")
    )
    p = _pipeline(db, make_settings(), client)
    p.run("full")
    assert {pl["doc_id"] for pl in _payloads(p)} == {"abcd1234"}
    staged = db.staged_document("asset", "zig00001")
    assert staged is not None
    assert staged["indexable"] is False
    assert staged["flags"]["excluded"] == "imported by zigbee2mqtt"
    assert all(r["entity_id"] != "zig00001" for r in db.aliases())
    # Opting in indexes it.
    p2 = _pipeline(
        db, make_settings(include_integration_sources=("zigbee2mqtt",)), client, qdrant=p.qdrant
    )
    p2.run("full", force=True)
    assert {pl["doc_id"] for pl in _payloads(p2)} == {"abcd1234", "zig00001"}
    # And an asset that becomes excluded loses its points on the next run.
    p.run("incremental", force=True)
    assert {pl["doc_id"] for pl in _payloads(p)} == {"abcd1234"}


def test_secrets_are_redacted_and_quarantined(db: Database, make_settings: SettingsFactory) -> None:
    client = FakeLinksClient(thing(notes="api_key = sk-abcdefghijklmnopqrstuvwxyz123"))
    p = _pipeline(db, make_settings(), client)
    result = p.run("full")
    assert result.counts.quarantined == 1
    assert "sk-abcdefghij" not in json.dumps(_payloads(p))
    assert "[REDACTED]" in _payloads(p)[0]["text"]


def test_llm_summary_and_topics(db: Database, make_settings: SettingsFactory) -> None:
    db.conn.execute(
        "INSERT INTO entity_aliases VALUES ('location', 'geowiki:rooms/a', 'Section 2'),"
        " ('location', 'geowiki:rooms/b', 'Section 2'), ('asset', 'other123', 'Big Laser')"
    )
    db.commit()
    llm = FakeLLM()
    p = _pipeline(db, make_settings(), FakeLinksClient(thing()), llm=llm)
    p.run("full")
    payload = _payloads(p)[0]
    assert payload["text"].startswith("A laser cutter for members.")
    assert payload["topics"] == ["laser-cutting"]
    # Ambiguous "Section 2" confirmed by the LLM; asset IDs never come from the linker.
    assert payload["location_ids"] == ["geowiki:resources/12", "geowiki:rooms/a"]
    assert payload["asset_ids"] == ["abcd1234"]


def test_llm_failures_fall_back(db: Database, make_settings: SettingsFactory) -> None:
    db.conn.execute(
        "INSERT INTO entity_aliases VALUES ('location', 'l1', 'Section 2'),"
        " ('location', 'l2', 'Section 2')"
    )
    db.commit()
    p = _pipeline(db, make_settings(), FakeLinksClient(thing()), llm=FakeLLM(fail=True))
    assert p.run("full").status == "ok"
    payload = _payloads(p)[0]
    assert payload["text"].startswith("Big Laser")
    assert payload["topics"] == []
    assert payload["location_ids"] == ["geowiki:resources/12"]
    assert p._confirm_link("x", "a", "location", "l1") is False


def test_reindex_rebuilds_from_staging(db: Database, make_settings: SettingsFactory) -> None:
    client = FakeLinksClient(thing())
    embedder = FakeEmbedder()
    p = _pipeline(db, make_settings(), client, embedder=embedder)
    p.run("full")
    client.fail = True  # reindex never reads the asset manager
    stray = point_id("asset", "gone0000", 0)
    p.member.client.upsert(
        "hs_member",
        points=[
            qm.PointStruct(
                id=stray,
                vector={"dense": [0.1] * 8},
                payload={"source": "asset", "doc_id": "gone0000", "sync_run_id": "old"},
            )
        ],
    )
    result = p.run("reindex")
    assert result.status == "ok"
    assert result.counts.seen == 1
    assert len(embedder.calls) == 2
    assert {pl["doc_id"] for pl in _payloads(p)} == {"abcd1234"}


def test_listing_failure_fails_the_run(db: Database, make_settings: SettingsFactory) -> None:
    client = FakeLinksClient(thing())
    client.fail = True
    p = _pipeline(db, make_settings(), client)
    with pytest.raises(RuntimeError, match="listing failed"):
        p.run("full")
    run = db.recent_runs("links")[0]
    assert run["ok"] is False
    assert "listing failed" in run["message"]


def test_bad_thing_is_skipped(db: Database, make_settings: SettingsFactory) -> None:
    bad = thing("bad00000", thing_id=2)
    bad["links"] = 5  # not a list: parsing fails
    client = FakeLinksClient(thing(), bad)
    p = _pipeline(db, make_settings(), client)
    result = p.run("full")
    assert result.status == "ok"
    assert any("bad00000" in e for e in result.counts.errors)
    assert {pl["doc_id"] for pl in _payloads(p)} == {"abcd1234"}


def test_bad_staged_row_is_skipped_when_indexing(
    db: Database, make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _pipeline(db, make_settings(), FakeLinksClient(thing()))

    def boom(row: Any, run_id: str) -> Any:
        raise ValueError("bad row")

    monkeypatch.setattr(p, "asset_points", boom)
    result = p.run("full")
    assert any("bad row" in e for e in result.counts.errors)


def test_index_failure_makes_next_run_restage(
    db: Database, make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeLinksClient(thing())
    p = _pipeline(db, make_settings(), client)

    def boom(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(p, "_embed_points", boom)
    with pytest.raises(RuntimeError, match="qdrant down"):
        p.run("incremental")
    staged = db.staged_versions("asset")
    assert staged["abcd1234"][0] is None


def test_stop_indexes_what_was_staged(db: Database, make_settings: SettingsFactory) -> None:
    stop = threading.Event()
    client = FakeLinksClient(thing(), thing("efgh5678", thing_id=2, name="Bandsaw"))
    p = _pipeline(db, make_settings(), client, stop=stop)
    original = p._stage_thing

    def stage_then_stop(*args: Any, **kwargs: Any) -> bool:
        changed = original(*args, **kwargs)
        stop.set()
        return changed

    p._stage_thing = stage_then_stop  # type: ignore[method-assign]
    result = p.run("full")
    assert result.message == "stopped early"
    assert p.member.count("asset") == 1


def test_busy_lock(db: Database, make_settings: SettingsFactory, database_url: str) -> None:
    p = _pipeline(db, make_settings(), FakeLinksClient(thing()))
    with Database.connect(database_url) as other, other.source_lock("links") as locked:
        assert locked
        result = p.run("full")
    assert result.status == "busy"


def test_status_and_last_full_run(
    db: Database, make_settings: SettingsFactory, database_url: str
) -> None:
    settings = make_settings(database_url=database_url)
    assert pipeline_module.last_full_run(settings) is None
    client = FakeLinksClient(
        thing(), thing("zig00001", thing_id=2, name="Sensor", integration_source="unifi")
    )
    _pipeline(db, settings, client).run("full")
    assert pipeline_module.last_full_run(settings) is not None
    report = pipeline_module.status(settings)
    assert report["assets"] == 1
    assert report["assets_excluded"] == 1
    assert report["runs"][0]["mode"] == "full"


def test_run_builds_and_closes_everything(
    db: Database,
    make_settings: SettingsFactory,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(database_url=database_url, llm_model="m")
    client = FakeLinksClient(thing())
    closed: list[str] = []
    monkeypatch.setattr(client, "close", lambda: closed.append("client"), raising=False)
    monkeypatch.setattr(pipeline_module, "build_client", lambda s: client)
    monkeypatch.setattr(pipeline_module, "build_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(
        pipeline_module, "QdrantClient", lambda **kwargs: QdrantClient(location=":memory:")
    )
    monkeypatch.setattr(LLM, "analyze_asset", lambda self, **kw: AssetAnalysis("", []))
    result = pipeline_module.run(settings, "full")
    assert result.status == "ok"
    assert closed == ["client"]


def test_builders(make_settings: SettingsFactory, db: Database) -> None:
    assert pipeline_module.build_llm(make_settings(), db) is None
    llm = pipeline_module.build_llm(make_settings(llm_model="m", vision_model="v"), db)
    assert llm is not None
    assert llm.vision_model == "v"
    llm.close()
    client = pipeline_module.build_client(make_settings())
    assert client.page_size == 500
    client.close()


def test_load_taxonomy(tmp_path: Any) -> None:
    assert "laser-cutting" in load_taxonomy(None)
    path = tmp_path / "t.txt"
    path.write_text("# comment\nfoo\n\nbar\n")
    assert load_taxonomy(str(path)) == ["foo", "bar"]
