from __future__ import annotations

import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from links2rag import commands, reports
from links2rag.config import Settings
from links2rag.db import Database
from links2rag.documents import point_id
from links2rag.llm import AssetAnalysis, LLMError
from links2rag.pipeline import Pipeline
from links2rag.search import search
from links2rag.sparse import sparse_vector
from tests.fakes import FakeEmbedder, FakeLinksClient, thing

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
SettingsFactory = Callable[..., Settings]


def _add_other_source(p: Pipeline, source: str, text: str, n: int) -> None:
    vector = FakeEmbedder().encode([text])[0]
    p.member.upsert(
        [
            qm.PointStruct(
                id=point_id(source, f"d{n}", 0),
                vector={"dense": vector, "sparse": sparse_vector(text)},
                payload={"source": source, "text": text, "url": f"https://{source}/{n}"},
            )
        ]
    )


class NicknameLLM:
    model = "fake"
    vision_model = None

    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.seen: list[list[str]] = []

    def analyze_asset(self, **kwargs: Any) -> AssetAnalysis:
        return AssetAnalysis("", [])

    def confirm_link(self, **kwargs: Any) -> bool:
        return False

    def propose_nicknames(self, *, name: str, facts: str, mentions: list[str]) -> list[str]:
        self.seen.append(mentions)
        if self.fail == "llm":
            raise LLMError("down")
        if self.fail == "other":
            raise RuntimeError("qdrant down")
        return ["the beast", "big laser"]


def _pipeline(db: Database, settings: Settings, llm: Any, qdrant: QdrantClient) -> Pipeline:
    return Pipeline(
        settings=settings,
        db=db,
        embedder=FakeEmbedder(),
        qdrant=qdrant,
        client=FakeLinksClient(thing()),
        llm=llm,
    )


def test_nicknames_are_proposed_from_wiki_and_slack(
    db: Database, make_settings: SettingsFactory
) -> None:
    qdrant = QdrantClient(location=":memory:")
    llm = NicknameLLM()
    p = _pipeline(db, make_settings(), llm, qdrant)
    _add_other_source(p, "wiki", "The beast (our Big Laser) needs air assist.", 1)
    _add_other_source(p, "slack", "is the beast free tonight?", 2)
    _add_other_source(p, "event", "Laser class", 3)
    p.run("full")
    assert set(llm.seen[0]) == {
        "The beast (our Big Laser) needs air assist.",
        "is the beast free tonight?",
    }
    queue = db.alias_queue()
    assert [(r["left_id"], r["right_id"]) for r in queue] == [("abcd1234", "the beast")]
    assert queue[0]["details"]["from"] == ["https://slack/2", "https://wiki/1"]
    # Unchanged on the next incremental run: nothing new is asked.
    p.run("incremental")
    assert len(llm.seen) == 1


@pytest.mark.parametrize("fail", ["llm", "other"])
def test_nickname_failures_are_logged(
    db: Database, make_settings: SettingsFactory, fail: str
) -> None:
    p = _pipeline(db, make_settings(), NicknameLLM(fail), QdrantClient(location=":memory:"))
    assert p.run("full").status == "ok"
    assert db.alias_queue() == []


def test_search_filters(db: Database, make_settings: SettingsFactory) -> None:
    qdrant = QdrantClient(location=":memory:")
    p = _pipeline(db, make_settings(), None, qdrant)
    p.run("full")
    _add_other_source(p, "wiki", "Big Laser settings", 1)
    embedder = FakeEmbedder()
    hits = search(qdrant, "hs_member", embedder, "Big Laser", limit=5)
    assert {(h.payload or {})["source"] for h in hits} == {"asset", "wiki"}
    hits = search(
        qdrant, "hs_member", embedder, "Big Laser", sources=["asset"], asset_id="abcd1234"
    )
    assert [(h.payload or {})["doc_id"] for h in hits] == ["abcd1234"]
    assert search(qdrant, "hs_member", embedder, "Big Laser", asset_id="nope") == []
    assert embedder.calls
    assert all(len(c) == 1 for c in embedder.calls)


def test_reports(db: Database, make_settings: SettingsFactory, tmp_path: Path) -> None:
    p = _pipeline(
        db, make_settings(data_dir=str(tmp_path)), None, QdrantClient(location=":memory:")
    )
    p.client.add(thing("nolinks1", thing_id=2, name="Mystery box", links=[]))  # type: ignore[attr-defined]
    p.run("full")
    db.upsert_manual_link(
        url="https://v.example.com/m.pdf", asset_ids=["abcd1234"], title="M", origin="link"
    )
    db.record_fetch("https://v.example.com/m.pdf", status="ok", manual_id="sha")
    db.record_fetch("https://v.example.com/m.pdf", status="dead", http_status=404, error="HTTP 404")
    db.add_quarantine(source="asset", source_doc_id="x", reason="secret", detail={}, run_id="r")
    db.queue_alias("abcd1234", "the beast", {"proposed_by": "llm"})
    db.commit()
    report = reports.build(db)
    assert report["assets"] == 2
    assert [a["asset_id"] for a in report["no_wiki_page"]] == ["nolinks1"]
    assert [a["asset_id"] for a in report["no_manual"]] == ["nolinks1"]
    assert [a["asset_id"] for a in report["no_location"]] == ["nolinks1"]
    assert len(report["no_training"]) == 2
    assert report["manual_links_failing"][0]["still_indexed"] is True
    assert report["quarantine"][0]["reason"] == "secret"
    assert report["aliases_to_review"][0]["alias"] == "the beast"
    assert "2 assets: 1 without a wiki page" in reports.summary(report)
    written = (tmp_path / "reports" / "links2rag.json").read_text()
    assert json.loads(written)["assets"] == 2  # written by the full run
    blocked = tmp_path / "file"
    blocked.write_text("")
    assert reports.write(report, str(blocked)) is None


def test_commands(
    db: Database, make_settings: SettingsFactory, database_url: str, tmp_path: Path
) -> None:
    settings = make_settings(database_url=database_url, data_dir=str(tmp_path))
    db.queue_alias("abcd1234", "the beast", {"proposed_by": "llm"})
    db.queue_alias("abcd1234", "lazzy", {})
    db.commit()
    pending = commands.run_command(settings, "aliases")
    assert [r["alias"] for r in pending] == ["the beast", "lazzy"]
    approved = commands.run_command(settings, "aliases", approve=pending[0]["id"])
    assert approved["decision"] == "approved"
    rejected = commands.run_command(settings, "aliases", reject=pending[1]["id"])
    assert rejected["decision"] == "rejected"
    assert commands.run_command(settings, "aliases") == []
    with pytest.raises(commands.CommandError, match="no proposed alias"):
        commands.run_command(settings, "aliases", approve=999)
    with pytest.raises(commands.CommandError, match="not both"):
        commands.run_command(settings, "aliases", approve=1, reject=2)
    report = commands.run_command(settings, "report")
    assert report["assets"] == 0
    assert (tmp_path / "reports" / "links2rag.json").exists()
    with pytest.raises(commands.CommandError, match="search text"):
        commands.run_command(settings, "query", text=" ")


def test_query(
    db: Database, make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    qdrant = QdrantClient(location=":memory:")
    settings = make_settings()
    _pipeline(db, settings, None, qdrant).run("full")
    monkeypatch.setattr(commands, "build_embedder", lambda s: FakeEmbedder())
    results = commands.query(
        settings, "Big Laser", limit=3, asset=None, sources=["asset"], client=qdrant
    )
    assert results[0]["title"] == "Big Laser"
    assert results[0]["asset_ids"] == ["abcd1234"]

    closed: list[str] = []

    class ClosingEmbedder(FakeEmbedder):
        def close(self) -> None:
            closed.append("embedder")

    class FakeQdrant:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def query_points(self, **kwargs: Any) -> Any:
            return types.SimpleNamespace(points=[])

        def close(self) -> None:
            closed.append("qdrant")

    monkeypatch.setattr(commands, "build_embedder", lambda s: ClosingEmbedder())
    monkeypatch.setattr(commands, "QdrantClient", FakeQdrant)
    assert commands.run_command(settings, "query", text="laser") == []
    assert closed == ["embedder", "qdrant"]
