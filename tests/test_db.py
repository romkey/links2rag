import subprocess
import sys
import uuid
from urllib.parse import urlsplit

import psycopg

from links2rag.db import Database, RunCounts, StagedDocument, migrate, sqlalchemy_url


def test_sqlalchemy_url() -> None:
    assert sqlalchemy_url("postgresql://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert sqlalchemy_url("postgres://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert sqlalchemy_url("sqlite://") == "sqlite://"


def test_migrate_is_idempotent(database_url: str) -> None:
    migrate(database_url)
    migrate(database_url)


def test_concurrent_migrations_on_a_fresh_database(database_url: str) -> None:
    # Two runs may migrate at once on an empty database; they must not race.
    name = f"l2r_fresh_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    fresh_url = urlsplit(database_url)._replace(path=f"/{name}").geturl()
    script = "import sys; from links2rag.db import migrate; migrate(sys.argv[1])"
    try:
        procs = [
            # S603: this interpreter running a fixed script.
            subprocess.Popen([sys.executable, "-c", script, fresh_url], stderr=subprocess.PIPE)  # noqa: S603
            for _ in range(2)
        ]
        for proc in procs:
            _, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, err.decode()
    finally:
        with psycopg.connect(database_url, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def _doc(doc_id: str = "a1", *, indexable: bool = True) -> StagedDocument:
    return StagedDocument(
        source="asset",
        source_doc_id=doc_id,
        title="T",
        url="https://x.example.com",
        tier="member",
        raw_content={"k": doc_id},
        extracted_text="text",
        content_hash="h",
        source_updated_at="2026-01-01",
        indexable=indexable,
    )


def test_runs_and_quarantine(db: Database) -> None:
    db.start_run(run_id="r1", source="links", mode="full")
    db.finish_run(run_id="r1", counts=RunCounts(seen=5), ok=True)
    db.start_run(run_id="r2", source="links", mode="full")
    assert db.previous_full_run_seen("links", "r2") == 5
    assert db.last_full_run_at("links") is not None
    assert [r["run_id"] for r in db.recent_runs("links")] == ["r2", "r1"]
    db.add_quarantine(source="asset", source_doc_id="a1", reason="x", detail={}, run_id="r1")
    assert db.quarantine_count(["asset"]) == 1
    assert db.quarantine_rows(["asset"])[0]["reason"] == "x"


def test_staging(db: Database) -> None:
    db.upsert_staging(_doc(), run_id="r1")
    db.upsert_staging(_doc("a2", indexable=False), run_id="r1")
    db.upsert_staging(_doc(), run_id="r2")
    assert db.staged_versions("asset") == {"a1": ("2026-01-01", "h"), "a2": ("2026-01-01", "h")}
    assert [r["source_doc_id"] for r in db.staged_documents("asset")] == ["a1"]
    assert db.staged_documents("asset", ["a2"]) == []
    row = db.staged_document("asset", "a2")
    assert row is not None
    assert row["raw_content"] == {"k": "a2"}
    db.forget_versions("asset", ["a1"])
    assert db.staged_versions("asset")["a1"][0] is None
    assert db.delete_staging("asset", ["a1", "a2"]) == 2


def test_llm_cache(db: Database) -> None:
    assert db.llm_cache_get("t", "m", "k") is None
    db.llm_cache_put("t", "m", "k", {"a": 1})
    db.llm_cache_put("t", "m", "k", {"a": 2})
    assert db.llm_cache_get("t", "m", "k") == {"a": 2}


def _asset(asset_id: str = "a1") -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "thing_id": 1,
        "slug": None,
        "name": "Laser",
        "label_name": None,
        "manufacturer": None,
        "model": None,
        "category": None,
        "ownership": "space",
        "location_ids": [],
        "url": None,
        "short_url": None,
        "retired_ids": ["old"],
        "related_asset_ids": [],
        "links": [{"type": "wiki"}],
        "indexable": True,
    }


def test_registry_and_aliases(db: Database) -> None:
    db.upsert_asset(_asset())
    db.upsert_asset({**_asset(), "name": "Big Laser"})
    assert [(r["asset_id"], r["name"], r["links"]) for r in db.assets()] == [
        ("a1", "Big Laser", [{"type": "wiki"}])
    ]
    db.replace_asset_aliases("a1", ["Laser", "Big Laser", ""], keep=set())
    db.replace_asset_aliases("a1", ["Big Laser"], keep={"the beast"})
    assert {r["alias"] for r in db.aliases()} == {"Big Laser"}
    db.delete_asset_aliases(["a1"])
    db.delete_assets(["a1"])
    assert db.assets() == []
    assert db.aliases() == []


def test_alias_review_queue(db: Database) -> None:
    assert db.queue_alias("a1", "the beast", {"from": "slack"}) is True
    assert db.queue_alias("a1", "the beast", {}) is False
    assert db.queue_alias("a1", "lazzy", {}) is True
    pending = db.alias_queue()
    assert [r["right_id"] for r in pending] == ["the beast", "lazzy"]
    approved = db.resolve_alias(pending[0]["id"], approve=True)
    assert approved is not None
    assert approved["details"]["decision"] == "approved"
    db.resolve_alias(pending[1]["id"], approve=False)
    assert db.resolve_alias(999, approve=True) is None
    assert db.alias_queue() == []
    assert len(db.alias_queue(pending_only=False)) == 2
    assert db.confirmed_nicknames("a1") == {"the beast"}
    assert {r["alias"] for r in db.aliases()} == {"the beast"}
    # Regenerated aliases keep the approved nickname.
    db.replace_asset_aliases("a1", ["Laser"], keep=db.confirmed_nicknames("a1"))
    assert {r["alias"] for r in db.aliases()} == {"Laser", "the beast"}


def test_manual_links(db: Database) -> None:
    db.upsert_manual_link(url="u1", asset_ids=["b", "a", "a"], title="M", origin="link")
    db.record_fetch("u1", status="ok", http_status=200, manual_id="sha", content_type="pdf")
    db.record_fetch("u1", status="dead", http_status=404, error="gone")
    row = db.manual_links()["u1"]
    assert row["asset_ids"] == ["a", "b"]
    assert row["status"] == "dead"
    assert row["manual_id"] == "sha"  # the last good file stays
    assert row["content_type"] == "pdf"
    assert row["last_ok_at"] is not None
    db.upsert_manual_link(
        url="u1", asset_ids=["c"], title=None, origin="discovered", parent_urls=["p", "p"]
    )
    assert db.manual_links()["u1"]["asset_ids"] == ["c"]
    assert db.manual_links()["u1"]["parent_urls"] == ["p"]
    db.delete_manual_links(["u1"])
    assert db.manual_links() == {}
