from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytesseract
import pytest
from qdrant_client import QdrantClient

from links2rag import manual_sync
from links2rag.config import Settings
from links2rag.db import Database
from links2rag.fetch import Fetched, Kind
from links2rag.llm import AssetAnalysis
from links2rag.pipeline import Pipeline
from tests.fakes import FakeEmbedder, FakeLinksClient, thing
from tests.pdfgen import ENGLISH, GERMAN, Page, build_pdf

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
SettingsFactory = Callable[..., Settings]

SAW_PDF = build_pdf(
    [
        Page([("Safety", 20), (ENGLISH, 10)]),
        Page([("Blade settings", 20)], table=[["Blade", "TPI"], ["1/4 in", "6"]]),
    ],
    title="Rikon 10-305 Bandsaw Manual",
)
LASER_PDF = build_pdf([Page([("Operation", 20), ("Focus the lens before cutting acrylic.", 10)])])
GERMAN_PDF = build_pdf([Page([("Sicherheit", 20), (GERMAN, 10)])])


class FakeFetcher:
    def __init__(self, responses: dict[str, Fetched | bytes]) -> None:
        self.responses = responses
        self.calls: list[str] = []
        self.closed = False

    def blocked_reason(self, url: str) -> str | None:
        return None

    def fetch(self, url: str) -> Fetched:
        self.calls.append(url)
        response = self.responses.get(url)
        if response is None:
            return Fetched("dead", url, http_status=404, error="HTTP 404")
        if isinstance(response, Fetched):
            return response
        kind: Kind = "pdf" if response.startswith(b"%PDF") else "html"
        return Fetched("ok", url, final_url=url, http_status=200, kind=kind, content=response)

    def close(self) -> None:
        self.closed = True


class FakeLLM:
    model = "fake"
    vision_model = "vlm"

    def analyze_asset(self, *, facts: str, taxonomy: list[str]) -> AssetAnalysis:
        return AssetAnalysis("", [])

    def confirm_link(self, **kwargs: Any) -> bool:
        return False

    def chunk_context(self, *, document_title: str, document: str, chunk: str) -> str:
        return f"Context for {document_title}."

    def manual_section(self, *, heading: str, text: str) -> str | None:
        return "operation"

    def describe_image(self, *, png: bytes, context: str) -> str | None:
        return "A diagram."


def _link(title: str, url: str) -> dict[str, Any]:
    return {"type": "custom", "title": title, "url": url, "note": None}


def _saw(
    key: str = "saw00001", thing_id: int = 1, url: str = "https://v.example.com/saw.pdf"
) -> Any:
    return thing(
        key,
        thing_id=thing_id,
        name="Bandsaw",
        manufacturer="Rikon",
        model="10-305",
        links=[_link("Manual", url)],
    )


def _laser() -> Any:
    return thing(
        "laser001",
        thing_id=3,
        name="Big Laser",
        links=[_link("User guide", "https://l.example.com/laser.pdf")],
    )


def _pipeline(
    db: Database,
    settings: Settings,
    client: FakeLinksClient,
    fetcher: FakeFetcher,
    *,
    qdrant: QdrantClient | None = None,
    llm: Any = None,
    embedder: FakeEmbedder | None = None,
) -> Pipeline:
    return Pipeline(
        settings=settings,
        db=db,
        embedder=embedder or FakeEmbedder(),
        qdrant=qdrant or QdrantClient(location=":memory:"),
        client=client,
        llm=llm,
        fetcher=fetcher,
    )


@pytest.fixture
def settings(make_settings: SettingsFactory, tmp_path: Path) -> Settings:
    return make_settings(manuals_enabled=True, data_dir=str(tmp_path / "data"))


def _manual_payloads(p: Pipeline) -> list[dict[str, Any]]:
    return sorted(p.member.payloads("manual"), key=lambda pl: (pl["doc_id"], pl["chunk_index"]))


def test_manuals_are_fetched_deduplicated_and_indexed(
    db: Database, settings: Settings, tmp_path: Path
) -> None:
    # Two saws link to the same file at different URLs; the laser has its own.
    client = FakeLinksClient(
        _saw(), _saw("saw00002", 2, "https://mirror.example.com/rikon.pdf"), _laser()
    )
    fetcher = FakeFetcher(
        {
            "https://v.example.com/saw.pdf": SAW_PDF,
            "https://mirror.example.com/rikon.pdf": SAW_PDF,
            "https://l.example.com/laser.pdf": LASER_PDF,
        }
    )
    p = _pipeline(db, settings, client, fetcher)
    result = p.run("full")
    assert result.status == "ok"
    assert len(fetcher.calls) == 3
    payloads = _manual_payloads(p)
    manual_ids = {pl["doc_id"] for pl in payloads}
    assert len(manual_ids) == 2
    saw = [pl for pl in payloads if "saw00001" in pl["asset_ids"]]
    assert {tuple(pl["asset_ids"]) for pl in saw} == {("saw00001", "saw00002")}
    # A bandsaw question never pulls in the laser manual: asset IDs come only from links.
    assert all("laser001" not in pl["asset_ids"] for pl in saw)
    summary, safety, table = saw[0], saw[1], saw[2]
    assert summary["block_kind"] == "summary"
    assert summary["title"] == "Rikon 10-305 Bandsaw Manual"
    assert "For: Bandsaw (Rikon 10-305)" in summary["text"]
    assert safety["manual_section"] == "safety"
    assert safety["page"] == 1
    assert safety["url"].endswith(".pdf#page=1")
    assert safety["authority"] == 2
    assert safety["tier"] == "member"
    assert table["block_kind"] == "table"
    assert table["page"] == 2
    assert "Blade | TPI" in table["text"]
    assert table["local_file"].startswith("manuals/")
    assert (tmp_path / "data" / table["local_file"]).exists()
    staged = db.staged_document("manual", summary["doc_id"])
    assert staged is not None
    assert staged["raw_content"]["asset_ids"] == ["saw00001", "saw00002"]
    assert len(staged["raw_content"]["urls"]) == 2
    report = db.manual_links()
    assert {r["status"] for r in report.values()} == {"ok"}
    # Unchanged and not due: nothing is fetched or re-embedded again.
    fetcher.calls.clear()
    p.run("incremental")
    assert fetcher.calls == []


def test_dead_links_keep_the_last_good_copy(db: Database, settings: Settings) -> None:
    client = FakeLinksClient(_saw())
    fetcher = FakeFetcher({"https://v.example.com/saw.pdf": SAW_PDF})
    p = _pipeline(db, settings, client, fetcher)
    p.run("full")
    later = datetime.now(UTC) + timedelta(days=8)
    assert p.manuals is not None
    p.manuals.now = lambda: later
    fetcher.responses.clear()  # the vendor removed the file
    p.run("incremental")
    row = db.manual_links()["https://v.example.com/saw.pdf"]
    assert row["status"] == "dead"
    assert row["manual_id"]
    payloads = _manual_payloads(p)
    assert payloads
    assert all(pl["link_dead"] for pl in payloads)


def test_downloads_and_manufacturer_pages_are_followed_one_level(
    db: Database, settings: Settings
) -> None:
    downloads = (
        b"<html><body><h1>Downloads</h1><a href='/files/saw.pdf'>Owner's manual</a>"
        b"<a href='/files/brochure.pdf'>Brochure</a></body></html>"
    )
    product = (
        b"<html><body><p>Buy it now</p><a href='/docs/laser-manual.pdf'>Manual</a>"
        b"<a href='/docs/price-list.pdf'>Prices</a></body></html>"
    )
    saw = _saw(url="https://v.example.com/downloads")
    laser = thing(
        "laser001", thing_id=3, links=[], manufacturer_url="https://thunder.example.com/nova"
    )
    fetcher = FakeFetcher(
        {
            "https://v.example.com/downloads": downloads,
            "https://v.example.com/files/saw.pdf": SAW_PDF,
            "https://v.example.com/files/brochure.pdf": build_pdf(
                [Page([("Brochure", 20), (ENGLISH, 10)])]
            ),
            "https://thunder.example.com/nova": product,
            "https://thunder.example.com/docs/laser-manual.pdf": LASER_PDF,
        }
    )
    p = _pipeline(db, settings, FakeLinksClient(saw, laser), fetcher)
    p.run("full")
    links = db.manual_links()
    assert links["https://v.example.com/downloads"]["status"] == "index"
    assert links["https://v.example.com/files/saw.pdf"]["origin"] == "discovered"
    assert links["https://v.example.com/files/saw.pdf"]["asset_ids"] == ["saw00001"]
    # The manufacturer page: only links that look like manuals are followed.
    assert "https://thunder.example.com/docs/price-list.pdf" not in links
    assert links["https://thunder.example.com/nova"]["status"] == "index"
    assets_of = {tuple(pl["asset_ids"]) for pl in _manual_payloads(p)}
    assert ("laser001",) in assets_of
    assert ("saw00001",) in assets_of
    # Discovered files stay while their page is linked; go when it isn't.
    p.client.things["saw00001"]["links"] = []  # type: ignore[attr-defined]
    p.run("incremental")
    assert "https://v.example.com/files/saw.pdf" not in db.manual_links()


def test_html_manuals_and_unwanted_pages(db: Database, settings: Settings) -> None:
    html_manual = b"<html><head><title>Drill press guide</title></head><body>" + (
        b"<h2>Operation</h2><p>" + ENGLISH.encode() * 60 + b"</p></body></html>"
    )
    shop = b"<html><body><p>" + b"Buy our products today. " * 400 + b"</p></body></html>"
    client = FakeLinksClient(
        thing(
            "drill001",
            thing_id=4,
            name="Drill press",
            links=[
                _link("Guide", "https://d.example.com/guide"),
                _link("Manual (shop)", "https://d.example.com/shop"),
                _link("Manual zip", "https://d.example.com/m.zip"),
            ],
        )
    )
    fetcher = FakeFetcher(
        {
            "https://d.example.com/guide": html_manual,
            "https://d.example.com/shop": shop,
            "https://d.example.com/m.zip": Fetched(
                "unsupported", "https://d.example.com/m.zip", error="unsupported type"
            ),
        }
    )
    p = _pipeline(db, settings, client, fetcher)
    p.run("full")
    payloads = _manual_payloads(p)
    titles = {pl["title"] for pl in payloads}
    assert "Drill press guide" in titles
    assert all(pl["page"] is None for pl in payloads)
    assert db.manual_links()["https://d.example.com/m.zip"]["status"] == "unsupported"


def test_removed_links_delete_manuals_and_files(
    db: Database, settings: Settings, tmp_path: Path
) -> None:
    client = FakeLinksClient(_saw(), _laser())
    fetcher = FakeFetcher(
        {"https://v.example.com/saw.pdf": SAW_PDF, "https://l.example.com/laser.pdf": LASER_PDF}
    )
    p = _pipeline(db, settings, client, fetcher)
    p.run("full")
    files = tmp_path / "data" / "manuals"
    assert len(list(files.iterdir())) == 2
    del client.things["laser001"]
    result = p.run("full")
    assert result.counts.deleted == 2  # the asset and its manual
    assert {tuple(pl["asset_ids"]) for pl in _manual_payloads(p)} == {("saw00001",)}
    assert len(list(files.iterdir())) == 1
    assert len(db.staged_versions("manual")) == 1


def test_safety_valve_keeps_manuals(db: Database, settings: Settings) -> None:
    things = [_saw()] + [thing(f"key{n:05d}", thing_id=10 + n, links=[]) for n in range(5)]
    client = FakeLinksClient(*things)
    p = _pipeline(db, settings, client, FakeFetcher({"https://v.example.com/saw.pdf": SAW_PDF}))
    p.run("full")
    for key in list(client.things)[:5]:
        del client.things[key]
    assert p.run("full").status == "valve"
    assert db.manual_links()
    assert _manual_payloads(p)


def test_dropins(db: Database, make_settings: SettingsFactory, tmp_path: Path) -> None:
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "saw.pdf").write_bytes(SAW_PDF)
    (drop / "saw.pdf.yaml").write_text("asset_keys: [saw00001]\ntitle: Saw manual\n")
    (drop / "notes.txt").write_bytes(b"plain text")
    (drop / "notes.txt.yaml").write_text("asset_keys: [saw00001]\n")
    settings = make_settings(
        manuals_enabled=True, data_dir=str(tmp_path / "data"), manuals_dir=str(drop)
    )
    p = _pipeline(
        db, settings, FakeLinksClient(_saw(url="https://x.example.com/none")), FakeFetcher({})
    )
    p.run("full")
    payloads = _manual_payloads(p)
    assert payloads
    assert payloads[0]["asset_ids"] == ["saw00001"]
    assert payloads[0]["manual_urls"] == []
    assert payloads[0]["url"] is None
    assert payloads[0]["link_dead"] is False
    statuses = {url.rsplit("/", 1)[-1]: r["status"] for url, r in db.manual_links().items()}
    assert statuses["notes.txt"] == "unsupported"
    assert statuses["none"] == "dead"
    # A drop-in that can't be read is recorded as an error.
    assert p.manuals is not None
    missing = manual_sync.DROPIN_PREFIX + str(drop / "missing.pdf")
    db.upsert_manual_link(url=missing, asset_ids=["saw00001"], title=None, origin="dropin")
    p.manuals._fetch_dropin(missing)
    assert db.manual_links()[missing]["status"] == "error"
    # A removed drop-in takes its manual with it.
    (drop / "saw.pdf").unlink()
    p.run("full")
    assert _manual_payloads(p) == []


def test_non_english_manuals_are_staged_not_indexed(db: Database, settings: Settings) -> None:
    client = FakeLinksClient(_saw())
    p = _pipeline(db, settings, client, FakeFetcher({"https://v.example.com/saw.pdf": GERMAN_PDF}))
    p.run("full")
    assert _manual_payloads(p) == []
    (manual_id,) = db.staged_versions("manual")
    staged = db.staged_document("manual", manual_id)
    assert staged is not None
    assert staged["indexable"] is False


def test_secrets_in_manuals_are_redacted(db: Database, settings: Settings) -> None:
    pdf = build_pdf(
        [Page([("Setup", 20), ("Default password = hunter2hunter2 for the admin", 10)])]
    )
    client = FakeLinksClient(_saw())
    p = _pipeline(db, settings, client, FakeFetcher({"https://v.example.com/saw.pdf": pdf}))
    result = p.run("full")
    assert result.counts.quarantined == 1
    assert "hunter2" not in json.dumps(_manual_payloads(p))


def test_llm_context_sections_and_diagrams(db: Database, settings: Settings) -> None:
    pdf = build_pdf(
        [Page([("Using it", 20), (ENGLISH, 10)]), Page([("Fig 1", 10)], shapes=60)],
        title="Saw",
    )
    client = FakeLinksClient(_saw())
    p = _pipeline(
        db, settings, client, FakeFetcher({"https://v.example.com/saw.pdf": pdf}), llm=FakeLLM()
    )
    p.run("full")
    payloads = _manual_payloads(p)
    body = [pl for pl in payloads if pl["block_kind"] == "text"]
    assert body[0]["context"] == "Context for Saw."
    assert body[0]["manual_section"] == "operation"
    diagram = [pl for pl in payloads if pl["block_kind"] == "diagram"]
    assert diagram[0]["machine_generated"] is True
    assert "A diagram." in diagram[0]["text"]


def test_reindex_and_force(db: Database, settings: Settings) -> None:
    client = FakeLinksClient(_saw())
    fetcher = FakeFetcher({"https://v.example.com/saw.pdf": SAW_PDF})
    embedder = FakeEmbedder()
    p = _pipeline(db, settings, client, fetcher, embedder=embedder)
    p.run("full")
    count = len(_manual_payloads(p))
    fetcher.calls.clear()
    p.run("reindex")
    assert fetcher.calls == []
    assert len(_manual_payloads(p)) == count
    p.run("full", force=True)
    assert fetcher.calls == ["https://v.example.com/saw.pdf"]


def test_bad_manuals_are_skipped(
    db: Database, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeLinksClient(_saw())
    p = _pipeline(
        db, settings, client, FakeFetcher({"https://v.example.com/saw.pdf": b"%PDF-junk"})
    )
    result = p.run("full")
    assert any(e.startswith("manual ") for e in result.counts.errors)
    assert _manual_payloads(p) == []
    # A staged manual whose points can't be built is skipped too.
    p2 = _pipeline(db, settings, client, FakeFetcher({"https://v.example.com/saw.pdf": SAW_PDF}))
    p2.run("full", force=True)
    assert p2.manuals is not None

    def boom(row: Any, run_id: str) -> Any:
        raise ValueError("bad")

    monkeypatch.setattr(p2.manuals, "manual_points", boom)
    result = p2.run("full", force=True)
    assert any("bad" in e for e in result.counts.errors)


def test_build_ocr(make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    assert manual_sync.build_ocr(make_settings(ocr_enabled=False)) is None

    def missing() -> str:
        raise OSError("not installed")

    monkeypatch.setattr(pytesseract, "get_tesseract_version", missing)
    assert manual_sync.build_ocr(make_settings()) is None
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "5")
    monkeypatch.setattr(pytesseract, "image_to_string", lambda image, lang: f"text {lang}")
    ocr = manual_sync.build_ocr(make_settings(ocr_language="eng+deu"))
    assert ocr is not None
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    assert ocr(buffer.getvalue()) == "text eng+deu"


def test_build_fetcher(make_settings: SettingsFactory) -> None:
    fetcher = manual_sync.build_fetcher(make_settings(manual_max_mb=2))
    assert fetcher.max_bytes == 2 * 1024 * 1024
    assert "links.example.com" in fetcher.skip_hosts
    fetcher.close()


def test_a_file_on_several_pages_keeps_every_asset(db: Database, settings: Settings) -> None:
    # Two saws reach the same PDF through different downloads pages; a third
    # asset links to the file directly.
    page_a = b"<html><body><a href='/files/saw.pdf'>Manual</a></body></html>"
    page_b = b"<html><body><a href='https://v.example.com/files/saw.pdf'>Manual</a></body></html>"
    direct = thing(
        "fence001",
        thing_id=5,
        name="Saw fence",
        links=[_link("Manual", "https://v.example.com/files/saw.pdf")],
    )
    client = FakeLinksClient(
        _saw(url="https://v.example.com/dl-a"),
        _saw("saw00002", 2, "https://www.v.example.com/dl-b"),
        direct,
    )
    fetcher = FakeFetcher(
        {
            "https://v.example.com/dl-a": page_a,
            "https://www.v.example.com/dl-b": page_b,
            "https://v.example.com/files/saw.pdf": SAW_PDF,
        }
    )
    p = _pipeline(db, settings, client, fetcher)
    p.run("full")
    row = db.manual_links()["https://v.example.com/files/saw.pdf"]
    assert row["asset_ids"] == ["fence001", "saw00001", "saw00002"]
    assert row["parent_urls"] == ["https://v.example.com/dl-a", "https://www.v.example.com/dl-b"]
    assert row["origin"] == "link"
    assert {tuple(pl["asset_ids"]) for pl in _manual_payloads(p)} == {
        ("fence001", "saw00001", "saw00002")
    }
    # One page stops listing the file: that page's asset loses it on the next check.
    fetcher.responses["https://www.v.example.com/dl-b"] = b"<html><body>Moved</body></html>"
    p.run("full", force=True)
    row = db.manual_links()["https://v.example.com/files/saw.pdf"]
    assert row["asset_ids"] == ["fence001", "saw00001"]
    assert row["parent_urls"] == ["https://v.example.com/dl-a"]


def test_secrets_never_reach_staging_or_the_llm(db: Database, settings: Settings) -> None:
    pdf = build_pdf(
        [Page([("Setup", 20), ("Default password = hunter2hunter2 for the admin", 10)])]
    )
    seen: list[str] = []

    class RecordingLLM(FakeLLM):
        def chunk_context(self, *, document_title: str, document: str, chunk: str) -> str:
            seen.extend([document, chunk])
            return "ctx"

    client = FakeLinksClient(_saw())
    p = _pipeline(
        db,
        settings,
        client,
        FakeFetcher({"https://v.example.com/saw.pdf": pdf}),
        llm=RecordingLLM(),
    )
    p.run("full")
    (manual_id,) = db.staged_versions("manual")
    staged = db.staged_document("manual", manual_id)
    assert staged is not None
    assert "hunter2" not in staged["extracted_text"]
    assert "hunter2" not in json.dumps(staged["raw_content"])
    assert seen
    assert all("hunter2" not in text for text in seen)


def test_a_failed_page_refetch_keeps_its_files(db: Database, settings: Settings) -> None:
    page = b"<html><body><a href='/files/saw.pdf'>Manual</a></body></html>"
    client = FakeLinksClient(_saw(url="https://v.example.com/downloads"))
    fetcher = FakeFetcher(
        {"https://v.example.com/downloads": page, "https://v.example.com/files/saw.pdf": SAW_PDF}
    )
    p = _pipeline(db, settings, client, fetcher)
    p.run("full")
    before = _manual_payloads(p)
    assert before
    # The downloads page is down at the next check: its file stays indexed.
    del fetcher.responses["https://v.example.com/downloads"]
    del fetcher.responses["https://v.example.com/files/saw.pdf"]
    p.run("full", force=True)
    links = db.manual_links()
    assert links["https://v.example.com/downloads"]["status"] == "dead"
    assert links["https://v.example.com/files/saw.pdf"]["parent_urls"] == [
        "https://v.example.com/downloads"
    ]
    after = _manual_payloads(p)
    assert len(after) == len(before)
    assert all(pl["link_dead"] for pl in after)
