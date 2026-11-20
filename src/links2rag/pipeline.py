"""One sync run: asset manager -> staging + registry -> sanitize -> index in Qdrant.

Modes:
- ``incremental``: list every thing (the API returns whole records, a few
  pages in all), stage and index the ones that changed, drop the ones that
  are gone.
- ``full``: the same, plus every point not stamped with this run's ID is
  deleted, unless the run saw less than half the things of the previous one.
- ``reindex``: rebuild every point from staging without reading the asset
  manager or downloading manuals, re-embedding everything (after a model or
  chunking change).

After the assets, the manuals they link to are synced (see manual_sync.py):
links due for a check are downloaded, files are staged once per SHA-256,
and their chunks indexed with the linking assets' IDs.

``force`` (``run --force``, or ``FORCE_REINDEX`` for serve's first scan)
restages, re-downloads, re-parses and re-embeds everything as well.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Protocol

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from links2rag import assets as asset_docs
from links2rag import reports
from links2rag.config import Settings
from links2rag.db import Database, RunCounts, StagedDocument
from links2rag.documents import (
    ASSET_SOURCE,
    MANUAL_SOURCE,
    TIER,
    base_payload,
    payload_hash,
    point_id,
)
from links2rag.embeddings import Embedder, build_embedder
from links2rag.linker import Linker
from links2rag.links_client import LinksClient
from links2rag.llm import LLM, LLMError
from links2rag.manual_sync import (
    DEAD_STATUSES,
    FetcherLike,
    ManualSync,
    build_fetcher,
    build_ocr,
)
from links2rag.qdrant_store import COMMON_INDEXES, LINKS_INDEXES, QdrantStore
from links2rag.search import search
from links2rag.sparse import SPARSE_ENCODER, sparse_vector

logger = logging.getLogger(__name__)

# The sync_runs source and lock name for this ingester's runs.
RUN_SOURCE = "links"
# A run that sees less than this share of the previous run's things deletes nothing.
SAFETY_RATIO = 0.5
# Other sources whose text the nickname pass reads ("the big laser is down").
NICKNAME_SOURCES = ("wiki", "slack")
NICKNAME_MENTIONS = 6

Mode = Literal["incremental", "full", "reindex"]
# Any: staged records and payloads are JSON.
JSON = dict[str, Any]


class LinksSource(Protocol):
    """What the pipeline needs from the asset manager (LinksClient, or a fake in tests)."""

    def list_things(self) -> list[JSON]: ...


@dataclass(frozen=True)
class RunResult:
    status: Literal["ok", "busy", "valve"]
    counts: RunCounts
    message: str = ""


def load_taxonomy(path: str | None) -> list[str]:
    text = (
        Path(path).read_text(encoding="utf-8")
        if path
        else (files("links2rag") / "taxonomy.txt").read_text(encoding="utf-8")
    )
    slugs = [line.strip() for line in text.splitlines()]
    return [s for s in slugs if s and not s.startswith("#")]


class Pipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        embedder: Embedder,
        qdrant: QdrantClient,
        client: LinksSource,
        llm: LLM | None = None,
        fetcher: FetcherLike | None = None,
        ocr: Callable[[bytes], str] | None = None,
        stop: threading.Event | None = None,
        beat: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.fetcher = fetcher
        self.db = db
        self.embedder = embedder
        self.client = client
        self.llm = llm
        self.stop = stop or threading.Event()
        self.qdrant = qdrant
        # Touched while work progresses, for the serve loop's health check.
        self.beat = beat or (lambda: None)
        self.taxonomy = load_taxonomy(settings.topic_taxonomy_file)
        self.member = QdrantStore(
            client=qdrant,
            collection=settings.member_collection,
            vector_size=settings.vector_size,
            distance=settings.distance,
            metadata={
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.vector_size,
                "embedding_document_prefix": settings.embedding_document_prefix,
                "embedding_query_prefix": settings.embedding_query_prefix,
                "sparse_encoder": SPARSE_ENCODER,
            },
            indexes=COMMON_INDEXES | LINKS_INDEXES,
        )
        self.linker = Linker(
            ((r["entity_type"], r["entity_id"], r["alias"]) for r in db.aliases()),
            self._confirm_link if llm is not None else None,
        )
        self.manuals = (
            ManualSync(
                settings=settings,
                db=db,
                store=self.member,
                fetcher=fetcher,
                embed=lambda store, items, force: self._embed_points(items, force=force),
                llm=llm,
                ocr=ocr,
                stop=self.stop,
                beat=self.beat,
            )
            if settings.manuals_enabled and fetcher is not None
            else None
        )

    def close(self) -> None:
        """Close the HTTP clients this pipeline opened; serve builds one per scan."""
        self.qdrant.close()
        for client in (self.embedder, self.llm, self.fetcher):
            closer = getattr(client, "close", None)
            if closer is not None:
                closer()

    def _confirm_link(self, text: str, alias: str, entity_type: str, entity_id: str) -> bool:
        if self.llm is None:
            return False
        self.beat()
        try:
            return self.llm.confirm_link(
                text=text, alias=alias, entity_type=entity_type, entity_id=entity_id
            )
        except LLMError as e:
            logger.warning("Link confirmation failed: %s", e)
            return False

    def _link(self, text: str) -> dict[str, list[str]]:
        """Linker hits for everything but assets: an asset's own IDs come from the
        asset manager, never from names mentioned in its text."""
        ids = self.linker.link(text).ids
        return {k: v for k, v in ids.items() if k != "asset_ids"}

    # -- run ------------------------------------------------------------------

    def run(self, mode: Mode, *, force: bool = False) -> RunResult:
        counts = RunCounts()
        with self.db.source_lock(RUN_SOURCE) as locked:
            if not locked:
                return RunResult("busy", counts, "another links2rag run holds the lock")
            run_id = f"{RUN_SOURCE}-{mode}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
            self.db.start_run(run_id=run_id, source=RUN_SOURCE, mode=mode)
            try:
                result = self._run(mode, run_id, counts, force=force or mode == "reindex")
            except Exception as e:
                # Record the failure on the status row, then let main() report it.
                counts.errors.append(f"{type(e).__name__}: {e}")
                self.db.conn.rollback()
                self.db.finish_run(run_id=run_id, counts=counts, ok=False, message=str(e))
                raise
            counts.quarantined = self.db.quarantine_count([ASSET_SOURCE, MANUAL_SOURCE])
            self.db.finish_run(
                run_id=run_id, counts=counts, ok=result.status == "ok", message=result.message
            )
            return result

    def _run(self, mode: Mode, run_id: str, counts: RunCounts, *, force: bool) -> RunResult:
        if mode == "reindex":
            counts.seen = len(self.db.staged_versions(ASSET_SOURCE))
            self.index_assets(None, run_id, counts, force=True)
            self._delete_stale(run_id)
            if self.manuals is not None:
                self.manuals.run(
                    run_id, counts, force=True, full=True, allow_delete=True, fetch=False
                )
            return RunResult("ok", counts)

        things = self.client.list_things()
        counts.seen = len(things)
        staged = self.db.staged_versions(ASSET_SOURCE)
        listed: set[str] = set()
        changed: list[str] = []
        for thing in things:
            if self.stop.is_set():
                break
            self.beat()
            key = str(thing.get("key") or thing.get("id"))
            listed.add(key)
            try:
                if self._stage_thing(thing, staged.get(key), run_id, counts, force=force):
                    changed.append(key)
                # One thing per transaction: a bad one rolls back only itself.
                self.db.commit()
            except Exception as e:
                self.db.conn.rollback()
                counts.errors.append(f"thing {key}: {type(e).__name__}: {e}")
                logger.exception("Failed to stage thing %s", key)
        self.db.commit()

        if self.stop.is_set():
            # Index what was staged; the next run sees the rest as changed.
            self._index_or_forget(changed, None, run_id, counts, force=force)
            return RunResult("ok", counts, "stopped early")

        gone = sorted(set(staged) - listed)
        previous = self.db.previous_full_run_seen(RUN_SOURCE, run_id)
        baseline = max(previous or 0, len(staged))
        status: Literal["ok", "valve"] = "ok"
        message = ""
        if gone and counts.seen < baseline * SAFETY_RATIO:
            status = "valve"
            message = (
                f"Saw {counts.seen} things but {baseline} were expected; not deleting "
                f"{len(gone)}. Check LINKS_URL and LINKS_API_KEY."
            )
            logger.error("Safety valve: %s", message)
        elif gone:
            self._delete_assets(gone, counts)

        todo = None if mode == "full" else changed
        self._index_or_forget(changed, todo, run_id, counts, force=force)
        self.propose_nicknames(todo)
        if mode == "full" and status == "ok":
            self._delete_stale(run_id)
        if self.manuals is not None and not self.stop.is_set():
            self.manuals.run(
                run_id, counts, force=force, full=mode == "full", allow_delete=status == "ok"
            )
        if mode == "full":
            report = reports.build(self.db)
            reports.write(report, self.settings.data_dir)
            logger.info("Report: %s", reports.summary(report))
        return RunResult(status, counts, message)

    def _index_or_forget(
        self,
        changed: list[str],
        todo: list[str] | None,
        run_id: str,
        counts: RunCounts,
        *,
        force: bool,
    ) -> None:
        """Index; if that fails, make the next run restage what changed."""
        try:
            self.index_assets(todo, run_id, counts, force=force)
        except Exception:
            self.db.conn.rollback()
            self.db.forget_versions(ASSET_SOURCE, changed)
            self.db.commit()
            raise

    def _delete_stale(self, run_id: str) -> None:
        removed = self.member.delete_stale(ASSET_SOURCE, run_id)
        if removed:
            logger.info("Deleted %d stale %s points", removed, ASSET_SOURCE)

    # -- staging ----------------------------------------------------------------

    def _stage_thing(
        self,
        thing: JSON,
        previous: tuple[str | None, str] | None,
        run_id: str,
        counts: RunCounts,
        *,
        force: bool,
    ) -> bool:
        """Stage one thing; True if it is new or changed (or force)."""
        digest = asset_docs.content_hash(thing)
        if previous is not None and previous[1] == digest and not force:
            return False
        asset = asset_docs.parse_thing(
            thing,
            space_names=self.settings.space_owner_names,
            geowiki_url=self.settings.geowiki_url,
        )
        reason = asset_docs.excluded(asset, self.settings.include_integration_sources)
        flags = dict(asset.flags)
        if reason:
            flags["excluded"] = reason
        if "secret_redacted" in flags:
            self.db.add_quarantine(
                source=ASSET_SOURCE,
                source_doc_id=asset.asset_id,
                reason="secret redacted",
                detail={"rules": flags["secret_redacted"], "name": asset.name},
                run_id=run_id,
            )
            logger.warning(
                "Redacted a secret in asset %s (%s); fix it in the asset manager",
                asset.asset_id,
                ", ".join(flags["secret_redacted"]),
            )
        self.db.upsert_staging(
            StagedDocument(
                source=ASSET_SOURCE,
                source_doc_id=asset.asset_id,
                title=asset.name,
                url=asset.url,
                tier=TIER,
                raw_content=thing,
                extracted_text="\n\n".join(
                    p for p in (asset_docs.facts(asset), asset_docs.body(asset)) if p
                ),
                content_hash=digest,
                source_updated_at=asset.updated_at,
                flags=flags,
                indexable=reason is None,
            ),
            run_id=run_id,
        )
        self.db.upsert_asset(asset_docs.registry_row(asset, indexable=reason is None))
        if reason is None:
            self.db.replace_asset_aliases(
                asset.asset_id,
                asset_docs.aliases(asset),
                keep=self.db.confirmed_nicknames(asset.asset_id),
            )
        else:
            self.db.delete_asset_aliases([asset.asset_id])
        if previous is None:
            counts.added += 1
        else:
            counts.updated += 1
        return True

    def _delete_assets(self, asset_ids: list[str], counts: RunCounts) -> None:
        self.member.delete_docs(ASSET_SOURCE, asset_ids)
        self.db.delete_staging(ASSET_SOURCE, asset_ids)
        self.db.delete_assets(asset_ids)
        self.db.delete_asset_aliases(asset_ids)
        self.db.commit()
        counts.deleted += len(asset_ids)
        logger.info("Deleted %d assets gone from the asset manager", len(asset_ids))

    # -- indexing ---------------------------------------------------------------

    def _parse_staged(self, row: JSON) -> asset_docs.Asset:
        return asset_docs.parse_thing(
            row["raw_content"],
            space_names=self.settings.space_owner_names,
            geowiki_url=self.settings.geowiki_url,
        )

    def _analysis(self, asset: asset_docs.Asset) -> tuple[str, list[str]]:
        """LLM summary and topics for an asset; empty without an LLM or on failure."""
        if self.llm is None:
            return "", []
        self.beat()
        try:
            result = self.llm.analyze_asset(
                facts=f"{asset_docs.facts(asset)}\n\n{asset_docs.body(asset)}",
                taxonomy=self.taxonomy,
            )
        except LLMError as e:
            logger.warning("Asset analysis failed for %s: %s", asset.asset_id, e)
            return "", []
        return result.description, result.topics

    def asset_points(self, row: JSON, run_id: str) -> list[tuple[str, JSON]]:
        asset = self._parse_staged(row)
        summary, topics = self._analysis(asset)
        texts = asset_docs.chunk_texts(
            asset,
            summary=summary,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        found = self._link(f"{asset.where}\n{asset_docs.body(asset)}")
        links = {
            **found,
            "asset_ids": [asset.asset_id],
            "location_ids": sorted(set(asset.location_ids) | set(found.get("location_ids", []))),
        }
        context = (
            f"Asset manager record for {asset.name}"
            + (f" ({asset.make_model})" if asset.make_model else "")
            + "."
        )
        items: list[tuple[str, JSON]] = []
        for index, text in enumerate(texts):
            payload = base_payload(
                source=ASSET_SOURCE,
                doc_id=asset.asset_id,
                chunk_index=index,
                title=asset.name,
                text=text,
                context=context,
                url=asset.url,
                run_id=run_id,
                created_at=asset.created_at,
                updated_at=asset.updated_at,
                heading_path=[c for c in (asset.category, asset.name) if c],
                lang="en",
                topics=topics,
                links=links,
            )
            payload.update(asset_docs.payload_extras(asset))
            items.append((point_id(ASSET_SOURCE, asset.asset_id, index), payload))
        return items

    def index_assets(
        self, asset_ids: list[str] | None, run_id: str, counts: RunCounts, *, force: bool
    ) -> None:
        """Index these staged assets (None: all of them), dropping leftover chunks."""
        if asset_ids == []:
            return
        rows = self.db.staged_documents(ASSET_SOURCE, asset_ids)
        items: list[tuple[str, JSON]] = []
        for row in rows:
            self.beat()
            try:
                items.extend(self.asset_points(row, run_id))
            except Exception as e:
                counts.errors.append(f"asset {row['source_doc_id']}: {type(e).__name__}: {e}")
                logger.exception("Failed to build points for asset %s", row["source_doc_id"])
        doc_ids = [r["source_doc_id"] for r in rows]
        keep = {pid for pid, _ in items}
        leftover = self.member.doc_point_ids(ASSET_SOURCE, doc_ids) - keep
        self.member.delete_ids(sorted(leftover))
        # Staged but no longer indexable (excluded since): remove their points.
        if asset_ids is not None:
            hidden = sorted(set(asset_ids) - set(doc_ids))
            self.member.delete_docs(ASSET_SOURCE, hidden)
        embedded = self._embed_points(items, force=force)
        logger.info("Indexed %d assets (%d chunks embedded)", len(rows), embedded)

    def propose_nicknames(self, asset_ids: list[str] | None) -> int:
        """Ask the LLM for nicknames members use, from wiki and Slack mentions.

        Proposals go to the review queue (``registry_review_queue``, kind
        ``asset_alias``); `links2rag aliases --approve ID` makes one an alias.
        Returns the number of new proposals.
        """
        if self.llm is None or asset_ids == []:
            return 0
        proposed = 0
        for row in self.db.staged_documents(ASSET_SOURCE, asset_ids):
            if self.stop.is_set():
                break
            self.beat()
            asset = self._parse_staged(row)
            try:
                hits = search(
                    self.qdrant,
                    self.settings.member_collection,
                    self.embedder,
                    " ".join(p for p in (asset.name, asset.make_model) if p),
                    limit=NICKNAME_MENTIONS,
                    sources=NICKNAME_SOURCES,
                )
                mentions = [str((h.payload or {}).get("text") or "") for h in hits]
                names = self.llm.propose_nicknames(
                    name=asset.name,
                    facts=asset_docs.facts(asset),
                    mentions=[m for m in mentions if m],
                )
            except LLMError as e:
                logger.warning("Nickname proposal failed for %s: %s", asset.asset_id, e)
                continue
            except Exception as e:
                logger.warning("Nickname pass stopped: %s", e)
                break
            sources = sorted({str((h.payload or {}).get("url")) for h in hits} - {"None"})
            known = {a.lower() for a in asset_docs.aliases(asset)}
            for name in names:
                if name.lower() in known:
                    continue
                if self.db.queue_alias(
                    asset.asset_id, name, {"proposed_by": "llm", "from": sources}
                ):
                    proposed += 1
            self.db.commit()
        if proposed:
            logger.info("Proposed %d asset nicknames for review", proposed)
        return proposed

    def _embed_points(self, items: list[tuple[str, JSON]], *, force: bool) -> int:
        """Upsert (point id, payload) items, embedding only changed text."""
        if not items:
            return 0
        store = self.member
        for _, payload in items:
            payload["payload_hash"] = payload_hash(payload)
        existing = {} if force else store.existing_state(i for i, _ in items)
        todo = [
            (i, p)
            for i, p in items
            if force or (existing.get(i) or (None,))[0] != p["content_hash"]
        ]
        same_text = [
            (i, p)
            for i, p in items
            if not force and i in existing and existing[i][0] == p["content_hash"]
        ]
        stale = [(i, p) for i, p in same_text if existing[i][1] != p["payload_hash"]]
        current = [i for i, p in same_text if existing[i][1] == p["payload_hash"]]
        store.replace_payloads(stale)
        if current:
            store.set_payload(current, {"sync_run_id": items[0][1]["sync_run_id"]})
        size = self.settings.batch_size
        for start in range(0, len(todo), size):
            self.beat()
            batch = todo[start : start + size]
            texts = [
                f"{p['context']}\n\n{p['text']}" if p["context"] else p["text"] for _, p in batch
            ]
            vectors = self.embedder.encode(texts, kind="document")
            store.upsert(
                [
                    qm.PointStruct(
                        id=pid,
                        vector={"dense": vector, "sparse": sparse_vector(text)},
                        payload=payload,
                    )
                    for (pid, payload), text, vector in zip(batch, texts, vectors, strict=True)
                ]
            )
        return len(todo)


def build_llm(settings: Settings, db: Database) -> LLM | None:
    if not settings.llm_model:
        logger.warning("LLM_MODEL is not set: no summaries, topics or chunk context this run")
        return None
    return LLM(
        model=settings.llm_model,
        base_url=settings.gateway_base_url,
        api_key=settings.gateway_api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        cache=db,
        vision_model=settings.vision_model,
    )


def build_pipeline(
    settings: Settings,
    db: Database,
    client: LinksSource,
    stop: threading.Event | None,
    beat: Callable[[], None] | None = None,
) -> Pipeline:
    return Pipeline(
        settings=settings,
        db=db,
        embedder=build_embedder(settings),
        qdrant=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        client=client,
        llm=build_llm(settings, db),
        fetcher=build_fetcher(settings) if settings.manuals_enabled else None,
        ocr=build_ocr(settings) if settings.manuals_enabled else None,
        stop=stop,
        beat=beat,
    )


def build_client(settings: Settings) -> LinksClient:
    return LinksClient(
        base_url=settings.links_url,
        api_key=settings.links_api_key,
        timeout_seconds=settings.request_timeout_seconds,
        page_size=settings.links_page_size,
    )


def run(
    settings: Settings,
    mode: Mode,
    stop: threading.Event | None = None,
    beat: Callable[[], None] | None = None,
    *,
    force: bool = False,
) -> RunResult:
    client = build_client(settings)
    try:
        with Database.connect(settings.database_url) as db:
            built = build_pipeline(settings, db, client, stop, beat)
            try:
                return built.run(mode, force=force)
            finally:
                built.close()
    finally:
        client.close()


def last_full_run(settings: Settings) -> datetime | None:
    """When the last full run finished: the serve loop's cue for the next one."""
    with Database.connect(settings.database_url) as db:
        return db.last_full_run_at(RUN_SOURCE)


def status(settings: Settings) -> JSON:
    """Recent runs, quarantine and asset counts: the source's status page."""
    with Database.connect(settings.database_url) as db:
        assets = db.assets()
        links = db.manual_links().values()
        return {
            "source": RUN_SOURCE,
            "runs": db.recent_runs(RUN_SOURCE),
            "quarantined": db.quarantine_count([ASSET_SOURCE, MANUAL_SOURCE]),
            "assets": sum(1 for a in assets if a["indexable"]),
            "assets_excluded": sum(1 for a in assets if not a["indexable"]),
            "aliases_pending_review": len(db.alias_queue()),
            "manuals": len(db.staged_versions(MANUAL_SOURCE)),
            "manual_links": len(links),
            "manual_links_failing": sum(1 for r in links if r["status"] in DEAD_STATUSES),
        }
