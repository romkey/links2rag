"""Manuals: links from assets -> downloads -> staged, parsed files -> hs_member points.

Every manual link on an indexable asset (and every drop-in file) gets a row
in ``asset_manual_links``. A link is downloaded when it is new, when its
last check is older than MANUAL_REFETCH_DAYS, or on a forced run. A link to
an HTML downloads page (or a manufacturer page) is followed one level, to
same-site PDFs. Files are deduplicated by SHA-256: one staged manual
(``source: manual``, ``doc_id``: the hash) per file, with every asset that
links to it in ``asset_ids``. A dead link keeps its last good file indexed.
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import pytesseract
from PIL import Image

from links2rag import assets as asset_docs
from links2rag.config import Settings
from links2rag.db import Database, RunCounts, StagedDocument
from links2rag.documents import MANUAL_SOURCE, TIER, base_payload, json_hash, point_id
from links2rag.fetch import (
    Fetched,
    Fetcher,
    is_downloads_page,
    normalize_url,
    pdf_links,
    sniff,
)
from links2rag.llm import LLM, LLMError
from links2rag.manuals import (
    PARSER_VERSION,
    chunk_manual,
    indexable_language,
    page_url,
    read_dropins,
    summary_text,
)
from links2rag.parse import ParsedManual, parse_html, parse_pdf
from links2rag.qdrant_store import QdrantStore
from links2rag.sanitize import redact

logger = logging.getLogger(__name__)

# Any: staged records and payloads are JSON.
JSON = dict[str, Any]
DROPIN_PREFIX = "dropin:"
# Fetch outcomes that leave a link worth reporting.
DEAD_STATUSES = frozenset({"dead", "error", "too_large", "unsupported", "blocked"})


class FetcherLike(Protocol):
    def blocked_reason(self, url: str) -> str | None: ...
    def fetch(self, url: str) -> Fetched: ...


# (store, items, force) -> number embedded.
EmbedPoints = Callable[[QdrantStore, list[tuple[str, JSON]], bool], int]


@dataclass
class Wanted:
    asset_ids: set[str] = field(default_factory=set)
    title: str | None = None
    origin: str = "link"


def redact_manual(parsed: ParsedManual) -> list[str]:
    """Redact secrets in a parsed manual in place; returns the rules that matched."""
    rules: set[str] = set()
    parsed.title = redact(parsed.title).text
    for section in parsed.sections:
        heading = redact(section.heading)
        section.heading = heading.text
        rules.update(heading.rules)
        for block in section.blocks:
            result = redact(block.text)
            block.text = result.text
            rules.update(result.rules)
    return sorted(rules)


def build_ocr(settings: Settings) -> Callable[[bytes], str] | None:
    """Tesseract OCR, or None when disabled or the binary is missing."""
    if not settings.ocr_enabled:
        return None
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        logger.warning("OCR_ENABLED but tesseract is not installed: scanned pages are skipped")
        return None

    def ocr(png: bytes) -> str:
        with Image.open(io.BytesIO(png)) as image:
            return str(pytesseract.image_to_string(image, lang=settings.ocr_language))

    return ocr


class ManualSync:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        store: QdrantStore,
        fetcher: FetcherLike,
        embed: EmbedPoints,
        llm: LLM | None = None,
        ocr: Callable[[bytes], str] | None = None,
        stop: threading.Event | None = None,
        beat: Callable[[], None] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.settings = settings
        self.db = db
        self.store = store
        self.fetcher = fetcher
        self.embed = embed
        self.llm = llm
        self.ocr = ocr
        self.stop = stop or threading.Event()
        self.beat = beat or (lambda: None)
        self.now = now
        self.files = Path(settings.data_dir) / "manuals"

    # -- links ----------------------------------------------------------------

    def wanted_links(self) -> dict[str, Wanted]:
        """Every manual URL the indexable assets link to, plus drop-ins."""
        pattern = self.settings.manual_pattern
        wanted: dict[str, Wanted] = {}
        for row in self.db.staged_documents("asset"):
            asset = asset_docs.parse_thing(
                row["raw_content"],
                space_names=self.settings.space_owner_names,
                geowiki_url=self.settings.geowiki_url,
            )
            for candidate in asset_docs.manual_candidates(
                asset,
                pattern,
                include_manufacturer_url=self.settings.manuals_from_manufacturer_url,
            ):
                url = normalize_url(candidate.url)
                entry = wanted.setdefault(
                    url, Wanted(title=candidate.title, origin=candidate.origin)
                )
                entry.asset_ids.add(asset.asset_id)
                if candidate.origin == "link":
                    entry.origin = "link"
                    entry.title = candidate.title or entry.title
        for dropin in read_dropins(self.settings.manuals_dir):
            key = DROPIN_PREFIX + str(dropin.path)
            wanted[key] = Wanted(set(dropin.asset_keys), dropin.title or dropin.path.stem, "dropin")
        return wanted

    def _due(self, row: JSON | None, force: bool) -> bool:
        if force or row is None or row["status"] == "pending" or row["last_checked_at"] is None:
            return True
        return bool(
            self.now() - row["last_checked_at"] >= timedelta(days=self.settings.manual_refetch_days)
        )

    def _store_file(self, content: bytes, kind: str) -> str:
        digest = sha256(content).hexdigest()
        self.files.mkdir(parents=True, exist_ok=True)
        path = self.files / f"{digest}.{kind}"
        if not path.exists():
            tmp = path.with_suffix(".part")
            tmp.write_bytes(content)
            tmp.replace(path)
        return digest

    def _fetch_dropin(self, url: str) -> None:
        path = Path(url.removeprefix(DROPIN_PREFIX))
        try:
            content = path.read_bytes()
        except OSError as e:
            self.db.record_fetch(url, status="error", error=str(e))
            return
        kind = sniff(content, None)
        if kind == "other":
            self.db.record_fetch(url, status="unsupported", error="not a PDF or HTML file")
            return
        digest = self._store_file(content, kind)
        self.db.record_fetch(url, status="ok", manual_id=digest, content_type=kind)

    def _fetch(self, url: str, wanted: Wanted, *, follow: bool) -> list[str]:
        """Fetch one link; returns the same-site PDFs to fetch next (downloads pages)."""
        self.beat()
        if url.startswith(DROPIN_PREFIX):
            self._fetch_dropin(url)
            return []
        result = self.fetcher.fetch(url)
        if result.status != "ok":
            logger.info("Manual link %s: %s (%s)", url, result.status, result.error)
            self.db.record_fetch(
                url,
                status=result.status,
                http_status=result.http_status,
                error=result.error,
                final_url=result.final_url,
            )
            return []
        if result.kind == "html":
            base = result.final_url or url
            manufacturer = wanted.origin == "manufacturer"
            links = pdf_links(
                result.content,
                base,
                pattern_match=lambda text: bool(self.settings.manual_pattern.search(text)),
                require_match=manufacturer,
            )
            if follow and (manufacturer or is_downloads_page(result.content, links)):
                self.db.record_fetch(
                    url,
                    status="index",
                    http_status=result.http_status,
                    content_type="html",
                    final_url=result.final_url,
                )
                return [link for link, _ in links]
            if manufacturer or not follow:
                # A product page with no manuals, or a page found on a downloads page.
                self.db.record_fetch(
                    url,
                    status="unsupported",
                    http_status=result.http_status,
                    error="HTML page, not a manual",
                    final_url=result.final_url,
                )
                return []
        digest = self._store_file(result.content, result.kind or "pdf")
        self.db.record_fetch(
            url,
            status="ok",
            http_status=result.http_status,
            manual_id=digest,
            content_type=result.kind,
            final_url=result.final_url,
        )
        return []

    def sync_links(self, *, force: bool, allow_delete: bool) -> int:
        """Bring asset_manual_links up to date and fetch what is due; returns fetches.

        A file found on downloads or manufacturer pages records every page it
        was on (``parent_urls``), and belongs to the assets of all of them that
        are still linked, as well as to any asset that links to it directly.
        """
        wanted = self.wanted_links()
        existing = self.db.manual_links()
        parents: dict[str, set[str]] = {
            url: set(row["parent_urls"]) for url, row in existing.items() if row["parent_urls"]
        }
        # Rows first: fetching records its outcome on the link's row.
        for url, entry in wanted.items():
            self.db.upsert_manual_link(
                url=url,
                asset_ids=sorted(entry.asset_ids),
                title=entry.title,
                origin=entry.origin,
                parent_urls=sorted(parents.get(url, set())),
            )
        self.db.commit()
        fetched = 0
        for url in sorted(wanted):
            if self.stop.is_set():
                break
            if not self._due(existing.get(url), force):
                continue
            children = self._fetch(url, wanted[url], follow=True)
            fetched += 1
            self.db.commit()
            # The page now lists exactly these files.
            for child, pages in parents.items():
                if child not in children:
                    pages.discard(url)
            for child in children:
                parents.setdefault(child, set()).add(url)

        links: dict[str, tuple[set[str], str | None, str, set[str]]] = {}
        for url, entry in wanted.items():
            links[url] = (set(entry.asset_ids), entry.title, entry.origin, set())
        for child, pages in parents.items():
            live = {p for p in pages if p in wanted}
            if not live:
                continue
            assets = set().union(*(wanted[p].asset_ids for p in live))
            title = next((wanted[p].title for p in sorted(live) if wanted[p].title), None)
            if child in links:
                own, own_title, origin, _ = links[child]
                links[child] = (own | assets, own_title or title, origin, live)
            else:
                links[child] = (assets, title, "discovered", live)

        gone = sorted(set(existing) - set(links))
        if gone and allow_delete:
            self.db.delete_manual_links(gone)
        for url, (assets, title, origin, pages) in links.items():
            self.db.upsert_manual_link(
                url=url,
                asset_ids=sorted(assets),
                title=title,
                origin=origin,
                parent_urls=sorted(pages),
            )
        self.db.commit()

        for url, (assets, title, origin, _) in sorted(links.items()):
            if self.stop.is_set() or origin != "discovered":
                continue
            if self._due(existing.get(url), force):
                self._fetch(url, Wanted(assets, title, origin), follow=False)
                fetched += 1
                self.db.commit()
        return fetched

    # -- staging ----------------------------------------------------------------

    def _groups(self) -> dict[str, JSON]:
        """manual_id -> what the links say about that file."""
        groups: dict[str, JSON] = {}
        for url, row in sorted(self.db.manual_links().items()):
            if not row["manual_id"]:
                continue
            group = groups.setdefault(
                row["manual_id"],
                {"asset_ids": set(), "urls": [], "titles": [], "kind": row["content_type"]},
            )
            group["asset_ids"].update(row["asset_ids"])
            group["titles"].extend([row["title"]] if row["title"] else [])
            group["urls"].append(
                {
                    "url": None if url.startswith(DROPIN_PREFIX) else url,
                    "origin": row["origin"],
                    "status": row["status"],
                    "final_url": row["final_url"],
                }
            )
        return groups

    def _parse(self, manual_id: str, kind: str, title: str) -> ParsedManual:
        path = self.files / f"{manual_id}.{kind}"
        data = path.read_bytes()
        if kind == "html":
            return parse_html(data, title=title)
        describe = None
        if self.llm is not None and self.llm.vision_model:
            llm = self.llm

            def describe(png: bytes, context: str) -> str | None:
                self.beat()
                return llm.describe_image(png=png, context=f"{title}, {context}")

        return parse_pdf(data, title=title, ocr=self.ocr, describe=describe)

    def stage(
        self, run_id: str, counts: RunCounts, *, force: bool, allow_delete: bool
    ) -> list[str]:
        """Stage every manual file the links point to; returns the changed manual IDs."""
        groups = self._groups()
        staged = self.db.staged_versions(MANUAL_SOURCE)
        changed: list[str] = []
        for manual_id, group in groups.items():
            if self.stop.is_set():
                break
            self.beat()
            asset_ids = sorted(group["asset_ids"])
            kind = group["kind"] or "pdf"
            mapping = {
                "asset_ids": asset_ids,
                "urls": group["urls"],
                "titles": sorted(set(group["titles"])),
                "parser": PARSER_VERSION,
            }
            digest = json_hash({"file": manual_id, **mapping})
            previous = staged.get(manual_id)
            if previous is not None and previous[1] == digest and not force:
                continue
            row = self.db.staged_document(MANUAL_SOURCE, manual_id) if previous else None
            reparse = (
                force
                or row is None
                or (row["raw_content"].get("parser") != PARSER_VERSION)
                or "parsed" not in row["raw_content"]
            )
            title = mapping["titles"][0] if mapping["titles"] else manual_id[:12]
            try:
                parsed = (
                    self._parse(manual_id, kind, title)
                    if reparse
                    else ParsedManual.from_json(row["raw_content"]["parsed"])  # type: ignore[index]
                )
            except Exception as e:
                self.db.conn.rollback()
                counts.errors.append(f"manual {manual_id[:12]}: {type(e).__name__}: {e}")
                logger.exception("Failed to parse manual %s", manual_id)
                continue
            # Redact before staging: nothing downstream (staging, the LLM's
            # context prompt, the index) sees a secret. The file stays as-is.
            rules = redact_manual(parsed)
            flags: JSON = {}
            if rules:
                flags["secret_redacted"] = rules
                self.db.add_quarantine(
                    source=MANUAL_SOURCE,
                    source_doc_id=manual_id,
                    reason="secret redacted",
                    detail={"rules": rules, "title": title},
                    run_id=run_id,
                )
            urls = [u["url"] for u in group["urls"] if u["url"]]
            self.db.upsert_staging(
                StagedDocument(
                    source=MANUAL_SOURCE,
                    source_doc_id=manual_id,
                    title=parsed.title or title,
                    url=urls[0] if urls else None,
                    tier=TIER,
                    raw_content={
                        **mapping,
                        "kind": kind,
                        "file": f"manuals/{manual_id}.{kind}",
                        "parsed": parsed.to_json(),
                    },
                    extracted_text=parsed.text,
                    content_hash=digest,
                    source_updated_at=None,
                    flags=flags,
                    indexable=any(indexable_language(s.lang) for s in parsed.sections),
                ),
                run_id=run_id,
            )
            self.db.commit()
            changed.append(manual_id)
            if previous is None:
                counts.added += 1
            else:
                counts.updated += 1
        orphans = sorted(set(staged) - set(groups))
        if orphans and allow_delete:
            self.store.delete_docs(MANUAL_SOURCE, orphans)
            self.db.delete_staging(MANUAL_SOURCE, orphans)
            self.db.commit()
            for manual_id in orphans:
                for path in self.files.glob(f"{manual_id}.*"):
                    path.unlink(missing_ok=True)
            counts.deleted += len(orphans)
            logger.info("Deleted %d manuals no asset links to any more", len(orphans))
        return changed

    # -- indexing -----------------------------------------------------------------

    def _asset_names(self, asset_ids: list[str]) -> list[str]:
        """'Name (make model)' for each asset, as the registry has it."""
        names: dict[str, str] = {}
        for r in self.db.assets():
            make_model = " ".join(p for p in (r["manufacturer"], r["model"]) if p)
            names[r["asset_id"]] = f"{r['name']} ({make_model})" if make_model else r["name"]
        return [names[a] for a in asset_ids if a in names]

    def _context(self, title: str, document: str, chunk: str, fallback: str) -> str:
        if self.llm is None:
            return fallback
        self.beat()
        try:
            return self.llm.chunk_context(document_title=title, document=document, chunk=chunk)
        except LLMError as e:
            logger.warning("Chunk context failed: %s", e)
            return fallback

    def _section(self, heading: str, text: str, kind: str | None) -> str | None:
        if kind is not None or self.llm is None or not heading:
            return kind
        self.beat()
        try:
            return self.llm.manual_section(heading=heading, text=text)
        except LLMError as e:
            logger.warning("Section classification failed: %s", e)
            return None

    def manual_points(self, row: JSON, run_id: str) -> list[tuple[str, JSON]]:
        raw = row["raw_content"]
        parsed = ParsedManual.from_json(raw["parsed"])
        kind = str(raw.get("kind") or "pdf")
        manual_id = row["source_doc_id"]
        asset_ids = list(raw.get("asset_ids") or [])
        title = parsed.title or row["title"] or manual_id[:12]
        urls = raw.get("urls") or []
        live = [u for u in urls if u.get("url") and u.get("status") == "ok"]
        any_url = [u for u in urls if u.get("url")]
        best = (live or any_url or [{}])[0]
        url = best.get("final_url") or best.get("url")
        link_dead = bool(any_url) and not live
        names = self._asset_names(asset_ids)
        document = parsed.text
        chunks = chunk_manual(
            parsed,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        extras = {
            "manual_kind": kind,
            "manual_urls": [u["url"] for u in any_url],
            "local_file": raw.get("file"),
            "link_dead": link_dead,
            "asset_names": names,
        }
        for_assets = f" for {'; '.join(names)}" if names else ""
        items: list[tuple[str, JSON]] = []
        summary = summary_text(parsed, title=title, assets=names)
        payload = base_payload(
            source=MANUAL_SOURCE,
            doc_id=manual_id,
            chunk_index=0,
            title=title,
            text=redact(summary).text,
            context=f"Overview of the manual {title}{for_assets}.",
            url=url,
            run_id=run_id,
            created_at=None,
            updated_at=None,
            heading_path=[title],
            lang="en",
            links={"asset_ids": asset_ids},
        )
        payload.update(extras, manual_section=None, block_kind="summary", machine_generated=False)
        items.append((point_id(MANUAL_SOURCE, manual_id, 0), payload))
        for index, chunk in enumerate(chunks, start=1):
            text = redact(chunk.text).text
            heading = chunk.heading_path[-1] if len(chunk.heading_path) > 1 else ""
            where = ", ".join(
                p
                for p in (heading and f"section {heading}", chunk.page and f"page {chunk.page}")
                if p
            )
            fallback = (
                f"From the manual {title}{for_assets}" + (f", {where}" if where else "") + "."
            )
            payload = base_payload(
                source=MANUAL_SOURCE,
                doc_id=manual_id,
                chunk_index=index,
                title=title,
                text=text,
                context=self._context(title, document, text, fallback),
                url=page_url(url, chunk.page, kind),
                run_id=run_id,
                created_at=None,
                updated_at=None,
                heading_path=chunk.heading_path,
                page=chunk.page,
                lang=chunk.lang or "en",
                links={"asset_ids": asset_ids},
            )
            payload.update(
                extras,
                manual_section=self._section(heading, text, chunk.section),
                block_kind=chunk.kind,
                machine_generated=chunk.machine_generated,
            )
            items.append((point_id(MANUAL_SOURCE, manual_id, index), payload))
        return items

    def index(
        self, manual_ids: list[str] | None, run_id: str, counts: RunCounts, *, force: bool
    ) -> None:
        if manual_ids == []:
            return
        rows = self.db.staged_documents(MANUAL_SOURCE, manual_ids)
        items: list[tuple[str, JSON]] = []
        for row in rows:
            self.beat()
            try:
                items.extend(self.manual_points(row, run_id))
            except Exception as e:
                counts.errors.append(f"manual {row['source_doc_id'][:12]}: {type(e).__name__}: {e}")
                logger.exception("Failed to build points for manual %s", row["source_doc_id"])
        doc_ids = [r["source_doc_id"] for r in rows]
        keep = {pid for pid, _ in items}
        self.store.delete_ids(sorted(self.store.doc_point_ids(MANUAL_SOURCE, doc_ids) - keep))
        if manual_ids is not None:
            self.store.delete_docs(MANUAL_SOURCE, sorted(set(manual_ids) - set(doc_ids)))
        embedded = self.embed(self.store, items, force)
        logger.info("Indexed %d manuals (%d chunks embedded)", len(rows), embedded)

    def run(
        self,
        run_id: str,
        counts: RunCounts,
        *,
        force: bool,
        full: bool,
        allow_delete: bool,
        fetch: bool = True,
    ) -> None:
        if fetch:
            fetched = self.sync_links(force=force, allow_delete=allow_delete)
            logger.info("Checked %d manual links", fetched)
        changed = self.stage(run_id, counts, force=force, allow_delete=allow_delete)
        self.index(None if full else changed, run_id, counts, force=force)
        if full and allow_delete:
            removed = self.store.delete_stale(MANUAL_SOURCE, run_id)
            if removed:
                logger.info("Deleted %d stale manual points", removed)


def build_fetcher(settings: Settings) -> Fetcher:
    return Fetcher(
        skip_hosts=settings.never_fetch_hosts,
        allow_private=settings.allow_private_hosts,
        max_bytes=settings.manual_max_mb * 1024 * 1024,
        timeout_seconds=settings.request_timeout_seconds,
    )
