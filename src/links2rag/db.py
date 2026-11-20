"""PostgreSQL: staging store, entity registry, manual links, run status.

Plain SQL with psycopg 3. The schema is in ``migrations/`` and is applied by
``migrate()``, which every command calls first.
"""

from __future__ import annotations

import contextlib
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any

import psycopg
from alembic import command
from alembic.config import Config
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# Any: rows are dicts of column name to whatever type the column holds.
Row = dict[str, Any]


def sqlalchemy_url(database_url: str) -> str:
    """DATABASE_URL as SQLAlchemy wants it, using the psycopg 3 driver."""
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg://" + database_url[len(prefix) :]
    return database_url


def migrate(database_url: str) -> None:
    """Bring the schema up to date (``alembic upgrade head``)."""
    config = Config()
    config.set_main_option("script_location", str(files("links2rag") / "migrations"))
    # Alembic's config uses %-interpolation; a password may contain %.
    config.set_main_option("sqlalchemy.url", sqlalchemy_url(database_url).replace("%", "%%"))
    command.upgrade(config, "head")


@dataclass(frozen=True)
class StagedDocument:
    source: str
    source_doc_id: str
    title: str | None
    url: str | None
    tier: str
    # Any: normalized source content, stored as JSON.
    raw_content: dict[str, Any]
    extracted_text: str
    content_hash: str
    source_updated_at: str | None
    # Any: flags are free-form JSON (e.g. {"secret_redacted": true}).
    flags: dict[str, Any] = field(default_factory=dict)
    indexable: bool = True


@dataclass
class RunCounts:
    seen: int = 0
    added: int = 0
    updated: int = 0
    deleted: int = 0
    quarantined: int = 0
    errors: list[str] = field(default_factory=list)


class Database:
    def __init__(self, conn: psycopg.Connection[Row]) -> None:
        self.conn = conn

    @classmethod
    @contextlib.contextmanager
    def connect(cls, database_url: str) -> Iterator[Database]:
        with psycopg.connect(database_url, row_factory=dict_row, connect_timeout=30) as conn:
            yield cls(conn)

    def commit(self) -> None:
        self.conn.commit()

    def _all(self, query: str, params: Iterable[object] | dict[str, object] = ()) -> list[Row]:
        with self.conn.cursor() as cur:
            cur.execute(query, params)  # type: ignore[arg-type]  # psycopg types params loosely
            return cur.fetchall()

    def _one(self, query: str, params: Iterable[object] | dict[str, object] = ()) -> Row | None:
        rows = self._all(query, params)
        return rows[0] if rows else None

    def _exec(self, query: str, params: Iterable[object] | dict[str, object] = ()) -> int:
        with self.conn.cursor() as cur:
            cur.execute(query, params)  # type: ignore[arg-type]  # psycopg types params loosely
            return cur.rowcount

    # -- locks and runs -----------------------------------------------------

    @contextlib.contextmanager
    def source_lock(self, source: str) -> Iterator[bool]:
        """A session advisory lock for one source; yields False if it's held."""
        key = zlib.crc32(f"links2rag:{source}".encode())
        row = self._one("SELECT pg_try_advisory_lock(%s) AS locked", (key,))
        self.commit()
        locked = bool(row and row["locked"])
        try:
            yield locked
        finally:
            if locked:
                self._exec("SELECT pg_advisory_unlock(%s)", (key,))
                self.commit()

    def start_run(self, *, run_id: str, source: str, mode: str) -> None:
        self._exec(
            "INSERT INTO sync_runs (run_id, source, mode, started_at) VALUES (%s, %s, %s, %s)",
            (run_id, source, mode, datetime.now(UTC)),
        )
        self.commit()

    def finish_run(
        self, *, run_id: str, counts: RunCounts, ok: bool, message: str | None = None
    ) -> None:
        self._exec(
            """
            UPDATE sync_runs SET finished_at = %s, ok = %s, seen = %s, added = %s,
                updated = %s, deleted = %s, quarantined = %s, errors = %s, message = %s
            WHERE run_id = %s
            """,
            (
                datetime.now(UTC),
                ok,
                counts.seen,
                counts.added,
                counts.updated,
                counts.deleted,
                counts.quarantined,
                Jsonb(counts.errors[:100]),
                message,
                run_id,
            ),
        )
        self.commit()

    def previous_full_run_seen(self, source: str, exclude_run_id: str) -> int | None:
        row = self._one(
            """
            SELECT seen FROM sync_runs
            WHERE source = %s AND mode = 'full' AND ok AND run_id <> %s
            ORDER BY started_at DESC LIMIT 1
            """,
            (source, exclude_run_id),
        )
        return int(row["seen"]) if row else None

    def last_full_run_at(self, source: str) -> datetime | None:
        """When the source's last full run finished, whatever its outcome."""
        row = self._one(
            "SELECT max(finished_at) AS at FROM sync_runs WHERE source = %s AND mode = 'full'",
            (source,),
        )
        return row["at"] if row else None

    def recent_runs(self, source: str, limit: int = 10) -> list[Row]:
        return self._all(
            "SELECT * FROM sync_runs WHERE source = %s ORDER BY started_at DESC LIMIT %s",
            (source, limit),
        )

    def quarantine_count(self, sources: Iterable[str]) -> int:
        row = self._one(
            "SELECT count(*) AS n FROM quarantine WHERE source = ANY(%s)", (list(sources),)
        )
        return int(row["n"]) if row else 0

    def quarantine_rows(self, sources: Iterable[str], limit: int = 200) -> list[Row]:
        return self._all(
            "SELECT * FROM quarantine WHERE source = ANY(%s) ORDER BY created_at DESC LIMIT %s",
            (list(sources), limit),
        )

    def add_quarantine(
        # Any: detail is free-form JSON.
        self,
        *,
        source: str,
        source_doc_id: str,
        reason: str,
        detail: dict[str, Any],
        run_id: str,
    ) -> None:
        self._exec(
            "INSERT INTO quarantine (source, source_doc_id, reason, detail, sync_run_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (source, source_doc_id, reason, Jsonb(detail), run_id),
        )

    # -- staging ------------------------------------------------------------

    def staged_versions(self, source: str) -> dict[str, tuple[str | None, str]]:
        """source_doc_id -> (source_updated_at, content_hash) for one source."""
        rows = self._all(
            "SELECT source_doc_id, source_updated_at, content_hash FROM staging_documents "
            "WHERE source = %s",
            (source,),
        )
        return {
            row["source_doc_id"]: (row["source_updated_at"], row["content_hash"]) for row in rows
        }

    def upsert_staging(self, doc: StagedDocument, *, run_id: str) -> None:
        self._exec(
            """
            INSERT INTO staging_documents (source, source_doc_id, title, url, tier, raw_content,
                extracted_text, content_hash, flags, indexable, source_updated_at, fetched_at,
                sync_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, source_doc_id) DO UPDATE SET
                title = EXCLUDED.title, url = EXCLUDED.url, tier = EXCLUDED.tier,
                raw_content = EXCLUDED.raw_content, extracted_text = EXCLUDED.extracted_text,
                content_hash = EXCLUDED.content_hash, flags = EXCLUDED.flags,
                indexable = EXCLUDED.indexable, source_updated_at = EXCLUDED.source_updated_at,
                fetched_at = EXCLUDED.fetched_at, sync_run_id = EXCLUDED.sync_run_id
            """,
            (
                doc.source,
                doc.source_doc_id,
                doc.title,
                doc.url,
                doc.tier,
                Jsonb(doc.raw_content),
                doc.extracted_text,
                doc.content_hash,
                Jsonb(doc.flags),
                doc.indexable,
                doc.source_updated_at,
                datetime.now(UTC),
                run_id,
            ),
        )

    def forget_versions(self, source: str, doc_ids: list[str]) -> None:
        """Make the next incremental run fetch these documents again (indexing failed)."""
        self._exec(
            "UPDATE staging_documents SET source_updated_at = NULL "
            "WHERE source = %s AND source_doc_id = ANY(%s)",
            (source, doc_ids),
        )

    def delete_staging(self, source: str, doc_ids: list[str]) -> int:
        return self._exec(
            "DELETE FROM staging_documents WHERE source = %s AND source_doc_id = ANY(%s)",
            (source, doc_ids),
        )

    def staged_documents(self, source: str, doc_ids: list[str] | None = None) -> list[Row]:
        """Indexable staged documents of one source (all, or these ids)."""
        if doc_ids is None:
            return self._all(
                "SELECT * FROM staging_documents WHERE source = %s AND indexable "
                "ORDER BY source_doc_id",
                (source,),
            )
        return self._all(
            "SELECT * FROM staging_documents WHERE source = %s AND indexable "
            "AND source_doc_id = ANY(%s) ORDER BY source_doc_id",
            (source, doc_ids),
        )

    def staged_document(self, source: str, doc_id: str) -> Row | None:
        """One staged document, indexable or not."""
        return self._one(
            "SELECT * FROM staging_documents WHERE source = %s AND source_doc_id = %s",
            (source, doc_id),
        )

    # -- caches ---------------------------------------------------------------

    # Any: cached outputs are JSON.
    def llm_cache_get(self, task: str, model: str, input_hash: str) -> Any:
        row = self._one(
            "SELECT output FROM llm_cache WHERE task = %s AND model = %s AND input_hash = %s",
            (task, model, input_hash),
        )
        return row["output"] if row else None

    # Any: cached outputs are JSON.
    def llm_cache_put(self, task: str, model: str, input_hash: str, output: Any) -> None:
        self._exec(
            """
            INSERT INTO llm_cache (task, model, input_hash, output) VALUES (%s, %s, %s, %s)
            ON CONFLICT (task, model, input_hash) DO UPDATE SET output = EXCLUDED.output,
                created_at = now()
            """,
            (task, model, input_hash, Jsonb(output)),
        )

    # -- entity registry --------------------------------------------------------

    def aliases(self) -> list[Row]:
        """Every entity alias, for the linker."""
        return self._all("SELECT entity_type, entity_id, alias FROM entity_aliases")

    def replace_asset_aliases(self, asset_id: str, aliases: Iterable[str], keep: set[str]) -> None:
        """Set an asset's generated aliases; aliases in ``keep`` (human-confirmed) stay."""
        wanted = {a for a in aliases if a}
        self._exec(
            "DELETE FROM entity_aliases WHERE entity_type = 'asset' AND entity_id = %s "
            "AND NOT (alias = ANY(%s))",
            (asset_id, sorted(wanted | keep)),
        )
        for alias in sorted(wanted):
            self._exec(
                "INSERT INTO entity_aliases (entity_type, entity_id, alias) "
                "VALUES ('asset', %s, %s) ON CONFLICT DO NOTHING",
                (asset_id, alias),
            )

    def confirmed_nicknames(self, asset_id: str) -> set[str]:
        """Nicknames a human approved for an asset; regenerating aliases keeps them."""
        rows = self._all(
            "SELECT right_id FROM registry_review_queue WHERE kind = 'asset_alias' "
            "AND left_id = %s AND details->>'decision' = 'approved'",
            (asset_id,),
        )
        return {row["right_id"] for row in rows}

    def delete_asset_aliases(self, asset_ids: list[str]) -> None:
        self._exec(
            "DELETE FROM entity_aliases WHERE entity_type = 'asset' AND entity_id = ANY(%s)",
            (asset_ids,),
        )

    # Any: details are free-form JSON.
    def queue_alias(self, asset_id: str, alias: str, details: dict[str, Any]) -> bool:
        """Queue a proposed nickname for review; False if it was already queued or decided."""
        return (
            self._exec(
                "INSERT INTO registry_review_queue (kind, left_id, right_id, details) "
                "VALUES ('asset_alias', %s, %s, %s) ON CONFLICT DO NOTHING",
                (asset_id, alias, Jsonb(details)),
            )
            > 0
        )

    def alias_queue(self, *, pending_only: bool = True) -> list[Row]:
        where = "kind = 'asset_alias'" + (" AND resolved_at IS NULL" if pending_only else "")
        return self._all(
            f"SELECT * FROM registry_review_queue WHERE {where} ORDER BY id"  # noqa: S608 - fixed SQL
        )

    def resolve_alias(self, queue_id: int, *, approve: bool) -> Row | None:
        """Approve (adds the alias) or reject a queued nickname."""
        row = self._one(
            "UPDATE registry_review_queue SET resolved_at = now(), "
            "details = details || jsonb_build_object('decision', %s::text) "
            "WHERE id = %s AND kind = 'asset_alias' RETURNING *",
            ("approved" if approve else "rejected", queue_id),
        )
        if row and approve:
            self._exec(
                "INSERT INTO entity_aliases (entity_type, entity_id, alias) "
                "VALUES ('asset', %s, %s) ON CONFLICT DO NOTHING",
                (row["left_id"], row["right_id"]),
            )
        self.commit()
        return row

    # Any: the asset row is JSON-ready values keyed by column.
    def upsert_asset(self, asset: dict[str, Any]) -> None:
        self._exec(
            """
            INSERT INTO registry_assets (asset_id, thing_id, slug, name, label_name, manufacturer,
                model, category, ownership, location_ids, url, short_url, retired_ids,
                related_asset_ids, links, indexable, updated_at)
            VALUES (%(asset_id)s, %(thing_id)s, %(slug)s, %(name)s, %(label_name)s,
                %(manufacturer)s, %(model)s, %(category)s, %(ownership)s, %(location_ids)s,
                %(url)s, %(short_url)s, %(retired_ids)s, %(related_asset_ids)s, %(links)s,
                %(indexable)s, now())
            ON CONFLICT (asset_id) DO UPDATE SET thing_id = EXCLUDED.thing_id,
                slug = EXCLUDED.slug, name = EXCLUDED.name, label_name = EXCLUDED.label_name,
                manufacturer = EXCLUDED.manufacturer, model = EXCLUDED.model,
                category = EXCLUDED.category, ownership = EXCLUDED.ownership,
                location_ids = EXCLUDED.location_ids, url = EXCLUDED.url,
                short_url = EXCLUDED.short_url, retired_ids = EXCLUDED.retired_ids,
                related_asset_ids = EXCLUDED.related_asset_ids, links = EXCLUDED.links,
                indexable = EXCLUDED.indexable, updated_at = now()
            """,
            {key: Jsonb(value) if key == "links" else value for key, value in asset.items()},
        )

    def delete_assets(self, asset_ids: list[str]) -> None:
        self._exec("DELETE FROM registry_assets WHERE asset_id = ANY(%s)", (asset_ids,))

    def assets(self) -> list[Row]:
        return self._all("SELECT * FROM registry_assets ORDER BY asset_id")

    # -- manual links -----------------------------------------------------------

    def manual_links(self) -> dict[str, Row]:
        return {row["url"]: row for row in self._all("SELECT * FROM asset_manual_links")}

    def upsert_manual_link(
        self,
        *,
        url: str,
        asset_ids: list[str],
        title: str | None,
        origin: str,
        parent_urls: list[str] | None = None,
    ) -> None:
        """Insert or replace a link. ``parent_urls``: the pages a discovered file was on."""
        self._exec(
            """
            INSERT INTO asset_manual_links (url, asset_ids, title, origin, parent_urls)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET asset_ids = EXCLUDED.asset_ids,
                title = EXCLUDED.title, origin = EXCLUDED.origin,
                parent_urls = EXCLUDED.parent_urls
            """,
            (url, sorted(set(asset_ids)), title, origin, sorted(set(parent_urls or []))),
        )

    def record_fetch(
        self,
        url: str,
        *,
        status: str,
        http_status: int | None = None,
        error: str | None = None,
        manual_id: str | None = None,
        content_type: str | None = None,
        final_url: str | None = None,
    ) -> None:
        """Record a fetch attempt; a failure keeps the last good file (manual_id)."""
        ok = status == "ok"
        self._exec(
            """
            UPDATE asset_manual_links SET status = %s, http_status = %s, error = %s,
                manual_id = CASE WHEN %s THEN %s ELSE manual_id END,
                content_type = COALESCE(%s, content_type), final_url = COALESCE(%s, final_url),
                last_checked_at = now(),
                last_ok_at = CASE WHEN %s THEN now() ELSE last_ok_at END
            WHERE url = %s
            """,
            (status, http_status, error, ok, manual_id, content_type, final_url, ok, url),
        )

    def delete_manual_links(self, urls: list[str]) -> None:
        self._exec("DELETE FROM asset_manual_links WHERE url = ANY(%s)", (urls,))
