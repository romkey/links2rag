# links2rag overview

**What:** the asset and equipment-manual source of the PDX Hackerspace RAG
pipeline (plans 00 "shared architecture" and 06 "asset manager and
equipment manuals"). Reads the asset manager
([pdxhackerspace-links](https://github.com/romkey/pdxhackerspace-links)),
which is the source of truth for `asset_id`. Member tier only: taken
together, asset data is an inventory of what is worth stealing, so nothing
goes to `hs_public`.
**Runs as:** a long-running worker with no port. The default command,
`serve`, scans every `SCAN_INTERVAL_MINUTES` (default 60): incremental
scans, plus a full reconcile when none has finished in
`FULL_SCAN_INTERVAL_HOURS` (default 24), so the first scan is full. It
touches a heartbeat file that `healthcheck` checks. `run --mode
incremental|full|reindex [--force]` does one scan and exits; `status` prints
recent runs as JSON. Every command except `healthcheck` first applies
pending schema migrations.
**Talks to:** the asset manager's read-only JSON API (a bearer API key),
PostgreSQL and Qdrant (both shared with the pipeline's other sources), and
an OpenAI-compatible gateway (embeddings and generation).

## Flow

1. `GET /api/v1/things`, every page. The list returns whole records, so
   every scan sees every thing; a scan costs a few requests.
2. For each thing whose content hash changed (scan counts and integration
   records don't count): scrub free text (secret scanner with a quarantine
   entry, then serial number, owner name, IPs, MACs, local hostnames and the
   thing's own network fields replaced by placeholders), write the staging
   row (the whole API record, including what is never indexed), the
   `registry_assets` row and the asset's aliases in `entity_aliases`.
   Things imported by an integration (UniFi, Zigbee2MQTT, Prusa Connect) are
   staged with `indexable = false` unless `INCLUDE_INTEGRATION_SOURCES`
   names their integration.
3. Things missing from the listing are deleted from staging, the registry,
   the aliases and Qdrant, unless the scan saw less than half the things
   expected (the safety valve; the run then exits 1).
4. Index: one `hs_member` point per asset (`source: asset`, authority 3),
   more only when the description and notes are long. Text: an optional LLM
   description, then a deterministic facts block (name, make and model,
   kind, ownership class, where, related things, link titles and notes,
   asset tag), then the description and notes. Dense + sparse vectors;
   unchanged text (by content hash) is not re-embedded, and a changed
   payload alone is rewritten without re-embedding.
5. `full` (and `reindex`) also deletes this source's points not stamped with
   the run's ID. A Postgres advisory lock stops overlapping runs.

## Forcing a rebuild

- `run --mode full --force`, or `FORCE_REINDEX=true` for serve's first scan:
  restage and re-embed everything, ignoring content hashes.
- `run --mode reindex`: rebuild every point from staging without reading
  the asset manager (after changing the embedding model or chunking).

## Files

- `__main__.py`: CLI, logging, SIGTERM, exit codes.
- `config.py`: `Settings.from_env()`; validates every variable.
- `links_client.py`: the asset manager's API client with retries.
- `assets.py`: things to asset records, scrubbed text, facts, chunks,
  aliases, manual candidates, registry rows (pure).
- `pipeline.py`: `Pipeline.run(mode, force=...)`: staging, registry,
  indexing, reconcile.
- `documents.py`: point IDs, hashes and the shared payload (pure).
- `db.py`: psycopg queries; `migrate()`. `migrations/`: Alembic schema.
- `sanitize.py`: secret scanner. `linker.py`: alias matching.
- `llm.py`: gateway chat tasks, cached in `llm_cache`.
- `embeddings.py`: dense embedders. `sparse.py`: keyword vectors.
- `qdrant_store.py`: collection, payload indexes, upserts, reconcile.
- `schedule.py`: the `serve` loop and heartbeat.
- `taxonomy.txt`: the shared topic taxonomy.

## Shared with the other ingesters (keep identical)

`sanitize.py`, `linker.py`, `sparse.py` (and its `SPARSE_ENCODER` id),
`chunking.py`, `schedule.py`, `embeddings.py`, `taxonomy.txt`, the shared
`CREATE TABLE IF NOT EXISTS` statements in `migrations/versions/v0001_*.py`,
and `point_id` / `base_payload` in `documents.py`. This repo's Alembic
version table is `links2rag_alembic_version`.

Shared tables used: `staging_documents` (sources `asset`, and `manual`),
`entity_aliases` (`entity_type = 'asset'`), `registry_review_queue`,
`quarantine`, `sync_runs` (source `links`), `llm_cache`. Own tables:
`registry_assets` (asset ID, retired keys and slugs, ownership class,
location IDs, links) and `asset_manual_links`.

## Payload

The shared payload (plan 00) plus: `asset_key`, `short_url`,
`manufacturer`, `model`, `category` (the asset manager template's name),
`ownership` (`space` or `member-loan`), `photos`, `links` (type, title,
URL), `related_asset_ids`, `retired_asset_ids`, `payload_hash`.

`asset_id` is the thing's short-URL key (printed on its QR label). Location
IDs are `geowiki:<path>` from a thing's "Where" link to the geowiki
(`GEOWIKI_URL`), plus linker matches on the where note.

## Conventions

- Configuration from environment variables; see docs/configuration.md.
- One bad thing is logged and skipped; it never stops the run.
- Live facts (is the laser working right now?) belong to the realtime tools
  service (plan 11), never to this index.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
