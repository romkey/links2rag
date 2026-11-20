# links2rag

The asset and equipment-manual source of the PDX Hackerspace RAG pipeline.
It reads every thing in the hackerspace asset manager
([pdxhackerspace-links](https://github.com/romkey/pdxhackerspace-links)), the
QR-code asset tracker, through its read-only API, stages it in the
pipeline's PostgreSQL, and indexes it, with the equipment manuals the things
link to, in the member-tier Qdrant collection `hs_member`.

- **Member tier only.** Taken together, asset data is an inventory of what
  is worth stealing; nothing goes to `hs_public`.
- **Never indexed:** serial numbers, owners' names (only "space-owned" or
  "on loan from a member"), IP and MAC addresses, hostnames, beacon IDs,
  integration records. They are staged, so indexes can be rebuilt without
  refetching.
- **Source of truth for `asset_id`:** each thing's short-URL key, the one on
  its QR label. `registry_assets` and `entity_aliases` let other sources
  link to assets ("the big laser" → its key).
- Wiki and Slack links on things are recorded, not fetched: those sources
  have their own ingesters.

See [docs/overview.md](docs/overview.md) for how it works and
[docs/configuration.md](docs/configuration.md) for every setting.

## Running it

```bash
cp .env.example .env          # set LINKS_URL, LINKS_API_KEY, DATABASE_URL, the gateway
docker compose -f docker-compose.prod.yaml up -d
```

It scans, sleeps `SCAN_INTERVAL_MINUTES` (default 60), and scans again.
The first scan on a fresh database is a full one.

One-off commands (prefix with
`docker compose -f docker-compose.prod.yaml run --rm app`):

```bash
links2rag run --mode incremental    # changed and deleted things
links2rag run --mode full           # everything, then reconcile
links2rag run --mode full --force   # restage, re-download and re-embed everything
links2rag run --mode reindex        # rebuild points from staging (new model or chunker)
links2rag status                    # recent runs and counts, as JSON
```

`FORCE_REINDEX=true` makes serve's first scan a forced full one.

## Querying the collection

Points carry the shared payload (plan 00): `source` (`asset` or `manual`),
`doc_id`, `text`, `context`, `url`, `tier`, `authority` (3 for assets, 2 for
manuals), `asset_ids`, `location_ids`, `topics`, `page` and more; see
[docs/overview.md](docs/overview.md#payload). Each point has a `dense`
vector (embed queries with the same model and `EMBEDDING_QUERY_PREFIX`) and
a `sparse` keyword vector (`links2rag.sparse.sparse_vector`, the encoder all
the pipeline's ingesters share).

When a question resolves to a specific asset, search its manuals with a
filter on `asset_ids` alongside the general search. Space-written content
(wiki, signs, policy) outranks manuals. Current status and location of an
asset belong to the realtime tools service, not this index.

## Development

```bash
docker compose up --build                                  # serve against local Qdrant and Postgres
docker compose -f docker-compose.test.yaml run --rm test   # tests
docker compose -f docker-compose.lint.yaml run --rm lint   # lint, format, types, secrets
```

Releases: Actions → Release → Run workflow. The production image is
published to `ghcr.io/romkey/links2rag`.

## License

MIT
