# Configuration

The ingester reads these environment variables at startup. Put them in
`.env` (copy `.env.example`); compose passes it to the container, and a
local run also loads `.env` from the current directory. An invalid value
stops the ingester (exit code 2) with a message naming the variable. An
empty value (`NAME=`) counts as unset. Variables shared with the pipeline's
other ingesters have the same names and meanings there; the embedding
settings must match them, since they all write `hs_member`.

## Asset manager

| Variable | Default | Meaning |
|----------|---------|---------|
| `LINKS_URL` | required | Base URL of the asset manager (pdxhackerspace-links), `http://` or `https://`. |
| `LINKS_API_KEY` | required | An API key from the asset manager's **Settings → API keys**, sent as a Bearer token. The API is read-only; create a key just for this ingester so it can be revoked on its own. |
| `LINKS_PUBLIC_URL` | `LINKS_URL` | Not used for links (the API returns each thing's own URL); its host is never fetched for manuals. |
| `LINKS_PAGE_SIZE` | `500` | Things per page when listing, 1 to 500. |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Timeout for each asset manager request, ≥ 1. |
| `INCLUDE_INTEGRATION_SOURCES` | none | Comma-separated: `unifi`, `zigbee2mqtt`, `prusa_connect`. Things imported by these integrations are indexed too. By default imported things are staged but not indexed: they are network infrastructure, sensors that may be in the access path, or members' own devices. Things created by hand and later linked to an integration are indexed either way. |
| `SPACE_OWNER_NAMES` | `pdx hackerspace,pdxhackerspace,hackerspace,ctrlh,space` | Owner values (case-insensitive) that mean the space owns a thing; a blank owner counts as the space too. Any other owner is indexed only as `member-loan` ("on loan from a member"); the name is staged but never indexed, and is replaced by `[OWNER]` in free text. |
| `GEOWIKI_URL` | unset | The geowiki's base URL. A thing's "Where" link to it becomes a location ID `geowiki:<path>` (e.g. `geowiki:resources/12`). Its host is never fetched for manuals. |
| `SKIP_LINK_HOSTS` | none | Comma-separated hosts never fetched for manuals, besides the asset manager and the geowiki. Set it to the wiki and Slack hosts: other ingesters index those. |

## PostgreSQL

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | required | `postgresql://user:password@host:5432/dbname`. The pipeline's shared database: the same one as the other ingesters. Holds the staging store, the entity registry (`registry_assets`, `entity_aliases`, the alias review queue), manual links, the LLM cache, quarantine and run status. Every run creates or updates the schema itself. |

## Qdrant

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant HTTP endpoint. The compose files set it to their own `qdrant` service. |
| `QDRANT_API_KEY` | unset | Qdrant API key, if your Qdrant requires one. |
| `QDRANT_MEMBER_COLLECTION` | `hs_member` | Member-tier text chunks: assets (`source: asset`) and manuals (`source: manual`). Never `hs_public`. |
| `QDRANT_DISTANCE` | `cosine` | `cosine`, `dot` or `euclid`. Used only when creating the collection. |
| `VECTOR_SIZE` | `768` | Embedding dimension, ≥ 1. Must match the model (or `EMBEDDING_DIMENSIONS`). An existing collection with a different size is an error. |

The collection has a `dense` and a `sparse` (keyword) named vector and
records the embedding model in its metadata; an existing collection embedded
with another model is an error (recreate it and run every source with
`--mode reindex`).

## Embeddings

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `openai` | `openai`: an OpenAI-compatible embeddings API (the pipeline's gateway, TEI, vLLM, Ollama's `/v1`, ...). Or `sentence_transformers` (also `sentence-transformers`, `st`), which needs the `sentence-transformers` extra. |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Model name for the provider. |
| `OPENAI_BASE_URL` | `http://localhost:8080/v1` | Embeddings API base URL; requests go to `<base>/embeddings`. In a container, use the server's reachable address, e.g. `http://host.docker.internal:8080/v1`. |
| `OPENAI_API_KEY` | unset | Sent as `Authorization: Bearer …`. |
| `OPENAI_TIMEOUT_SECONDS` | `120` | Timeout for each embedding request, ≥ 1. |
| `EMBEDDING_DIMENSIONS` | unset | Shorter vectors (`text-embedding-3-*` only); must equal `VECTOR_SIZE` when set. |
| `EMBEDDING_QUERY_PREFIX` | empty | Prefix for search queries, kept exactly (spaces included). Used by `links2rag query` and by `build_embedder(settings).encode(texts, kind="query")`. |
| `EMBEDDING_DOCUMENT_PREFIX` | empty | Prefix added to everything the ingester embeds. Changing it changes every vector: recreate the collection and run `--mode reindex`. |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Model context in tokens; longer texts are truncated (about 4 characters per token). `0` disables truncation. |
| `HF_CACHE_DIR` | unset | Hugging Face model cache for `sentence_transformers`; else `SENTENCE_TRANSFORMERS_HOME`, then `HF_HOME`. |
| `SENTENCE_TRANSFORMERS_HOME` | unset | See `HF_CACHE_DIR`. |
| `HF_HOME` | unset | See `HF_CACHE_DIR`. |

Use the same model and prefixes as the pipeline's other ingesters (for
`nomic-ai/modernbert-embed-base`: `"search_query: "` and
`"search_document: "`).

## Generation (ingest-time LLM tasks)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_MODEL` | unset | Chat model on the gateway (`POST <base>/chat/completions`). Writes each asset's description and maps it to the topic taxonomy, writes context lines for manual chunks, classifies manual sections that keyword rules can't, proposes asset nicknames for review, and confirms ambiguous linker matches. Results are cached in Postgres by input hash. Unset: these steps are skipped and the deterministic text is indexed. |
| `LLM_BASE_URL` | `OPENAI_BASE_URL` | Gateway base URL for generation. |
| `LLM_API_KEY` | `OPENAI_API_KEY` | Sent as `Authorization: Bearer …`. |
| `LLM_TIMEOUT_SECONDS` | `120` | Timeout per request, ≥ 1. |
| `VISION_MODEL` | unset | Vision model on the gateway for short descriptions of manual pages that are mostly a diagram, marked machine-generated. Needs `LLM_MODEL` set too. |
| `TOPIC_TAXONOMY_FILE` | packaged `taxonomy.txt` | Shared topic taxonomy: one slug per line, `#` comments. |

## Manuals

| Variable | Default | Meaning |
|----------|---------|---------|
| `MANUALS_ENABLED` | `true` | Download, parse and index manuals. |
| `MANUAL_LINK_PATTERN` | manual, datasheet, user guide, quick start, guide, instructions, handbook, documentation, or a `.pdf` URL | Case-insensitive regular expression. Custom links whose title or URL matches are manuals. The asset manager has no "manual" link type. |
| `MANUALS_FROM_MANUFACTURER_URL` | `true` | Also look one level into each thing's manufacturer link for manual PDFs (links whose text or URL matches the pattern, on the same site). The manufacturer page itself is not indexed. |
| `MANUAL_REFETCH_DAYS` | `7` | Days before a manual is downloaded again, ≥ 1. A changed link is fetched on the next scan. `--force` refetches everything. |
| `MANUAL_MAX_MB` | `50` | Largest file downloaded, in MB. |
| `MANUALS_DIR` | unset | Manuals placed by hand: each file with a sidecar `<file>.yaml` or `<file>.json` holding `asset_keys` (a list) and optionally `title` and `url`. |
| `DATA_DIR` | `/data` | Where downloaded manuals (`manuals/<sha256>.<ext>`) and reports (`reports/*.json`) are kept. Mount a volume here. |
| `ALLOW_PRIVATE_HOSTS` | `true` | Fetch manual links on private, loopback and link-local addresses, such as the space's own servers. Set `false` to fetch only hosts that resolve to public addresses. |
| `OCR_ENABLED` | `true` | OCR PDF pages with no text layer (scans) with tesseract, which the image includes. |
| `OCR_LANGUAGE` | `eng` | Tesseract language(s), e.g. `eng+deu`. |

## Chunking and logging

| Variable | Default | Meaning |
|----------|---------|---------|
| `CHUNK_SIZE` | `1800` | Target chunk length in characters (about 450 tokens), ≥ 1. |
| `CHUNK_OVERLAP` | `150` | Characters repeated at the start of the next chunk when a section is split, ≥ 0 and < `CHUNK_SIZE`. |
| `BATCH_SIZE` | `64` | Points per embedding request and Qdrant upsert, ≥ 1. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |

## Scheduling and forcing

The default command, `serve`, runs until stopped: it scans, sleeps
`SCAN_INTERVAL_MINUTES`, and scans again. A scan is incremental unless no
full scan has finished within `FULL_SCAN_INTERVAL_HOURS`, so the first scan
on a fresh database is a full one. `run --mode ...` does a single scan.

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCAN_INTERVAL_MINUTES` | `60` | Minutes to sleep after one scan ends before the next starts, ≥ 1. |
| `FULL_SCAN_INTERVAL_HOURS` | `24` | Hours between full scans (which also reconcile deletions by run ID), ≥ 1. |
| `FORCE_REINDEX` | `false` | `true`: serve's first scan is a full one that restages, re-downloads and re-embeds everything, as `run --mode full --force` does. Later scans are normal. Set it back to `false` afterwards. |

`true`/`false` settings also accept `1`/`0`, `yes`/`no` and `on`/`off`.

## Compose and tooling (not read by the app)

| Variable | Meaning |
|----------|---------|
| `APP_VERSION` | Release of the image `docker-compose.prod.yaml` runs (default `latest`). |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | The `db` service's account and database in `docker-compose.prod.yaml`; match them in `DATABASE_URL`. |
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. |
