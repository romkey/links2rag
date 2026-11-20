# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-20

### Added

- Asset source of the PDX Hackerspace RAG pipeline (plans 00 and 06): every
  thing in the asset manager is staged in PostgreSQL and indexed in
  `hs_member` (`source: asset`), member tier only. Serial numbers, owner
  names, network identifiers and integration records are staged but never
  indexed; free text is scrubbed of secrets (quarantined) and identifiers.
- `registry_assets` and asset aliases in `entity_aliases`, so other sources
  can link to assets by name, label, make and model.
- `serve` (scan, sleep `SCAN_INTERVAL_MINUTES`, repeat), `run --mode
  incremental|full|reindex`, `--force` and `FORCE_REINDEX` for a complete
  rebuild, `status`, `migrate` and `healthcheck`.
- Equipment manuals (`source: manual`): manual links, manufacturer pages
  (one level) and drop-in files are downloaded, deduplicated by SHA-256 and
  kept in `DATA_DIR`; PDFs are parsed with their outline or font-size
  headings, tables kept whole with their caption, scanned pages OCRed with
  tesseract and diagram pages described by `VISION_MODEL`. Chunks follow the
  manual's sections, carry their page and a `#page=N` link, and are tagged
  with a `manual_section` kind. Only English is indexed. Dead links are
  recorded and keep their last good copy.
- Nickname proposals: with `LLM_MODEL`, the LLM reads wiki and Slack
  mentions of each asset and proposes the names members use; `links2rag
  aliases` lists them and `--approve`/`--reject` decide.
- A report after every full run (and `links2rag report`) of assets with no
  wiki page, manual, location or training link, failing manual links,
  quarantine and aliases to review, also written to
  `DATA_DIR/reports/links2rag.json`.
- `links2rag query`: hybrid search of `hs_member`, optionally filtered to
  one asset or source.
