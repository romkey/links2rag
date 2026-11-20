# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.0.0. When re-scanning, apply the entries in
its UPGRADING.md newer than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

- The asset manager's API shape comes from its source:
  `app/services/api/v1/thing_serializer.rb` and
  `app/controllers/api/v1/things_controller.rb` in
  https://github.com/romkey/pdxhackerspace-links. Check there rather than
  guessing; the API has no status, training or category fields.
- This is one source of the PDX Hackerspace RAG pipeline (plans 00 and 06).
  The modules listed under "Shared with the other ingesters" in
  docs/overview.md must stay identical to their copies in
  audiobookshelf-rag-ingester and calibre-web2rag.
- Keep `assets.py` and `documents.py` pure (no I/O).
- Member tier only. Never write to `hs_public`, and never index serial
  numbers, owner names, network or radio identifiers, or integration records.
  The canary tests in tests/test_pipeline.py check this; keep them.
- Never delete anything in a run that was stopped or past the safety valve.
- Tests need PostgreSQL (`DATABASE_URL`); `docker-compose.test.yaml` and CI
  provide one.
