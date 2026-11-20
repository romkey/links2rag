from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from links2rag.config import Settings
from links2rag.db import Database, migrate

# Tables the tests empty before each database test.
_TABLES = (
    "staging_documents",
    "quarantine",
    "sync_runs",
    "llm_cache",
    "entity_aliases",
    "registry_review_queue",
    "registry_assets",
    "asset_manual_links",
)


def base_settings() -> Settings:
    return Settings(
        links_url="http://links.local",
        links_api_key="test-key",  # not-a-secret: dummy test key
        links_public_url="https://links.example.com",
        database_url="postgresql://unused",
        qdrant_url="http://localhost:6333",
        geowiki_url="https://geowiki.example.com",
        embedding_provider="openai",
        embedding_model="dummy",
        openai_base_url="http://localhost:11434/v1",
        openai_timeout_seconds=30,
        chunk_size=400,
        chunk_overlap=40,
        batch_size=100,
        vector_size=8,
        manuals_enabled=False,
        data_dir="/nonexistent",
    )


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Settings for tests; keyword arguments override fields."""

    # Any: overrides may be any Settings field; dataclasses.replace checks names.
    def _make(**overrides: Any) -> Settings:
        return dataclasses.replace(base_settings(), **overrides)

    return _make


def _database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.fail(
            "Tests need PostgreSQL: run them with "
            "`docker compose -f docker-compose.test.yaml run --rm test`, or set DATABASE_URL"
        )
    return url


@pytest.fixture(scope="session")
def database_url() -> str:
    url = _database_url()
    migrate(url)
    return url


@pytest.fixture
def db(database_url: str) -> Iterator[Database]:
    """A connection to an emptied, migrated database."""
    with Database.connect(database_url) as database:
        database.conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
        database.commit()
        yield database
        database.conn.rollback()
