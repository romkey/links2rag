"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example. Names shared with the pipeline's other
ingesters (DATABASE_URL, QDRANT_*, EMBEDDING_*, LLM_*, SCAN_INTERVAL_MINUTES,
...) mean the same thing in each of them.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

EMBEDDING_PROVIDERS = frozenset({"openai", "sentence_transformers", "sentence-transformers", "st"})
DISTANCES = frozenset({"cosine", "dot", "euclid"})
INTEGRATION_SOURCES = frozenset({"unifi", "zigbee2mqtt", "prusa_connect"})
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

# Custom links whose title or URL matches are treated as manuals.
DEFAULT_MANUAL_LINK_PATTERN = (
    r"manual|datasheet|data[ _-]?sheet|user[ _-]?guide|quick[ _-]?start|guide|instructions"
    r"|handbook|documentation|\.pdf(?:$|[?#])"
)
# Owner values that mean the space owns the thing (case-insensitive). A blank
# owner counts as space-owned too.
DEFAULT_SPACE_OWNERS = ("pdx hackerspace", "pdxhackerspace", "hackerspace", "ctrlh", "space")


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


def _url(name: str, value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must start with http:// or https://, not {value!r}")
    return value.rstrip("/")


def _host(url: str | None) -> str | None:
    return (urlsplit(url).hostname or None) if url else None


@dataclass(frozen=True)
class Settings:
    links_url: str
    links_api_key: str
    database_url: str
    links_public_url: str | None = None
    links_page_size: int = 500
    request_timeout_seconds: int = 60
    include_integration_sources: tuple[str, ...] = ()
    space_owner_names: tuple[str, ...] = DEFAULT_SPACE_OWNERS
    geowiki_url: str | None = None
    skip_link_hosts: tuple[str, ...] = ()
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    member_collection: str = "hs_member"
    embedding_provider: str = "openai"
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    hf_cache_dir: str | None = None
    openai_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str | None = None
    openai_timeout_seconds: int = 120
    chunk_size: int = 1800
    chunk_overlap: int = 150
    batch_size: int = 64
    distance: str = "cosine"
    vector_size: int = 768
    embedding_context_length: int = 8192
    embedding_dimensions: int | None = None
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: int = 120
    vision_model: str | None = None
    topic_taxonomy_file: str | None = None
    manuals_enabled: bool = True
    manual_link_pattern: str = DEFAULT_MANUAL_LINK_PATTERN
    manuals_from_manufacturer_url: bool = True
    manual_refetch_days: int = 7
    manual_max_mb: int = 50
    manuals_dir: str | None = None
    data_dir: str = "/data"
    allow_private_hosts: bool = True
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    scan_interval_minutes: int = 60
    full_scan_interval_hours: int = 24
    force_reindex: bool = False
    log_level: str = "INFO"

    @property
    def link_base_url(self) -> str:
        """Base of the links stored with each point: LINKS_PUBLIC_URL, else LINKS_URL."""
        return self.links_public_url or self.links_url

    @property
    def gateway_base_url(self) -> str:
        """Where generation requests go: LLM_BASE_URL, else the embeddings server."""
        return self.llm_base_url or self.openai_base_url

    @property
    def gateway_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key

    @property
    def manual_pattern(self) -> re.Pattern[str]:
        return re.compile(self.manual_link_pattern, re.IGNORECASE)

    @property
    def never_fetch_hosts(self) -> frozenset[str]:
        """Hosts whose pages other ingesters index, or that need a login: never fetched."""
        hosts = {h.lower() for h in self.skip_link_hosts}
        for url in (self.links_url, self.links_public_url, self.geowiki_url):
            host = _host(url)
            if host:
                hosts.add(host.lower())
        return frozenset(hosts)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        def required(name: str) -> str:
            value = (env.get(name) or "").strip()
            if not value:
                raise ConfigError(f"Missing required environment variable: {name}")
            return value

        def optional(name: str) -> str | None:
            # An empty value (NAME= in .env) means unset.
            return (env.get(name) or "").strip() or None

        def integer(name: str, default: int, minimum: int) -> int:
            text = (env.get(name) or "").strip() or str(default)
            try:
                value = int(text)
            except ValueError:
                raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
            if value < minimum:
                raise ConfigError(f"{name} must be at least {minimum}, not {value}")
            return value

        def boolean(name: str, default: bool) -> bool:
            text = (env.get(name) or "").strip().lower()
            if not text:
                return default
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ConfigError(f"{name} must be true or false, not {text!r}")

        def choice(name: str, default: str, allowed: frozenset[str]) -> str:
            value = ((env.get(name) or "").strip() or default).lower()
            if value not in allowed:
                options = ", ".join(sorted(allowed))
                raise ConfigError(f"{name} must be one of {options}, not {value!r}")
            return value

        def listing(name: str) -> tuple[str, ...]:
            return tuple(part.strip() for part in (env.get(name) or "").split(",") if part.strip())

        # About 450 tokens at ~4 characters per token (plan: 300-500 tokens).
        chunk_size = integer("CHUNK_SIZE", 1800, 1)
        chunk_overlap = integer("CHUNK_OVERLAP", 150, 0)
        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})"
            )

        vector_size = integer("VECTOR_SIZE", 768, 1)
        # 0 or unset: the model's full size.
        embedding_dimensions = integer("EMBEDDING_DIMENSIONS", 0, 0) or None
        if embedding_dimensions is not None and embedding_dimensions != vector_size:
            raise ConfigError(
                f"EMBEDDING_DIMENSIONS ({embedding_dimensions}) must equal "
                f"VECTOR_SIZE ({vector_size})"
            )

        log_level = ((env.get("LOG_LEVEL") or "").strip() or "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        member = optional("QDRANT_MEMBER_COLLECTION") or "hs_member"
        if member == "hs_public":
            raise ConfigError("This ingester is member tier only; it never writes to hs_public")

        integrations = tuple(s.lower() for s in listing("INCLUDE_INTEGRATION_SOURCES"))
        unknown = sorted(set(integrations) - INTEGRATION_SOURCES)
        if unknown:
            raise ConfigError(
                "INCLUDE_INTEGRATION_SOURCES may list only "
                f"{', '.join(sorted(INTEGRATION_SOURCES))}, not {', '.join(unknown)}"
            )

        pattern = optional("MANUAL_LINK_PATTERN") or DEFAULT_MANUAL_LINK_PATTERN
        try:
            re.compile(pattern)
        except re.error as e:
            raise ConfigError(f"MANUAL_LINK_PATTERN is not a valid regular expression: {e}") from e

        owners = tuple(s.lower() for s in listing("SPACE_OWNER_NAMES")) or DEFAULT_SPACE_OWNERS
        public = optional("LINKS_PUBLIC_URL")
        geowiki = optional("GEOWIKI_URL")
        return cls(
            links_url=_url("LINKS_URL", required("LINKS_URL")),
            links_api_key=required("LINKS_API_KEY"),
            links_public_url=_url("LINKS_PUBLIC_URL", public) if public else None,
            links_page_size=integer("LINKS_PAGE_SIZE", 500, 1),
            request_timeout_seconds=integer("REQUEST_TIMEOUT_SECONDS", 60, 1),
            include_integration_sources=integrations,
            space_owner_names=owners,
            geowiki_url=_url("GEOWIKI_URL", geowiki) if geowiki else None,
            skip_link_hosts=tuple(h.lower() for h in listing("SKIP_LINK_HOSTS")),
            database_url=required("DATABASE_URL"),
            qdrant_url=_url("QDRANT_URL", optional("QDRANT_URL") or "http://qdrant:6333"),
            qdrant_api_key=optional("QDRANT_API_KEY"),
            member_collection=member,
            llm_model=optional("LLM_MODEL"),
            llm_base_url=optional("LLM_BASE_URL"),
            llm_api_key=optional("LLM_API_KEY"),
            llm_timeout_seconds=integer("LLM_TIMEOUT_SECONDS", 120, 1),
            vision_model=optional("VISION_MODEL"),
            topic_taxonomy_file=optional("TOPIC_TAXONOMY_FILE"),
            embedding_provider=choice("EMBEDDING_PROVIDER", "openai", EMBEDDING_PROVIDERS),
            embedding_model=optional("EMBEDDING_MODEL") or "nomic-ai/modernbert-embed-base",
            hf_cache_dir=(
                optional("HF_CACHE_DIR")
                or optional("SENTENCE_TRANSFORMERS_HOME")
                or optional("HF_HOME")
            ),
            openai_base_url=_url(
                "OPENAI_BASE_URL", optional("OPENAI_BASE_URL") or "http://localhost:8080/v1"
            ),
            openai_api_key=optional("OPENAI_API_KEY"),
            openai_timeout_seconds=integer("OPENAI_TIMEOUT_SECONDS", 120, 1),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=integer("BATCH_SIZE", 64, 1),
            distance=choice("QDRANT_DISTANCE", "cosine", DISTANCES),
            vector_size=vector_size,
            # 0 disables truncation.
            embedding_context_length=integer("EMBEDDING_CONTEXT_LENGTH", 8192, 0),
            embedding_dimensions=embedding_dimensions,
            # Kept exactly as given: the trailing space usually matters.
            embedding_query_prefix=env.get("EMBEDDING_QUERY_PREFIX", ""),
            embedding_document_prefix=env.get("EMBEDDING_DOCUMENT_PREFIX", ""),
            manuals_enabled=boolean("MANUALS_ENABLED", True),
            manual_link_pattern=pattern,
            manuals_from_manufacturer_url=boolean("MANUALS_FROM_MANUFACTURER_URL", True),
            manual_refetch_days=integer("MANUAL_REFETCH_DAYS", 7, 1),
            manual_max_mb=integer("MANUAL_MAX_MB", 50, 1),
            manuals_dir=optional("MANUALS_DIR"),
            data_dir=optional("DATA_DIR") or "/data",
            allow_private_hosts=boolean("ALLOW_PRIVATE_HOSTS", True),
            ocr_enabled=boolean("OCR_ENABLED", True),
            ocr_language=optional("OCR_LANGUAGE") or "eng",
            scan_interval_minutes=integer("SCAN_INTERVAL_MINUTES", 60, 1),
            full_scan_interval_hours=integer("FULL_SCAN_INTERVAL_HOURS", 24, 1),
            force_reindex=boolean("FORCE_REINDEX", False),
            log_level=log_level,
        )
