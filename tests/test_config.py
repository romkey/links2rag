import pytest

from links2rag.config import DEFAULT_SPACE_OWNERS, ConfigError, Settings

BASE = {
    "LINKS_URL": "https://links.example.com/",
    "LINKS_API_KEY": "links_dummy",  # not-a-secret
    "DATABASE_URL": "postgresql://u:p@db/x",  # not-a-secret
}


def test_defaults() -> None:
    s = Settings.from_env(BASE)
    assert s.links_url == "https://links.example.com"
    assert s.link_base_url == "https://links.example.com"
    assert s.member_collection == "hs_member"
    assert s.scan_interval_minutes == 60
    assert s.full_scan_interval_hours == 24
    assert s.force_reindex is False
    assert s.space_owner_names == DEFAULT_SPACE_OWNERS
    assert s.include_integration_sources == ()
    assert s.manuals_enabled is True
    # Manuals often live on the space's own servers.
    assert s.allow_private_hosts is True
    assert s.manual_pattern.search("User Manual")
    assert s.gateway_base_url == s.openai_base_url
    assert s.gateway_api_key is None
    assert s.never_fetch_hosts == frozenset({"links.example.com"})


def test_everything_set() -> None:
    s = Settings.from_env(
        {
            **BASE,
            "LINKS_PUBLIC_URL": "https://public.example.com",
            "GEOWIKI_URL": "https://geo.example.com",
            "SKIP_LINK_HOSTS": "Wiki.example.com, slack.com",
            "INCLUDE_INTEGRATION_SOURCES": "Prusa_Connect",
            "SPACE_OWNER_NAMES": "CTRL-H",
            "LLM_MODEL": "m",
            "LLM_BASE_URL": "http://gw/v1",
            "LLM_API_KEY": "k",  # not-a-secret
            "OPENAI_API_KEY": "o",  # not-a-secret
            "VECTOR_SIZE": "512",
            "EMBEDDING_DIMENSIONS": "512",
            "SCAN_INTERVAL_MINUTES": "15",
            "FORCE_REINDEX": "yes",
            "MANUALS_ENABLED": "off",
            "LOG_LEVEL": "debug",
            "EMBEDDING_QUERY_PREFIX": "q: ",
        }
    )
    assert s.link_base_url == "https://public.example.com"
    assert s.never_fetch_hosts == frozenset(
        {
            "links.example.com",
            "public.example.com",
            "geo.example.com",
            "wiki.example.com",
            "slack.com",
        }
    )
    assert s.include_integration_sources == ("prusa_connect",)
    assert s.space_owner_names == ("ctrl-h",)
    assert s.gateway_base_url == "http://gw/v1"
    assert s.gateway_api_key == "k"
    assert s.embedding_dimensions == 512
    assert s.scan_interval_minutes == 15
    assert s.force_reindex is True
    assert s.manuals_enabled is False
    assert s.log_level == "DEBUG"
    assert s.embedding_query_prefix == "q: "


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"LINKS_URL": ""}, "LINKS_URL"),
        ({"LINKS_URL": "links.example.com"}, "must start with http"),
        ({"CHUNK_SIZE": "abc"}, "whole number"),
        ({"CHUNK_SIZE": "0"}, "at least 1"),
        ({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, "smaller than CHUNK_SIZE"),
        ({"EMBEDDING_DIMENSIONS": "100"}, "must equal VECTOR_SIZE"),
        ({"LOG_LEVEL": "LOUD"}, "not a logging level"),
        ({"QDRANT_MEMBER_COLLECTION": "hs_public"}, "member tier only"),
        ({"INCLUDE_INTEGRATION_SOURCES": "unifi,nope"}, "not nope"),
        ({"MANUAL_LINK_PATTERN": "("}, "not a valid regular expression"),
        ({"FORCE_REINDEX": "maybe"}, "true or false"),
        ({"QDRANT_DISTANCE": "manhattan"}, "must be one of"),
    ],
)
def test_invalid(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({**BASE, **overrides})
