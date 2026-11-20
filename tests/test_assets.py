import re

from links2rag import assets
from links2rag.config import DEFAULT_MANUAL_LINK_PATTERN, DEFAULT_SPACE_OWNERS
from tests.fakes import HOSTNAME, IEEE, IP, OWNER, SERIAL, thing

GEOWIKI = "https://geowiki.example.com"


def _parse(**kwargs: object) -> assets.Asset:
    return assets.parse_thing(
        thing(**kwargs),  # type: ignore[arg-type]
        space_names=DEFAULT_SPACE_OWNERS,
        geowiki_url=GEOWIKI,
    )


def test_parse_thing_basics() -> None:
    asset = _parse()
    assert asset.asset_id == "abcd1234"
    assert asset.make_model == "Thunder Laser Nova 51"
    assert asset.category == "Laser cutter"
    assert asset.where == "Section 2, by the back door"
    assert asset.location_ids == ["geowiki:resources/12"]
    assert asset.ownership == "space"
    assert asset.photos[0]["filename"] == "laser.jpg"
    assert asset.flags == {}


def test_ownership_never_names_the_member() -> None:
    assert assets.ownership(None, DEFAULT_SPACE_OWNERS) == "space"
    assert assets.ownership("PDX Hackerspace", DEFAULT_SPACE_OWNERS) == "space"
    assert assets.ownership(OWNER, DEFAULT_SPACE_OWNERS) == "member-loan"
    asset = _parse(owner=OWNER, notes=f"Ask {OWNER} before using it.")
    text = assets.facts(asset) + assets.body(asset)
    assert OWNER not in text
    assert "[OWNER]" in text
    assert "on loan" in text


def test_location_id_only_for_geowiki_urls() -> None:
    assert assets.location_id(None, GEOWIKI) is None
    assert assets.location_id("https://geowiki.example.com/r/1", None) is None
    assert assets.location_id("https://other.example.com/r/1", GEOWIKI) is None
    assert assets.location_id("https://geowiki.example.com/", GEOWIKI) is None
    assert assets.location_id("https://GEOWIKI.example.com/maps/2/rooms/3", GEOWIKI) == (
        "geowiki:maps/2/rooms/3"
    )


def test_scrub_removes_identifiers_and_secrets() -> None:
    raw = thing()
    text = (
        f"Serial {SERIAL}, at {IP} ({HOSTNAME}), mac {IEEE}, also 192.168.1.5, "
        "fe80:0:0:0:1:2:3:4, printer.local, password: hunter2hunter2"
    )
    cleaned, rules = assets.scrub(text, raw, owner_is_member=False)
    for canary in (SERIAL, IP, HOSTNAME, IEEE, "192.168.1.5", "printer.local", "hunter2"):
        assert canary not in cleaned
    assert "[SERIAL]" in cleaned
    assert "[IP]" in cleaned
    assert "[HOST]" in cleaned
    assert "[MAC]" in cleaned
    assert rules == ["assigned-secret"]
    assert assets.scrub("", raw, owner_is_member=False) == ("", [])


def test_secret_flag_on_asset() -> None:
    asset = _parse(notes="wifi_password = s3cretpass99")
    assert asset.flags == {"secret_redacted": ["assigned-secret"]}
    assert "s3cretpass99" not in asset.notes


def test_facts_and_chunks() -> None:
    asset = _parse(
        related=[{"id": 2, "key": "remote01", "name": "Laser chiller", "note": "cooling"}],
        links=[
            {"type": "wiki", "title": "Wiki", "url": "https://wiki.example.com/laser"},
            {"type": "slack", "title": "Slack", "url": None, "note": "Ask in #laser"},
            {"type": "custom", "title": "Manual", "url": "https://v.example.com/m.pdf"},
        ],
    )
    text = assets.facts(asset)
    assert "Related: Laser chiller (cooling)" in text
    assert "Links: Wiki, Manual" in text
    assert "Slack: Ask in #laser" in text
    assert "Asset tag: abcd1234" in text
    chunks = assets.chunk_texts(asset, summary="Summary.", chunk_size=400, chunk_overlap=40)
    assert len(chunks) == 1
    assert chunks[0].startswith("Summary.\n\nBig Laser")
    assert "Never cut PVC" in chunks[0]


def test_long_body_is_split() -> None:
    asset = _parse(description="Cuts things. " * 200)
    chunks = assets.chunk_texts(asset, chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 2
    assert chunks[1].startswith("Big Laser: ")
    bare = _parse(description="", notes="", links=[])
    assert assets.chunk_texts(bare, chunk_size=300, chunk_overlap=30) == [assets.facts(bare)]


def test_facts_variants() -> None:
    asset = _parse(manufacturer=None, model=None, template=None, owner=OWNER, links=[])
    asset.label_name = "LASER"
    text = assets.facts(asset)
    assert "Make and model" not in text
    assert "Kind:" not in text
    assert "Where:" not in text
    asset.label_name = "Laser 1"
    asset.location_ids = ["geowiki:x"]
    asset.where_url = "https://geowiki.example.com/x"
    text = assets.facts(asset)
    assert "Also labelled: Laser 1" in text
    assert "Where: see https://geowiki.example.com/x" in text


def test_aliases() -> None:
    asset = _parse()
    asset.label_name = "Laser"
    assert assets.aliases(asset) == [
        "Big Laser",
        "Laser",
        "Nova 51",
        "Thunder Laser Nova 51",
        "big laser",
    ]


def test_manual_candidates() -> None:
    asset = _parse(
        links=[
            {"type": "custom", "title": "User Manual", "url": "https://v.example.com/a"},
            {"type": "custom", "title": "Parts", "url": "https://v.example.com/b.pdf"},
            {"type": "custom", "title": "Vendor", "url": "https://v.example.com/shop"},
            {"type": "custom", "title": "Empty", "url": None},
            {"type": "wiki", "title": "Wiki manual", "url": "https://wiki.example.com/x"},
        ],
        manufacturer_url="https://thunderlaser.example.com",
    )
    pattern = re.compile(DEFAULT_MANUAL_LINK_PATTERN, re.IGNORECASE)
    found = assets.manual_candidates(asset, pattern, include_manufacturer_url=True)
    assert [(c.url, c.origin) for c in found] == [
        ("https://v.example.com/a", "link"),
        ("https://v.example.com/b.pdf", "link"),
        ("https://thunderlaser.example.com", "manufacturer"),
    ]
    assert len(assets.manual_candidates(asset, pattern, include_manufacturer_url=False)) == 2


def test_excluded_integrations() -> None:
    asset = _parse(integration_source="unifi")
    assert assets.excluded(asset, ()) == "imported by unifi"
    assert assets.excluded(asset, ("unifi",)) is None
    assert assets.excluded(_parse(), ()) is None


def test_content_hash_ignores_scans() -> None:
    a = thing()
    b = thing()
    b["qr_scan_count"] = 99
    b["updated_at"] = "2027-01-01T00:00:00Z"
    b["integrations"] = {}
    assert assets.content_hash(a) == assets.content_hash(b)
    b["notes"] = "changed"
    assert assets.content_hash(a) != assets.content_hash(b)


def test_registry_row_and_payload() -> None:
    asset = _parse(
        aliases=[{"key": "oldkey12", "slug": "old-laser"}, {"key": None, "slug": None}],
        related=[{"id": 2, "key": "remote01", "name": "Chiller", "note": None}],
    )
    row = assets.registry_row(asset, indexable=True)
    assert row["retired_ids"] == ["old-laser", "oldkey12"]
    assert row["related_asset_ids"] == ["remote01"]
    extras = assets.payload_extras(asset)
    assert extras["asset_key"] == "abcd1234"
    assert extras["retired_asset_ids"] == ["old-laser", "oldkey12"]
    assert all(link["url"] for link in extras["links"])
    for canary in (SERIAL, IP, HOSTNAME, IEEE):
        assert canary not in str(row)
        assert canary not in str(extras)


def test_parse_tolerates_odd_records() -> None:
    raw = thing(name="", links=[], template=None)
    raw["links"] = ["junk", {"type": "custom", "title": "", "url": "https://x.example.com"}]
    raw["related_things"] = ["junk"]
    raw["photos"] = [{"url": None}, "junk"]
    raw["key"] = ""
    asset = assets.parse_thing(raw, space_names=DEFAULT_SPACE_OWNERS)
    assert asset.asset_id == "1"
    assert asset.name == "1"
    assert asset.links[0].title == "Custom"
    assert asset.photos == []
    assert asset.related == []
