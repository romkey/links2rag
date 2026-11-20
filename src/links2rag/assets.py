"""Asset manager things -> asset records, index text, aliases and manual candidates.

Pure functions, no I/O. The whole API record is staged; only what is built
here is indexed. Never indexed: serial number, the owner's name, network and
radio identifiers, integration records, scan counts and the AR marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from links2rag.chunking import split_text
from links2rag.documents import json_hash
from links2rag.sanitize import redact

# Any: API JSON.
JSON = dict[str, Any]

# Fields that change without the thing changing (scans, imports).
_VOLATILE_FIELDS = frozenset(
    {"visit_count", "qr_scan_count", "nfc_scan_count", "integrations", "updated_at"}
)
# Thing fields that identify the device or person, never indexed.
_PRIVATE_FIELDS = (
    ("serial_number", "[SERIAL]"),
    ("ip_address", "[IP]"),
    ("reserved_ip_address", "[IP]"),
    ("hostname", "[HOST]"),
    ("ieee_address", "[MAC]"),
    ("ble_beacon_uuid", "[BEACON]"),
    ("network_name", "[NETWORK]"),
)
_IPV4 = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")
_IPV6 = re.compile(r"\b(?:[0-9a-f]{1,4}:){4,7}[0-9a-f]{1,4}\b", re.IGNORECASE)
_MAC = re.compile(r"\b[0-9a-f]{2}(?:[:-][0-9a-f]{2}){5}(?:[:-][0-9a-f]{2}){0,2}\b", re.IGNORECASE)
_LOCAL_HOST = re.compile(
    r"\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]+)*\.(?:local|lan|internal|home\.arpa)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Link:
    type: str
    title: str
    url: str | None
    note: str | None


@dataclass(frozen=True)
class ManualCandidate:
    url: str
    title: str | None
    origin: str  # "link" (a custom link) or "manufacturer" (manufacturer_url)


@dataclass
class Asset:
    asset_id: str
    thing_id: int
    name: str
    url: str | None
    slug: str | None = None
    short_url: str | None = None
    label_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    category: str | None = None
    description: str = ""
    notes: str = ""
    ownership: str = "space"
    where: str = ""
    where_url: str | None = None
    location_ids: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    related: list[JSON] = field(default_factory=list)
    retired_ids: list[str] = field(default_factory=list)
    photos: list[JSON] = field(default_factory=list)
    manufacturer_url: str | None = None
    integration_source: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def make_model(self) -> str:
        return " ".join(p for p in (self.manufacturer, self.model) if p)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _opt(value: object) -> str | None:
    return _text(value) or None


def content_hash(thing: JSON) -> str:
    """Hash of a thing's content, ignoring fields that change on every scan or import."""
    return json_hash({k: v for k, v in thing.items() if k not in _VOLATILE_FIELDS})


def ownership(owner: str | None, space_names: tuple[str, ...]) -> str:
    """'space' for space-owned (or no owner recorded), else 'member-loan'. Never the name."""
    value = (owner or "").strip().lower()
    if not value or value in space_names:
        return "space"
    return "member-loan"


def location_id(url: str | None, geowiki_url: str | None) -> str | None:
    """A geowiki URL as a location ID (``geowiki:<path>``); None for other URLs."""
    if not url or not geowiki_url:
        return None
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != (urlsplit(geowiki_url).hostname or "").lower():
        return None
    path = parts.path.strip("/")
    return f"geowiki:{path}" if path else None


def scrub(text: str, thing: JSON, *, owner_is_member: bool) -> tuple[str, list[str]]:
    """Remove secrets, network identifiers and identifying values from free text.

    Returns the cleaned text and the secret-scanner rules that matched (an
    empty list when no secret was found).
    """
    if not text:
        return "", []
    result = redact(text)
    cleaned = result.text
    for name, placeholder in _PRIVATE_FIELDS:
        value = _text(thing.get(name))
        if len(value) >= 4:
            cleaned = re.sub(re.escape(value), placeholder, cleaned, flags=re.IGNORECASE)
    owner = _text(thing.get("owner"))
    if owner_is_member and len(owner) >= 2:
        cleaned = re.sub(rf"\b{re.escape(owner)}\b", "[OWNER]", cleaned, flags=re.IGNORECASE)
    cleaned = _MAC.sub("[MAC]", cleaned)
    cleaned = _IPV6.sub("[IP]", cleaned)
    cleaned = _IPV4.sub("[IP]", cleaned)
    cleaned = _LOCAL_HOST.sub("[HOST]", cleaned)
    return cleaned, result.rules


def parse_thing(
    thing: JSON,
    *,
    space_names: tuple[str, ...],
    geowiki_url: str | None = None,
) -> Asset:
    """An API thing as an asset record, with free text already scrubbed."""
    key = _text(thing.get("key")) or str(thing.get("id"))
    owner_class = ownership(_opt(thing.get("owner")), space_names)
    member = owner_class == "member-loan"
    secret_rules: set[str] = set()

    def clean(value: object) -> str:
        text, rules = scrub(_text(value), thing, owner_is_member=member)
        secret_rules.update(rules)
        return text

    links: list[Link] = []
    where = ""
    where_url = None
    for raw in thing.get("links") or []:
        if not isinstance(raw, dict):
            continue
        link = Link(
            type=_text(raw.get("type")) or "custom",
            title=clean(raw.get("title")) or _text(raw.get("type")).title(),
            url=_opt(raw.get("url")),
            note=clean(raw.get("note")) or None,
        )
        if link.type == "where":
            where = link.note or ""
            where_url = link.url
        links.append(link)

    location = location_id(where_url, geowiki_url)
    template = thing.get("template")
    related = [
        {
            "asset_id": _text(r.get("key")) or str(r.get("id")),
            "name": clean(r.get("name")),
            "note": clean(r.get("note")) or None,
        }
        for r in thing.get("related_things") or []
        if isinstance(r, dict)
    ]
    retired = sorted(
        {
            value
            for alias in thing.get("aliases") or []
            if isinstance(alias, dict)
            for value in (_text(alias.get("key")), _text(alias.get("slug")))
            if value
        }
    )
    photos = [
        {"url": p.get("url"), "content_type": p.get("content_type"), "filename": p.get("filename")}
        for p in thing.get("photos") or []
        if isinstance(p, dict) and p.get("url")
    ]
    asset = Asset(
        asset_id=key,
        thing_id=int(thing.get("id") or 0),
        name=clean(thing.get("name")) or key,
        url=_opt(thing.get("url")),
        slug=_opt(thing.get("slug")),
        short_url=_opt(thing.get("short_url")),
        label_name=clean(thing.get("label_name")) or None,
        manufacturer=clean(thing.get("manufacturer")) or None,
        model=clean(thing.get("model")) or None,
        category=_opt(template.get("name")) if isinstance(template, dict) else None,
        description=clean(thing.get("description")),
        notes=clean(thing.get("notes")),
        ownership=owner_class,
        where=where,
        where_url=where_url,
        location_ids=[location] if location else [],
        links=links,
        related=related,
        retired_ids=retired,
        photos=photos,
        manufacturer_url=_opt(thing.get("manufacturer_url")),
        integration_source=_opt(thing.get("integration_source")),
        created_at=_opt(thing.get("created_at")),
        updated_at=_opt(thing.get("updated_at")),
    )
    if secret_rules:
        asset.flags["secret_redacted"] = sorted(secret_rules)
    return asset


def facts(asset: Asset) -> str:
    """The deterministic facts block: what the index says about an asset."""
    lines = [asset.name]
    if asset.label_name and asset.label_name.lower() != asset.name.lower():
        lines.append(f"Also labelled: {asset.label_name}")
    if asset.make_model:
        lines.append(f"Make and model: {asset.make_model}")
    if asset.category:
        lines.append(f"Kind: {asset.category}")
    lines.append(
        "Owned by the space"
        if asset.ownership == "space"
        else "Owned by a member and on loan to the space"
    )
    if asset.where:
        lines.append(f"Where: {asset.where}")
    elif asset.location_ids:
        lines.append(f"Where: see {asset.where_url}")
    if asset.related:
        related = "; ".join(
            r["name"] + (f" ({r['note']})" if r.get("note") else "") for r in asset.related
        )
        lines.append(f"Related: {related}")
    titled = [link.title for link in asset.links if link.type not in {"where", "ar"} and link.url]
    if titled:
        lines.append(f"Links: {', '.join(titled)}")
    notes = [
        f"{link.title}: {link.note}"
        for link in asset.links
        if link.note and link.type not in {"where"}
    ]
    lines.extend(notes)
    lines.append(f"Asset tag: {asset.asset_id}")
    return "\n".join(lines)


def body(asset: Asset) -> str:
    parts = []
    if asset.description:
        parts.append(f"Description: {asset.description}")
    if asset.notes:
        parts.append(f"Notes: {asset.notes}")
    return "\n\n".join(parts)


def chunk_texts(
    asset: Asset, *, summary: str = "", chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Index texts: chunk 0 is the whole record (summary, facts, start of the body);
    long descriptions and notes continue in further chunks."""
    head = "\n\n".join(p for p in (summary, facts(asset)) if p)
    rest = body(asset)
    if not rest:
        return [head]
    if len(head) + len(rest) <= chunk_size * 2:
        return [f"{head}\n\n{rest}"]
    pieces = split_text(rest, chunk_size, chunk_overlap)
    return [f"{head}\n\n{pieces[0]}"] + [f"{asset.name}: {piece}" for piece in pieces[1:]]


def aliases(asset: Asset) -> list[str]:
    """Names the linker matches to this asset: name, label, make + model, model, slug."""
    found = {asset.name, asset.label_name or "", asset.make_model, asset.model or ""}
    if asset.slug:
        found.add(asset.slug.replace("-", " "))
    return sorted(a.strip() for a in found if a and a.strip())


def manual_candidates(
    asset: Asset, pattern: re.Pattern[str], *, include_manufacturer_url: bool
) -> list[ManualCandidate]:
    """Links that look like manuals, plus the manufacturer page to search for them."""
    found: list[ManualCandidate] = []
    for link in asset.links:
        if link.type != "custom" or not link.url:
            continue
        if pattern.search(link.title) or pattern.search(link.url):
            found.append(ManualCandidate(link.url, link.title, "link"))
    if include_manufacturer_url and asset.manufacturer_url:
        found.append(
            ManualCandidate(asset.manufacturer_url, asset.make_model or None, "manufacturer")
        )
    return found


def excluded(asset: Asset, include_integrations: tuple[str, ...]) -> str | None:
    """Why an asset is staged but not indexed, or None."""
    if asset.integration_source and asset.integration_source not in include_integrations:
        return f"imported by {asset.integration_source}"
    return None


def registry_row(asset: Asset, *, indexable: bool) -> JSON:
    return {
        "asset_id": asset.asset_id,
        "thing_id": asset.thing_id,
        "slug": asset.slug,
        "name": asset.name,
        "label_name": asset.label_name,
        "manufacturer": asset.manufacturer,
        "model": asset.model,
        "category": asset.category,
        "ownership": asset.ownership,
        "location_ids": asset.location_ids,
        "url": asset.url,
        "short_url": asset.short_url,
        "retired_ids": asset.retired_ids,
        "related_asset_ids": [r["asset_id"] for r in asset.related],
        "links": [
            {"type": link.type, "title": link.title, "url": link.url} for link in asset.links
        ],
        "indexable": indexable,
    }


def payload_extras(asset: Asset) -> JSON:
    """This source's fields on top of the shared payload."""
    return {
        "asset_key": asset.asset_id,
        "short_url": asset.short_url,
        "manufacturer": asset.manufacturer,
        "model": asset.model,
        "category": asset.category,
        "ownership": asset.ownership,
        "photos": asset.photos,
        "links": [
            {"type": link.type, "title": link.title, "url": link.url}
            for link in asset.links
            if link.url
        ],
        "related_asset_ids": [r["asset_id"] for r in asset.related],
        "retired_asset_ids": asset.retired_ids,
    }
