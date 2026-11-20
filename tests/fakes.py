"""Fake asset manager data, client and embedder shared by the tests."""

import copy
import hashlib
from typing import Any

from links2rag.embeddings import EmbedKind

# Any: these build and return the asset manager's API JSON, which is free-form.
type JSON = dict[str, Any]

# Canaries: none of these may appear in any indexed text or payload.
SERIAL = "SN-CANARY-90210"
OWNER = "Pat Canary"
IP = "10.42.0.17"
HOSTNAME = "laser-canary"
IEEE = "aa:bb:cc:dd:ee:ff"


def thing(
    key: str = "abcd1234",
    *,
    thing_id: int = 1,
    name: str = "Big Laser",
    updated_at: str = "2026-11-01T00:00:00Z",
    description: str = "A 100 W CO2 laser cutter for wood and acrylic.",
    notes: str = "Never cut PVC. Air assist must be on.",
    owner: str | None = None,
    manufacturer: str | None = "Thunder Laser",
    model: str | None = "Nova 51",
    links: list[JSON] | None = None,
    template: str | None = "Laser cutter",
    integration_source: str | None = None,
    related: list[JSON] | None = None,
    aliases: list[JSON] | None = None,
    manufacturer_url: str | None = None,
) -> JSON:
    return {
        "id": thing_id,
        "key": key,
        "slug": name.lower().replace(" ", "-"),
        "name": name,
        "label_name": None,
        "description": description,
        "notes": notes,
        "owner": owner,
        "manufacturer": manufacturer,
        "model": model,
        "serial_number": SERIAL,
        "manufacturer_url": manufacturer_url,
        "ip_address": IP,
        "reserved_ip_address": None,
        "network_name": None,
        "hostname": HOSTNAME,
        "ieee_address": IEEE,
        "ble_beacon_uuid": None,
        "ar_anchor_note": None,
        "public_access": False,
        "integration_source": integration_source,
        "labelled_at": None,
        "visit_count": 3,
        "qr_scan_count": 5,
        "nfc_scan_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": updated_at,
        "url": f"https://links.example.com/things/{key}",
        "short_url": f"https://links.example.com/{key}",
        "template": {"id": 1, "name": template} if template else None,
        "links": links
        if links is not None
        else [
            {
                "id": 1,
                "type": "wiki",
                "title": "Wiki",
                "url": "https://wiki.example.com/laser",
                "note": None,
            },
            {
                "id": 2,
                "type": "where",
                "title": "Where",
                "url": "https://geowiki.example.com/resources/12",
                "note": "Section 2, by the back door",
            },
        ],
        "related_things": related or [],
        "aliases": aliases or [],
        "photos": [
            {
                "filename": "laser.jpg",
                "content_type": "image/jpeg",
                "byte_size": 100,
                "url": "https://links.example.com/rails/active_storage/blobs/x/laser.jpg",
            }
        ],
        "ar_anchor": None,
        "integrations": {
            "unifi_devices": [],
            "unifi_hosts": [{"id": 1, "ip_address": IP, "hostname": HOSTNAME}],
            "zigbee2mqtt_devices": [],
            "prusa_connect_printers": [],
        },
    }


class FakeLinksClient:
    """Stands in for LinksClient: an in-memory asset manager."""

    def __init__(self, *things: JSON) -> None:
        self.things: dict[str, JSON] = {t["key"]: t for t in things}
        self.calls = 0
        self.fail = False

    def add(self, item: JSON) -> None:
        self.things[item["key"]] = item

    def list_things(self) -> list[JSON]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("listing failed")
        return [copy.deepcopy(t) for t in self.things.values()]


class FakeEmbedder:
    """Deterministic 8-dimension vectors from a hash of the text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        self.calls.append(list(chunks))
        out = []
        for t in chunks:
            digest = hashlib.sha256(t.encode()).digest()
            out.append([b / 255 + 0.01 for b in digest[:8]])
        return out

    @property
    def texts(self) -> list[str]:
        return [t for call in self.calls for t in call]
