"""Reports: assets missing links, failing manual links, quarantine, aliases to review.

Built from the registry and staging after each full run, written as JSON to
``DATA_DIR/reports/``, and printed by `links2rag report`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from links2rag.db import Database
from links2rag.documents import ASSET_SOURCE, MANUAL_SOURCE, now_rfc3339
from links2rag.manual_sync import DEAD_STATUSES

logger = logging.getLogger(__name__)

# Any: report rows are JSON.
JSON = dict[str, Any]


def build(db: Database) -> JSON:
    assets = [a for a in db.assets() if a["indexable"]]
    links = db.manual_links()
    with_manual = {
        asset_id for row in links.values() if row["manual_id"] for asset_id in row["asset_ids"]
    }

    def brief(asset: JSON) -> JSON:
        return {"asset_id": asset["asset_id"], "name": asset["name"], "url": asset["url"]}

    def has(asset: JSON, link_type: str) -> bool:
        return any(link.get("type") == link_type and link.get("url") for link in asset["links"])

    return {
        "generated_at": now_rfc3339(),
        "assets": len(assets),
        "no_wiki_page": [brief(a) for a in assets if not has(a, "wiki")],
        "no_manual": [brief(a) for a in assets if a["asset_id"] not in with_manual],
        "no_location": [brief(a) for a in assets if not a["location_ids"]],
        # The asset manager has no training field yet: every asset is listed
        # until the member manager's trainings are linked (plan 06).
        "no_training": [brief(a) for a in assets],
        "manual_links_failing": [
            {
                "url": url,
                "status": row["status"],
                "http_status": row["http_status"],
                "error": row["error"],
                "asset_ids": row["asset_ids"],
                "last_ok_at": row["last_ok_at"],
                "still_indexed": bool(row["manual_id"]),
            }
            for url, row in sorted(links.items())
            if row["status"] in DEAD_STATUSES
        ],
        "quarantine": [
            {
                "source": r["source"],
                "doc_id": r["source_doc_id"],
                "reason": r["reason"],
                "detail": r["detail"],
                "at": r["created_at"],
            }
            for r in db.quarantine_rows([ASSET_SOURCE, MANUAL_SOURCE])
        ],
        "aliases_to_review": [
            {"id": r["id"], "asset_id": r["left_id"], "alias": r["right_id"], **r["details"]}
            for r in db.alias_queue()
        ],
    }


def write(report: JSON, data_dir: str) -> Path | None:
    """Write the report as DATA_DIR/reports/links2rag.json; None if that fails."""
    path = Path(data_dir) / "reports" / "links2rag.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    except OSError as e:
        logger.warning("Cannot write report %s: %s", path, e)
        return None
    return path


def summary(report: JSON) -> str:
    return (
        f"{report['assets']} assets: {len(report['no_wiki_page'])} without a wiki page, "
        f"{len(report['no_manual'])} without a manual, {len(report['no_location'])} without a "
        f"location; {len(report['manual_links_failing'])} manual links failing, "
        f"{len(report['quarantine'])} quarantined, {len(report['aliases_to_review'])} "
        "aliases to review"
    )
