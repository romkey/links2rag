"""Parsed manuals -> index chunks, section kinds and summaries; drop-in manuals.

Pure functions apart from reading the drop-in directory. Chunks follow the
manual's own sections, one page at a time so every chunk has a page number;
tables and diagram descriptions are never split.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from links2rag.chunking import split_text
from links2rag.parse import ParsedManual, Section

logger = logging.getLogger(__name__)

# Bumped when parsing or chunking changes, so staged manuals are re-parsed.
PARSER_VERSION = "1"
INDEXED_LANGUAGES = frozenset({"en"})

# First match wins, so the specific kinds come before "operation".
_SECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "troubleshooting",
        re.compile(r"troubleshoot|problem|error code|fault|faq|diagnos", re.IGNORECASE),
    ),
    (
        "safety",
        re.compile(r"safety|warning|caution|danger|hazard|precaution", re.IGNORECASE),
    ),
    (
        "maintenance",
        re.compile(r"maintenance|maintain|cleaning|lubricat|\bcare\b|servic", re.IGNORECASE),
    ),
    (
        "specifications",
        re.compile(
            r"specification|technical data|\bspecs?\b|dimensions|\bratings?\b", re.IGNORECASE
        ),
    ),
    ("parts", re.compile(r"\bparts?\b|exploded|accessor|spare", re.IGNORECASE)),
    (
        "setup",
        re.compile(
            r"install|set ?up|assembl|unpack|getting started|quick ?start|connect", re.IGNORECASE
        ),
    ),
    (
        "operation",
        re.compile(r"operat|\busing\b|\buse\b|usage|controls|how to", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class ManualChunk:
    text: str
    heading_path: list[str]
    page: int | None
    kind: str  # text | table | diagram
    machine_generated: bool
    section: str | None
    lang: str | None


@dataclass(frozen=True)
class DropIn:
    path: Path
    asset_keys: list[str]
    title: str | None
    url: str | None


def section_kind(heading: str) -> str | None:
    """The manual_section kind a heading names, from keyword rules; None if unclear."""
    for kind, pattern in _SECTION_RULES:
        if pattern.search(heading):
            return kind
    return None


def indexable_language(lang: str | None) -> bool:
    """English, or too little text to tell (headings, tables of numbers)."""
    return lang is None or lang in INDEXED_LANGUAGES


def _text_chunks(
    text: str, section: Section, path: list[str], page: int | None, sizes: tuple[int, int]
) -> list[ManualChunk]:
    """A section's text on one page, split only when it is long."""
    chunk_size, chunk_overlap = sizes
    text = text.strip()
    if not text:
        return []
    heading = f"{section.heading}\n" if section.heading else ""
    pieces = (
        [text] if len(text) <= chunk_size * 1.3 else split_text(text, chunk_size, chunk_overlap)
    )
    kind = section_kind(section.heading)
    return [
        ManualChunk(heading + piece, path, page, "text", False, kind, section.lang)
        for piece in pieces
    ]


def chunk_manual(parsed: ParsedManual, *, chunk_size: int, chunk_overlap: int) -> list[ManualChunk]:
    sizes = (chunk_size, chunk_overlap)
    chunks: list[ManualChunk] = []
    for section in parsed.sections:
        if not indexable_language(section.lang):
            continue
        path = [p for p in (parsed.title, section.heading) if p]
        kind = section_kind(section.heading)
        pending: list[str] = []
        pending_page: int | None = None
        for block in section.blocks:
            if block.kind == "text":
                if pending and block.page != pending_page:
                    text = "\n".join(pending)
                    chunks.extend(_text_chunks(text, section, path, pending_page, sizes))
                    pending = []
                pending_page = block.page
                pending.append(block.text)
                continue
            chunks.extend(_text_chunks("\n".join(pending), section, path, pending_page, sizes))
            pending = []
            block_kind = kind
            if block.kind == "table" and block_kind is None:
                block_kind = "specifications"
            chunks.append(
                ManualChunk(
                    block.text,
                    path,
                    block.page,
                    block.kind,
                    block.machine_generated,
                    block_kind,
                    section.lang,
                )
            )
        chunks.extend(_text_chunks("\n".join(pending), section, path, pending_page, sizes))
    return chunks


def summary_text(parsed: ParsedManual, *, title: str, assets: list[str]) -> str:
    """The manual's document-level point: what it is, for what, and what it covers."""
    lines = [f"Manual: {title}"]
    if assets:
        lines.append(f"For: {'; '.join(assets)}")
    if parsed.pages:
        lines.append(f"{parsed.pages} pages")
    headings = [s.heading for s in parsed.sections if s.heading and indexable_language(s.lang)]
    if headings:
        lines.append("Sections: " + "; ".join(headings[:60]))
    other = sorted({s.lang for s in parsed.sections if s.lang and not indexable_language(s.lang)})
    if other:
        lines.append(f"Also in: {', '.join(other)} (not indexed)")
    return "\n".join(lines)


def page_url(url: str | None, page: int | None, kind: str) -> str | None:
    """A link that opens a PDF at the chunk's page."""
    if not url:
        return None
    if kind == "pdf" and page:
        return f"{url.split('#')[0]}#page={page}"
    return url


# -- drop-in manuals -------------------------------------------------------------

_SIDE_SUFFIXES = (".yaml", ".yml", ".json")


# Any: sidecar values are strings or lists of strings.
def _parse_yaml(text: str) -> dict[str, Any]:
    """The tiny subset of YAML sidecars use: ``key: value``, ``key: [a, b]`` and lists."""
    # Any: see above.
    data: dict[str, Any] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current:
            data.setdefault(current, [])
            if isinstance(data[current], list):
                data[current].append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", 1))
        current = key
        if value.startswith("[") and value.endswith("]"):
            data[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        elif value:
            data[key] = value.strip("'\"")
    return data


def read_sidecar(path: Path) -> DropIn | None:
    """A drop-in manual from its sidecar; None if the sidecar is unusable."""
    sidecar = next(
        (
            path.with_name(path.name + s)
            for s in _SIDE_SUFFIXES
            if path.with_name(path.name + s).exists()
        ),
        None,
    )
    if sidecar is None:
        return None
    try:
        text = sidecar.read_text(encoding="utf-8")
        data = json.loads(text) if sidecar.suffix == ".json" else _parse_yaml(text)
    except (OSError, ValueError) as e:
        logger.warning("Cannot read sidecar %s: %s", sidecar, e)
        return None
    if not isinstance(data, dict):
        return None
    keys = data.get("asset_keys") or data.get("assets") or data.get("asset_key") or []
    if isinstance(keys, str):
        keys = [keys]
    keys = [str(k).strip() for k in keys if str(k).strip()]
    if not keys:
        logger.warning("Sidecar %s names no asset_keys; skipping %s", sidecar, path.name)
        return None
    title = data.get("title")
    url = data.get("url")
    return DropIn(
        path=path,
        asset_keys=keys,
        title=str(title) if title else None,
        url=str(url) if url else None,
    )


def read_dropins(directory: str | None) -> list[DropIn]:
    """Manuals placed by hand: each file with a ``<file>.yaml`` or ``.json`` sidecar."""
    if not directory:
        return []
    root = Path(directory)
    if not root.is_dir():
        logger.warning("MANUALS_DIR %s is not a directory", directory)
        return []
    found: list[DropIn] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in _SIDE_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        dropin = read_sidecar(path)
        if dropin is not None:
            found.append(dropin)
    return found
