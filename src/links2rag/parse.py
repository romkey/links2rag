"""Manual files -> sections of text and tables, page by page.

PDFs: text with layout awareness (pdfplumber), tables extracted as tables
with their caption and header row, the outline (pypdf) or, without one,
large-font lines as section headings. Pages with almost no text layer are
OCRed; pages that are mostly a diagram get a short vision-model description
when a describer is given. HTML: headings split sections; tables are kept
whole.

Each section is tagged with its language; only English is indexed.
"""

from __future__ import annotations

import io
import logging
import re
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pdfplumber
from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

logger = logging.getLogger(__name__)

MACHINE_GENERATED = "[Machine-generated description of a diagram]"
# A page with less text than this has no usable text layer (a scan) or is a diagram.
_MIN_TEXT = 40
_DIAGRAM_TEXT = 250
_DIAGRAM_SHAPES = 40
_HEADING_RATIO = 1.25
_MAX_HEADING = 90
_OCR_RESOLUTION = 200
# How far above a table (in points) its caption may be.
_CAPTION_REACH = 60

# (png bytes, context) -> description, or None.
Describe = Callable[[bytes, str], str | None]
# (png bytes) -> text.
Ocr = Callable[[bytes], str]

# Function words of the languages manuals most often come in; the language
# with the most hits wins. English needs a clear margin over the others.
_LANG_WORDS = {
    "en": "the and of to is in for with on be this that are not you your it as or by at from",
    "de": "der die und das ist nicht mit sie den zu ein eine für auf dem des werden oder bei",
    "fr": "le la les et des est pour une dans pas avec sur vous ne que du au ou par",
    "es": "el la los las y de que en un una para con por no es del se su al o",
    "it": "il la di che e per un una non con del della sono le gli da in",
    "nl": "de het een en van is niet op te voor met dat zijn bij of aan",
}
_LANG_SETS = {lang: frozenset(words.split()) for lang, words in _LANG_WORDS.items()}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SPACES = re.compile(r"[ \t]+")


@dataclass
class Block:
    """Text or a table on one page (page is None for HTML)."""

    text: str
    page: int | None
    kind: str = "text"  # text | table | diagram
    machine_generated: bool = False


@dataclass
class Section:
    heading: str
    blocks: list[Block] = field(default_factory=list)
    lang: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def first_page(self) -> int | None:
        return next((b.page for b in self.blocks if b.page is not None), None)


@dataclass
class ParsedManual:
    title: str
    sections: list[Section]
    pages: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    diagram_pages: list[int] = field(default_factory=list)
    kind: str = "pdf"

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)

    # Any: staged as JSON.
    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "kind": self.kind,
            "pages": self.pages,
            "ocr_pages": self.ocr_pages,
            "diagram_pages": self.diagram_pages,
            "sections": [
                {
                    "heading": s.heading,
                    "lang": s.lang,
                    "blocks": [
                        {
                            "text": b.text,
                            "page": b.page,
                            "kind": b.kind,
                            "machine_generated": b.machine_generated,
                        }
                        for b in s.blocks
                    ],
                }
                for s in self.sections
            ],
        }

    @classmethod
    # Any: staged JSON.
    def from_json(cls, data: dict[str, Any]) -> ParsedManual:
        return cls(
            title=str(data.get("title") or ""),
            kind=str(data.get("kind") or "pdf"),
            pages=int(data.get("pages") or 0),
            ocr_pages=list(data.get("ocr_pages") or []),
            diagram_pages=list(data.get("diagram_pages") or []),
            sections=[
                Section(
                    heading=str(s.get("heading") or ""),
                    lang=s.get("lang"),
                    blocks=[
                        Block(
                            text=str(b.get("text") or ""),
                            page=b.get("page"),
                            kind=str(b.get("kind") or "text"),
                            machine_generated=bool(b.get("machine_generated")),
                        )
                        for b in s.get("blocks") or []
                    ],
                )
                for s in data.get("sections") or []
            ],
        )


def detect_language(text: str) -> str | None:
    """Best guess among a few common manual languages; None for too little text."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 8:
        return None
    hits = Counter(
        {lang: sum(1 for w in words if w in vocab) for lang, vocab in _LANG_SETS.items()}
    )
    lang, best = hits.most_common(1)[0]
    if best < max(2, len(words) // 40):
        return None
    return lang


def _clean(text: str) -> str:
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def table_text(rows: list[list[str | None]], caption: str, page: int | None) -> str:
    """A table as text: caption, then the header row, then each row, cells joined by |."""
    cleaned = [[_SPACES.sub(" ", (c or "").replace("\n", " ")).strip() for c in r] for r in rows]
    cleaned = [r for r in cleaned if any(r)]
    where = f" (page {page})" if page else ""
    head = f"Table{where}: {caption}" if caption else f"Table{where}"
    return "\n".join([head, *(" | ".join(r) for r in cleaned)])


# -- PDF -----------------------------------------------------------------------


def _outline(data: bytes) -> list[tuple[int, str, int]]:
    """(level, title, 1-based page) for each outline entry, in page order."""
    try:
        reader = PdfReader(io.BytesIO(data))
        entries: list[tuple[int, str, int]] = []

        # Any: pypdf outline items are Destinations or nested lists.
        def walk(items: list[Any], level: int) -> None:
            for item in items:
                if isinstance(item, list):
                    walk(item, level + 1)
                    continue
                page = reader.get_destination_page_number(item)
                title = str(getattr(item, "title", "") or "").strip()
                if title and page is not None and page >= 0:
                    entries.append((level, title, page + 1))

        walk(reader.outline, 0)
    except Exception as e:
        logger.debug("No usable outline: %s", e)
        return []
    return sorted(entries, key=lambda e: e[2])


# Any: pdfplumber page and word objects are untyped.
def _lines(page: Any, tables: list[Any]) -> list[tuple[str, float]]:
    """(text, font size) per line of the page, outside tables."""
    region = page
    for table in tables:
        try:
            region = region.outside_bbox(table.bbox)
        except ValueError:
            continue
    words = region.extract_words(extra_attrs=["size"], keep_blank_chars=False, use_text_flow=True)
    lines: dict[int, list[Any]] = {}
    for word in words:
        lines.setdefault(round(word["top"] / 3), []).append(word)
    out: list[tuple[str, float]] = []
    for key in sorted(lines):
        row = sorted(lines[key], key=lambda w: w["x0"])
        out.append((" ".join(w["text"] for w in row), max(float(w["size"]) for w in row)))
    return out


def _png(page: Any) -> bytes:
    buffer = io.BytesIO()
    page.to_image(resolution=_OCR_RESOLUTION).original.save(buffer, format="PNG")
    return buffer.getvalue()


def _caption(page: Any, bbox: tuple[float, float, float, float]) -> str:
    top = bbox[1]
    try:
        above = page.crop((0, max(0, top - _CAPTION_REACH), page.width, top)).extract_text() or ""
    except ValueError:
        return ""
    lines = [line.strip() for line in above.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_pdf(
    data: bytes,
    *,
    title: str = "",
    ocr: Ocr | None = None,
    describe: Describe | None = None,
) -> ParsedManual:
    outline = _outline(data)
    starts: dict[int, list[str]] = {}
    for _, heading, start_page in outline:
        starts.setdefault(start_page, []).append(heading)
    sections: list[Section] = [Section(heading="")]
    ocr_pages: list[int] = []
    diagram_pages: list[int] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        meta_title = str((pdf.metadata or {}).get("Title") or "").strip()
        pages = list(pdf.pages)
        page_lines: list[list[tuple[str, float]]] = []
        page_tables: list[list[Block]] = []
        for number, page in enumerate(pages, start=1):
            found = page.find_tables()
            tables: list[Block] = []
            for table in found:
                rows = table.extract()
                if rows and len(rows) >= 2:
                    caption = _caption(page, table.bbox)
                    tables.append(Block(table_text(rows, caption, number), number, "table"))
            lines = _lines(page, found)
            text_len = sum(len(t) for t, _ in lines)
            # A scan is a page image with no text layer.
            if ocr is not None and text_len < _MIN_TEXT and (page.images or text_len == 0):
                try:
                    text = ocr(_png(page))
                except Exception as e:
                    logger.warning("OCR failed on page %d: %s", number, e)
                    text = ""
                if text.strip():
                    ocr_pages.append(number)
                    lines = [(line, 0.0) for line in _clean(text).splitlines() if line]
                    text_len = sum(len(t) for t, _ in lines)
            shapes = len(page.curves) + len(page.lines) + len(page.rects) + len(page.images)
            if (
                describe is not None
                and text_len < _DIAGRAM_TEXT
                and not tables
                and (page.images or shapes >= _DIAGRAM_SHAPES)
            ):
                try:
                    note = describe(_png(page), f"page {number}")
                except Exception as e:
                    logger.warning("Diagram description failed on page %d: %s", number, e)
                    note = None
                if note:
                    diagram_pages.append(number)
                    tables.append(Block(f"{MACHINE_GENERATED} {note}", number, "diagram", True))
            page_lines.append(lines)
            page_tables.append(tables)

        # Body text size: the median over characters, so a few big headings
        # don't count as much as the paragraphs under them.
        sizes = [s for lines in page_lines for t, s in lines if s > 0 for _ in range(len(t))]
        body = statistics.median(sizes) if sizes else 0.0

        for number, (lines, tables) in enumerate(zip(page_lines, page_tables, strict=True), 1):
            for heading in starts.get(number, []):
                sections.append(Section(heading=heading))
            paragraph: list[str] = []

            def flush(page_number: int = number, buffer: list[str] = paragraph) -> None:
                text = _clean("\n".join(buffer))
                if text:
                    sections[-1].blocks.append(Block(text, page_number))
                buffer.clear()

            for text, size in lines:
                is_heading = (
                    not outline
                    and body > 0
                    and size >= body * _HEADING_RATIO
                    and len(text) <= _MAX_HEADING
                    and any(c.isalpha() for c in text)
                )
                if is_heading:
                    flush()
                    sections.append(Section(heading=text.strip()))
                else:
                    paragraph.append(text)
            flush()
            sections[-1].blocks.extend(tables)

    sections = [s for s in sections if s.blocks]
    for section in sections:
        section.lang = detect_language(section.heading + "\n" + section.text)
    return ParsedManual(
        title=meta_title or title,
        sections=sections,
        pages=len(page_lines),
        ocr_pages=ocr_pages,
        diagram_pages=diagram_pages,
    )


# -- HTML ----------------------------------------------------------------------

_HEADINGS = ("h1", "h2", "h3", "h4")
_DROP = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg")


def html_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if isinstance(h1, Tag) else ""


def parse_html(data: bytes, *, title: str = "") -> ParsedManual:
    soup = BeautifulSoup(data, "html.parser")
    page_title = html_title(soup)
    for tag in soup.find_all(_DROP):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    sections: list[Section] = [Section(heading="")]
    for element in root.find_all([*_HEADINGS, "p", "li", "pre", "table", "dd", "dt"]):
        if not isinstance(element, Tag):
            continue
        # Text inside a table or list item is taken with its parent.
        if element.name != "table" and element.find_parent("table"):
            continue
        if element.name in ("p", "pre") and element.find_parent("li"):
            continue
        if element.name in _HEADINGS:
            heading = element.get_text(" ", strip=True)
            if heading:
                sections.append(Section(heading=heading))
            continue
        if element.name == "table":
            rows = [
                [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in element.find_all("tr")
            ]
            caption_tag = element.find("caption")
            caption = caption_tag.get_text(" ", strip=True) if isinstance(caption_tag, Tag) else ""
            if len(rows) >= 2:
                sections[-1].blocks.append(
                    Block(table_text([list(r) for r in rows], caption, None), None, "table")
                )
            continue
        text = element.get_text(" ", strip=True)
        if text:
            sections[-1].blocks.append(Block(text, None))
    # Merge consecutive paragraphs of a section into one block.
    for section in sections:
        merged: list[Block] = []
        for block in section.blocks:
            if merged and block.kind == "text" and merged[-1].kind == "text":
                merged[-1] = Block(f"{merged[-1].text}\n{block.text}", None)
            else:
                merged.append(block)
        section.blocks = merged
        section.lang = detect_language(section.heading + "\n" + section.text)
    return ParsedManual(
        title=page_title or title,
        sections=[s for s in sections if s.blocks],
        kind="html",
    )
