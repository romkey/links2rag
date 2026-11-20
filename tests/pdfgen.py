"""Builds small PDFs for the parser tests: lines of text, ruled tables, an outline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Page:
    # (text, font size) from the top of the page down.
    lines: list[tuple[str, float]] = field(default_factory=list)
    # Rows of cells drawn as a ruled grid below the text.
    table: list[list[str]] | None = None
    # Extra drawn lines (shapes), to look like a diagram.
    shapes: int = 0


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content(page: Page) -> bytes:
    ops: list[str] = []
    y = 750.0
    for text, size in page.lines:
        ops.append(f"BT /F1 {size} Tf 50 {y:.1f} Td ({_escape(text)}) Tj ET")
        y -= size * 1.6
    if page.table:
        rows = page.table
        cols = max(len(r) for r in rows)
        width, height = 120.0, 20.0
        top = y - 20
        left = 50.0
        for r in range(len(rows) + 1):
            yy = top - r * height
            ops.append(f"{left} {yy} m {left + cols * width} {yy} l S")
        for c in range(cols + 1):
            xx = left + c * width
            ops.append(f"{xx} {top} m {xx} {top - len(rows) * height} l S")
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                x = left + c * width + 5
                yy = top - (r + 1) * height + 6
                ops.append(f"BT /F1 9 Tf {x} {yy} Td ({_escape(cell)}) Tj ET")
    for n in range(page.shapes):
        ops.append(f"{50 + n * 3} 100 m {60 + n * 3} 300 l S")
    return "\n".join(ops).encode("latin-1")


def build_pdf(
    pages: list[Page],
    *,
    title: str | None = None,
    outline: list[tuple[str, int]] | None = None,
) -> bytes:
    """A PDF; ``outline`` is (title, 0-based page index) entries, one level."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # filled in below
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    for page in pages:
        stream = _content(page)
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            add(
                (
                    f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
                ).encode()
            )
        )
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects[pages_obj - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    outline_ref = ""
    if outline:
        root = add(b"")
        items = [add(b"") for _ in outline]
        for n, (heading, index) in enumerate(outline):
            parts = [
                f"/Title ({_escape(heading)})",
                f"/Parent {root} 0 R",
                f"/Dest [{page_ids[index]} 0 R /XYZ 0 792 0]",
            ]
            if n > 0:
                parts.append(f"/Prev {items[n - 1]} 0 R")
            if n < len(items) - 1:
                parts.append(f"/Next {items[n + 1]} 0 R")
            objects[items[n] - 1] = ("<< " + " ".join(parts) + " >>").encode()
        objects[root - 1] = (
            f"<< /Type /Outlines /First {items[0]} 0 R /Last {items[-1]} 0 R /Count {len(items)} >>"
        ).encode()
        outline_ref = f" /Outlines {root} 0 R"
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R{outline_ref} >>".encode()
    info = add(f"<< /Title ({_escape(title)}) >>".encode()) if title else None

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"<< /Size {len(objects) + 1} /Root {catalog} 0 R"
    if info:
        trailer += f" /Info {info} 0 R"
    out += f"trailer\n{trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


ENGLISH = (
    "Always wear eye protection when you operate the saw. Keep your hands away from "
    "the blade and use the push stick for narrow cuts. The guard must be in place."
)
GERMAN = (
    "Tragen Sie immer eine Schutzbrille, wenn Sie die Säge bedienen. Halten Sie die "
    "Hände von dem Sägeblatt fern und benutzen Sie den Schiebestock für schmale Schnitte."
)
