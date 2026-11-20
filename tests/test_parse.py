from links2rag.parse import (
    MACHINE_GENERATED,
    Block,
    ParsedManual,
    Section,
    detect_language,
    parse_html,
    parse_pdf,
    table_text,
)
from tests.pdfgen import ENGLISH, GERMAN, Page, build_pdf


def _headings(parsed: ParsedManual) -> list[str]:
    return [s.heading for s in parsed.sections]


def test_detect_language() -> None:
    assert detect_language(ENGLISH) == "en"
    assert detect_language(GERMAN) == "de"
    assert detect_language("Cut 3 mm") is None
    assert detect_language("Lorem ipsum dolor sit amet consectetur adipiscing elit sed") is None


def test_pdf_headings_from_font_size() -> None:
    data = build_pdf(
        [
            Page([("Safety", 20), (ENGLISH, 10), ("Operation", 20), ("Turn it on.", 10)]),
            Page([("Maintenance", 20), ("Clean the table weekly.", 10)]),
        ],
        title="Bandsaw Manual",
    )
    parsed = parse_pdf(data, title="fallback")
    assert parsed.title == "Bandsaw Manual"
    assert parsed.pages == 2
    assert _headings(parsed) == ["Safety", "Operation", "Maintenance"]
    assert parsed.sections[0].lang == "en"
    assert parsed.sections[2].blocks[0].page == 2
    assert parsed.sections[2].first_page == 2


def test_pdf_outline_wins_over_font_sizes() -> None:
    data = build_pdf(
        [Page([("Big words", 20), ("Intro text here.", 10)]), Page([("Specs here.", 10)])],
        outline=[("Introduction", 0), ("Specifications", 1)],
    )
    parsed = parse_pdf(data, title="T")
    assert parsed.title == "T"
    assert _headings(parsed) == ["Introduction", "Specifications"]
    assert "Big words" in parsed.sections[0].text


def test_pdf_tables_are_kept_whole_with_caption() -> None:
    table = [
        ["Material", "Power", "Speed"],
        ["Acrylic 3mm", "60%", "20"],
        ["Plywood 3mm", "70%", "15"],
    ]
    data = build_pdf([Page([("Cut settings", 10)], table=table)])
    parsed = parse_pdf(data)
    tables = [b for s in parsed.sections for b in s.blocks if b.kind == "table"]
    assert len(tables) == 1
    text = tables[0].text
    assert text.startswith("Table (page 1): Cut settings")
    assert "Material | Power | Speed" in text
    assert "Acrylic 3mm | 60% | 20" in text
    # The table's text is not repeated as loose page text.
    loose = "\n".join(b.text for s in parsed.sections for b in s.blocks if b.kind == "text")
    assert "Acrylic" not in loose


def test_pdf_ocr_for_pages_without_text() -> None:
    data = build_pdf([Page([]), Page([("Real text on this page for sure.", 10)])])
    seen: list[bytes] = []

    def ocr(png: bytes) -> str:
        seen.append(png)
        return "Scanned  page\n\n\ntext from OCR"

    parsed = parse_pdf(data, ocr=ocr)
    assert parsed.ocr_pages == [1]
    assert seen[0].startswith(b"\x89PNG")
    assert "Scanned page\ntext from OCR" in parsed.text
    # OCR failures are logged and skipped.

    def broken(png: bytes) -> str:
        raise RuntimeError("no tesseract")

    assert parse_pdf(data, ocr=broken).ocr_pages == []


def test_pdf_diagram_pages_get_a_description() -> None:
    data = build_pdf(
        [Page([("Fig. 3", 10)], shapes=50), Page([(ENGLISH, 10), (ENGLISH, 10)], shapes=50)]
    )

    def describe(png: bytes, context: str) -> str | None:
        return f"An exploded view ({context})."

    parsed = parse_pdf(data, describe=describe)
    assert parsed.diagram_pages == [1]
    diagram = [b for s in parsed.sections for b in s.blocks if b.kind == "diagram"]
    assert diagram[0].text == f"{MACHINE_GENERATED} An exploded view (page 1)."
    assert diagram[0].machine_generated

    def broken(png: bytes, context: str) -> str | None:
        raise RuntimeError("vision down")

    assert parse_pdf(data, describe=broken).diagram_pages == []


def test_parsed_manual_round_trips_json() -> None:
    parsed = ParsedManual(
        title="M",
        sections=[Section("S", [Block("t", 2, "table", False)], "en")],
        pages=3,
        ocr_pages=[1],
        diagram_pages=[2],
    )
    again = ParsedManual.from_json(parsed.to_json())
    assert again == parsed
    assert ParsedManual.from_json({}).sections == []


def test_table_text() -> None:
    assert table_text([["a", None], ["", ""], ["b\nc", "d"]], "", None) == "Table\na | \nb c | d"


def test_parse_html() -> None:
    html = b"""<html><head><title>Rikon 10-305 manual</title><script>x()</script></head>
    <body><nav>Home</nav><main>
      <p>Intro paragraph about the saw.</p>
      <h2>Safety rules</h2><p>Keep the guard on.</p><p>Wear glasses.</p>
      <ul><li><p>Item in a list</p></li></ul>
      <h2>Blade sizes</h2>
      <table><caption>Blades</caption><tr><th>Width</th><th>TPI</th></tr>
        <tr><td>1/4"</td><td>6</td></tr></table>
      <table><tr><td>lonely</td></tr></table>
      <h3></h3>
    </main></body></html>"""
    parsed = parse_html(html, title="fallback")
    assert parsed.kind == "html"
    assert parsed.title == "Rikon 10-305 manual"
    assert _headings(parsed) == ["", "Safety rules", "Blade sizes"]
    assert parsed.sections[1].blocks[0].text == "Keep the guard on.\nWear glasses.\nItem in a list"
    table = parsed.sections[2].blocks[0]
    assert table.kind == "table"
    assert table.text == 'Table: Blades\nWidth | TPI\n1/4" | 6'
    assert parse_html(b"<h1>Only</h1><p>x</p>").title == "Only"
    assert parse_html(b"<p>x</p>", title="T").title == "T"
