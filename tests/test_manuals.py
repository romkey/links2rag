from pathlib import Path

from links2rag.manuals import (
    chunk_manual,
    indexable_language,
    page_url,
    read_dropins,
    read_sidecar,
    section_kind,
    summary_text,
)
from links2rag.parse import Block, ParsedManual, Section


def test_section_kind() -> None:
    assert section_kind("Safety Instructions") == "safety"
    assert section_kind("Troubleshooting guide") == "troubleshooting"
    assert section_kind("Routine maintenance") == "maintenance"
    assert section_kind("Technical data") == "specifications"
    assert section_kind("Replacement parts list") == "parts"
    assert section_kind("Assembly") == "setup"
    assert section_kind("Operating the saw") == "operation"
    assert section_kind("Introduction") is None


def test_chunk_manual() -> None:
    parsed = ParsedManual(
        title="Rikon 10-305",
        sections=[
            Section(
                "Safety rules",
                [
                    Block("Keep the guard on.", 2),
                    Block("Wear glasses.", 2),
                    Block("Unplug before changing blades.", 3),
                    Block("Table (page 3)\nA | B\n1 | 2", 3, "table"),
                ],
                "en",
            ),
            Section("Sicherheit", [Block("Tragen Sie eine Brille.", 4)], "de"),
            Section("", [Block("Spec table", 5, "table"), Block("x " * 400, 6)], None),
        ],
        pages=6,
    )
    chunks = chunk_manual(parsed, chunk_size=300, chunk_overlap=30)
    assert [(c.page, c.kind, c.section) for c in chunks[:3]] == [
        (2, "text", "safety"),
        (3, "text", "safety"),
        (3, "table", "safety"),
    ]
    assert chunks[0].text == "Safety rules\nKeep the guard on.\nWear glasses."
    assert chunks[0].heading_path == ["Rikon 10-305", "Safety rules"]
    assert all("Brille" not in c.text for c in chunks)  # German is not indexed
    assert chunks[3].section == "specifications"  # an untitled table
    assert chunks[3].heading_path == ["Rikon 10-305"]
    assert len(chunks) > 5  # the long untitled text was split
    summary = summary_text(parsed, title="Rikon 10-305", assets=["Bandsaw (Rikon 10-305)"])
    assert summary == (
        "Manual: Rikon 10-305\nFor: Bandsaw (Rikon 10-305)\n6 pages\n"
        "Sections: Safety rules\nAlso in: de (not indexed)"
    )
    assert summary_text(ParsedManual("M", []), title="M", assets=[]) == "Manual: M"


def test_language_and_page_urls() -> None:
    assert indexable_language(None)
    assert indexable_language("en")
    assert not indexable_language("fr")
    assert page_url(None, 3, "pdf") is None
    assert (
        page_url("https://v.example.com/m.pdf#x", 3, "pdf") == "https://v.example.com/m.pdf#page=3"
    )
    assert page_url("https://v.example.com/m", 3, "html") == "https://v.example.com/m"
    assert page_url("https://v.example.com/m.pdf", None, "pdf") == "https://v.example.com/m.pdf"


def test_dropins(tmp_path: Path) -> None:
    (tmp_path / "saw.pdf").write_bytes(b"%PDF")
    (tmp_path / "saw.pdf.yaml").write_text(
        "# the bandsaw\ntitle: 'Saw manual'\nurl: https://v.example.com/saw.pdf\n"
        "asset_keys:\n  - abcd1234  # main saw\n  - efgh5678\n"
    )
    (tmp_path / "laser.pdf").write_bytes(b"%PDF")
    (tmp_path / "laser.pdf.json").write_text('{"asset_key": "laser123"}')
    (tmp_path / "drill.pdf").write_bytes(b"%PDF")
    (tmp_path / "drill.pdf.yml").write_text("asset_keys: [drill001, drill002]\nnot yaml line\n")
    (tmp_path / "nosidecar.pdf").write_bytes(b"%PDF")
    (tmp_path / "empty.pdf").write_bytes(b"%PDF")
    (tmp_path / "empty.pdf.yaml").write_text("title: no keys\n")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF")
    (tmp_path / "bad.pdf.json").write_text("[1, 2]")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF")
    (tmp_path / "broken.pdf.json").write_text("{nope")
    (tmp_path / ".hidden.pdf").write_bytes(b"%PDF")
    found = {d.path.name: d for d in read_dropins(str(tmp_path))}
    assert sorted(found) == ["drill.pdf", "laser.pdf", "saw.pdf"]
    assert found["saw.pdf"].asset_keys == ["abcd1234", "efgh5678"]
    assert found["saw.pdf"].title == "Saw manual"
    assert found["saw.pdf"].url == "https://v.example.com/saw.pdf"
    assert found["laser.pdf"].asset_keys == ["laser123"]
    assert found["drill.pdf"].asset_keys == ["drill001", "drill002"]
    assert read_dropins(None) == []
    assert read_dropins(str(tmp_path / "missing")) == []
    assert read_sidecar(tmp_path / "nosidecar.pdf") is None
