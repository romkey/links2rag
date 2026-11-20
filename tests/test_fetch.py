from collections.abc import Callable

import httpx
import pytest

from links2rag.fetch import (
    MAX_DISCOVERED,
    Fetcher,
    _resolve,
    is_downloads_page,
    normalize_url,
    pdf_links,
    site_of,
    sniff,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _fetcher(
    handler: Handler,
    *,
    allow_private: bool = False,
    max_bytes: int = 10_000,
    addresses: dict[str, list[str]] | None = None,
) -> Fetcher:
    table = addresses or {}

    def resolver(host: str) -> list[str]:
        if host == "nowhere.example.com":
            raise OSError("no such host")
        return table.get(host, ["93.184.216.34"])

    return Fetcher(
        skip_hosts=frozenset({"links.example.com", "wiki.example.com"}),
        allow_private=allow_private,
        max_bytes=max_bytes,
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
        resolver=resolver,
    )


def _ok(body: bytes, content_type: str = "application/pdf") -> Handler:
    return lambda request: httpx.Response(200, content=body, headers={"content-type": content_type})


def test_fetch_pdf_and_html() -> None:
    fetcher = _fetcher(_ok(b"%PDF-1.4 stuff"))
    result = fetcher.fetch("https://vendor.example.com/m.pdf")
    assert (result.status, result.kind, result.http_status) == ("ok", "pdf", 200)
    assert result.content == b"%PDF-1.4 stuff"
    html = _fetcher(_ok(b"<!DOCTYPE html><p>x</p>", "text/html; charset=utf-8"))
    result = html.fetch("https://vendor.example.com/m")
    assert (result.kind, result.content_type) == ("html", "text/html")
    other = _fetcher(_ok(b"PK\x03\x04", "application/zip")).fetch("https://vendor.example.com/z")
    assert other.status == "unsupported"
    fetcher.close()


def test_redirects_are_checked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "/b"})
        if request.url.path == "/b":
            return httpx.Response(302, headers={"location": "https://links.example.com/secret"})
        return httpx.Response(200, content=b"%PDF")

    result = _fetcher(handler).fetch("https://vendor.example.com/a")
    assert result.status == "blocked"
    assert result.final_url == "https://links.example.com/secret"

    def loop(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    assert _fetcher(loop).fetch("https://vendor.example.com/x").error == "too many redirects"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("ftp://vendor.example.com/m.pdf", "scheme ftp"),
        ("https:///m.pdf", "no host"),
        ("https://user:pw@vendor.example.com/m.pdf", "credentials"),
        ("https://wiki.example.com/page", "never fetched"),
        ("https://team.slack.com/archives/C1", "never fetched"),
        ("https://intranet.example.com/m.pdf", "non-public"),
        ("http://10.0.0.5/m.pdf", "non-public"),
        ("https://nowhere.example.com/m.pdf", "cannot resolve"),
    ],
)
def test_blocked(url: str, reason: str) -> None:
    fetcher = _fetcher(_ok(b"%PDF"), addresses={"intranet.example.com": ["192.168.1.9"]})
    assert reason in (fetcher.blocked_reason(url) or "")


def test_unresolvable_host_is_dead_and_private_hosts_can_be_allowed() -> None:
    assert _fetcher(_ok(b"%PDF")).fetch("https://nowhere.example.com/m.pdf").status == "dead"
    fetcher = _fetcher(_ok(b"%PDF"), allow_private=True)
    assert fetcher.blocked_reason("http://10.0.0.5/m.pdf") is None


def test_http_errors_and_size_limits() -> None:
    assert _fetcher(lambda r: httpx.Response(404)).fetch("https://v.example.com/a").status == "dead"
    result = _fetcher(lambda r: httpx.Response(500)).fetch("https://v.example.com/a")
    assert (result.status, result.http_status) == ("error", 500)
    big = _fetcher(_ok(b"x" * 20_000)).fetch("https://v.example.com/a")
    assert big.status == "too_large"
    declared = _fetcher(
        lambda r: httpx.Response(200, headers={"content-length": "999999"}, content=b"%PDF")
    ).fetch("https://v.example.com/a")
    assert declared.status == "too_large"

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    assert _fetcher(boom).fetch("https://v.example.com/a").status == "error"


def test_helpers() -> None:
    assert normalize_url(" HTTPS://Vendor.Example.com/A.pdf#page=2 ") == (
        "https://vendor.example.com/A.pdf"
    )
    assert site_of("downloads.vendor.example.com.") == "example.com"
    assert sniff(b"  %PDF-1.7", None) == "pdf"
    assert sniff(b"<html>", None) == "html"
    assert sniff(b"data", "application/pdf") == "pdf"
    assert sniff(b"data", None) == "other"
    assert _resolve("localhost")


def test_pdf_links_and_downloads_pages() -> None:
    html = b"""<html><body>
      <a href="/files/manual.pdf">User manual</a>
      <a href="/files/brochure.pdf">Brochure</a>
      <a href="https://cdn.vendor.example.com/spec.PDF">Datasheet</a>
      <a href="https://other.org/manual.pdf">Elsewhere</a>
      <a href="/page.html">Page</a>
      <a href="/files/manual.pdf#p2">Again</a>
    </body></html>"""
    base = "https://www.vendor.example.com/downloads"

    def match(text: str) -> bool:
        return "manual" in text.lower() or "datasheet" in text.lower()

    every = pdf_links(html, base, pattern_match=match, require_match=False)
    assert [u for u, _ in every] == [
        "https://www.vendor.example.com/files/manual.pdf",
        "https://www.vendor.example.com/files/brochure.pdf",
        "https://cdn.vendor.example.com/spec.PDF",
    ]
    matching = pdf_links(html, base, pattern_match=match, require_match=True)
    assert [t for _, t in matching] == ["User manual", "Datasheet"]
    assert is_downloads_page(html, every)
    assert not is_downloads_page(html, [])
    long_page = b"<p>" + b"word " * 3000 + b"</p>"
    assert not is_downloads_page(long_page, every)
    many = b"".join(b'<a href="/f%d.pdf">x</a>' % n for n in range(20))
    assert len(pdf_links(many, base, pattern_match=match, require_match=False)) == MAX_DISCOVERED
