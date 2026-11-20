"""Downloading manuals: host checks, size limits, file type, downloads pages.

Only http(s) URLs on hosts that resolve to public addresses are fetched
(unless ALLOW_PRIVATE_HOSTS), never the asset manager, geowiki, Slack or a
host in SKIP_LINK_HOSTS, and every redirect is checked the same way. No
credentials or cookies are ever sent.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag

from links2rag import __version__

logger = logging.getLogger(__name__)

USER_AGENT = f"links2rag/{__version__} (+https://github.com/romkey/links2rag)"
_MAX_REDIRECTS = 5
# A downloads page links to the files; a manual written as HTML has text.
_DOWNLOADS_PAGE_MAX_TEXT = 6000
MAX_DISCOVERED = 5
# Never fetched whatever the settings say: other ingesters index Slack.
_ALWAYS_SKIP = ("slack.com",)
_DEAD = frozenset({404, 410})

Kind = Literal["pdf", "html", "other"]
Status = Literal["ok", "dead", "error", "blocked", "too_large", "unsupported"]
Resolver = Callable[[str], list[str]]


@dataclass(frozen=True)
class Fetched:
    status: Status
    url: str
    final_url: str | None = None
    http_status: int | None = None
    kind: Kind | None = None
    content_type: str | None = None
    content: bytes = b""
    error: str | None = None


def _resolve(host: str) -> list[str]:
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]


def normalize_url(url: str) -> str:
    """Drop the fragment and lowercase the scheme and host, for deduplication."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def _host_matches(host: str, names: frozenset[str] | tuple[str, ...]) -> bool:
    return any(host == name or host.endswith("." + name) for name in names)


def site_of(host: str) -> str:
    """The last two labels of a host name, a cheap stand-in for its registrable domain."""
    labels = host.lower().rstrip(".").split(".")
    return ".".join(labels[-2:])


def sniff(content: bytes, content_type: str | None) -> Kind:
    head = content[:1024].lstrip()
    if head.startswith(b"%PDF") or (content_type or "").startswith("application/pdf"):
        return "pdf"
    lowered = head[:512].lower()
    if (content_type or "").startswith(("text/html", "application/xhtml")) or (
        lowered.startswith((b"<!doctype html", b"<html"))
    ):
        return "html"
    return "other"


class Fetcher:
    def __init__(
        self,
        *,
        skip_hosts: frozenset[str],
        allow_private: bool,
        max_bytes: int,
        timeout_seconds: int,
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver = _resolve,
    ) -> None:
        self.skip_hosts = skip_hosts
        self.allow_private = allow_private
        self.max_bytes = max_bytes
        self._resolver = resolver
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,text/html;q=0.9,*/*"},
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def blocked_reason(self, url: str) -> str | None:
        """Why a URL may not be fetched, or None if it may."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return f"scheme {parts.scheme or '(none)'} is not http(s)"
        host = (parts.hostname or "").lower()
        if not host:
            return "no host"
        if parts.username or parts.password:
            return "URL carries credentials"
        if _host_matches(host, self.skip_hosts) or _host_matches(host, _ALWAYS_SKIP):
            return f"{host} is never fetched"
        if self.allow_private:
            return None
        try:
            addresses = [host] if _is_ip(host) else self._resolver(host)
        except OSError as e:
            return f"cannot resolve {host}: {e}"
        for address in addresses:
            if not ipaddress.ip_address(address.split("%")[0]).is_global:
                return f"{host} resolves to a non-public address"
        return None

    def _peer_reason(self, response: httpx.Response) -> str | None:
        """Why the connected address is not allowed, or None."""
        if self.allow_private:
            return None
        stream = response.extensions.get("network_stream")
        addr = stream.get_extra_info("server_addr") if stream is not None else None
        if not addr:
            return None
        try:
            address = ipaddress.ip_address(str(addr[0]).split("%")[0])
        except ValueError:
            return None
        if not address.is_global:
            return f"connected to non-public address {address}"
        return None

    def fetch(self, url: str) -> Fetched:
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            reason = self.blocked_reason(current)
            if reason:
                status: Status = "dead" if reason.startswith("cannot resolve") else "blocked"
                return Fetched(status, url, final_url=current, error=reason)
            try:
                with self._client.stream("GET", current) as response:
                    # The name is resolved again to connect; a host that
                    # answered with a public address for the check could
                    # answer differently now (DNS rebinding). Check the
                    # address actually reached before reading anything.
                    peer = self._peer_reason(response)
                    if peer:
                        return Fetched("blocked", url, final_url=current, error=peer)
                    if response.is_redirect and "location" in response.headers:
                        current = urljoin(current, response.headers["location"])
                        continue
                    content_type = response.headers.get("content-type", "").split(";")[0].strip()
                    if response.status_code >= 400:
                        return Fetched(
                            "dead" if response.status_code in _DEAD else "error",
                            url,
                            final_url=current,
                            http_status=response.status_code,
                            error=f"HTTP {response.status_code}",
                        )
                    declared = int(response.headers.get("content-length") or 0)
                    if declared > self.max_bytes:
                        return Fetched("too_large", url, final_url=current, error="too large")
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            return Fetched("too_large", url, final_url=current, error="too large")
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    kind = sniff(content, content_type)
                    return Fetched(
                        "ok" if kind != "other" else "unsupported",
                        url,
                        final_url=current,
                        http_status=response.status_code,
                        kind=kind,
                        content_type=content_type or None,
                        content=content,
                        error=None if kind != "other" else f"unsupported type {content_type}",
                    )
            except httpx.HTTPError as e:
                return Fetched("error", url, final_url=current, error=f"{type(e).__name__}: {e}")
        return Fetched("error", url, final_url=current, error="too many redirects")


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def page_text_length(html: bytes) -> int:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(("script", "style", "nav", "header", "footer")):
        tag.decompose()
    return len(soup.get_text(" ", strip=True))


def pdf_links(
    html: bytes, base_url: str, *, pattern_match: Callable[[str], bool], require_match: bool
) -> list[tuple[str, str]]:
    """Same-site PDF links on a page as (url, link text), at most MAX_DISCOVERED.

    With ``require_match``, only links whose text or URL matches the manual
    pattern (manufacturer pages link to brochures and price lists too).
    """
    soup = BeautifulSoup(html, "html.parser")
    site = site_of(urlsplit(base_url).hostname or "")
    found: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        if not isinstance(anchor, Tag):
            continue
        href = urljoin(base_url, str(anchor.get("href")))
        parts = urlsplit(href)
        if not parts.path.lower().endswith(".pdf"):
            continue
        if site_of(parts.hostname or "") != site:
            continue
        text = anchor.get_text(" ", strip=True)
        # The file name without ".pdf": every link here is a PDF, so a pattern
        # that matches ".pdf" would match them all.
        stem = parts.path.rsplit("/", 1)[-1][:-4].replace("-", " ").replace("_", " ")
        if require_match and not (pattern_match(text) or pattern_match(stem)):
            continue
        found.setdefault(normalize_url(href), text)
        if len(found) >= MAX_DISCOVERED:
            break
    return list(found.items())


def is_downloads_page(html: bytes, links: list[tuple[str, str]]) -> bool:
    """A page that links to manual files and has little text of its own."""
    return bool(links) and page_text_length(html) <= _DOWNLOADS_PAGE_MAX_TEXT
