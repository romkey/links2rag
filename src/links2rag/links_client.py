"""Read-only client for the asset manager's JSON API (pdxhackerspace-links).

Only ``GET /api/v1/things`` (paginated) and ``GET /api/v1/things/:id`` are
used, with a bearer API key. The key should belong to a dedicated key
created for this ingester under Settings -> API keys.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Any: API responses are JSON.
JSON = dict[str, Any]

_RETRIES = 3
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class LinksError(RuntimeError):
    """The asset manager refused a request or returned something unusable."""


class LinksClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        page_size: int = 500,
        transport: httpx.BaseTransport | None = None,
        retry_delay: float = 2.0,
    ) -> None:
        self.page_size = page_size
        self._retry_delay = retry_delay
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, str | int] | None = None) -> JSON:
        for attempt in range(1, _RETRIES + 1):
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as e:
                if attempt == _RETRIES:
                    raise LinksError(f"GET {path} failed: {e}") from e
                logger.warning("GET %s failed (%s); retrying", path, e)
                time.sleep(self._retry_delay * attempt)
                continue
            if response.status_code in _RETRY_STATUS and attempt < _RETRIES:
                logger.warning("GET %s returned %d; retrying", path, response.status_code)
                time.sleep(self._retry_delay * attempt)
                continue
            if response.status_code == 401:
                raise LinksError(f"GET {path}: 401, check LINKS_API_KEY")
            if response.status_code >= 400:
                raise LinksError(f"GET {path} returned {response.status_code}")
            try:
                body = response.json()
            except ValueError as e:
                raise LinksError(f"GET {path} returned invalid JSON") from e
            if not isinstance(body, dict):
                raise LinksError(f"GET {path} returned {type(body).__name__}, not an object")
            return body
        raise LinksError(f"GET {path} failed")  # pragma: no cover - the loop always returns

    def list_things(self) -> list[JSON]:
        """Every thing, all pages, in id order."""
        things: list[JSON] = []
        page: int | None = 1
        while page:
            body = self._get("/api/v1/things", {"page": page, "per_page": self.page_size})
            batch = body.get("things")
            if not isinstance(batch, list):
                raise LinksError("GET /api/v1/things returned no 'things' list")
            things.extend(t for t in batch if isinstance(t, dict))
            pagination = body.get("pagination") or {}
            next_page = pagination.get("next")
            page = int(next_page) if next_page else None
        return things

    def get_thing(self, thing_id: str) -> JSON:
        body = self._get(f"/api/v1/things/{thing_id}")
        thing = body.get("thing")
        if not isinstance(thing, dict):
            raise LinksError(f"GET /api/v1/things/{thing_id} returned no 'thing'")
        return thing
